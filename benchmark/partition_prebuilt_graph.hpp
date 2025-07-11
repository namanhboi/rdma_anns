#include <cstring>
#include <immintrin.h> // needed to include this to make sure that the code compiles since in DiskANN/include/utils.h it uses this library.
#include "ann_exception.h"
#include "defaults.h"
#include "defs.h"
#include "distance.h"
#include "in_mem_data_store.h"
#include "in_mem_graph_store.h"
#include "index_build_params.h"
#include "index_factory.h"
#include "linux_aligned_file_reader.h"
#include "metis_io.h"
#include "parameters.h"
#include "partitioning.h"
#include "pq_flash_index.h"
#include <boost/program_options.hpp>
#include <cascade/service_client_api.hpp>
#include <fstream>
#include <iostream>
#include <libaio.h>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <utils/graph.h>
#include "index_config.h"
#include "utils.h"

#define HEAD_INDEX_R 32
#define HEAD_INDEX_L 50
#define HEAD_INDEX_ALPHA 1.2
#define HEAD_INDEX_PERCENTAGE 0.05


#define NUM_THREADS 16
#define WHOLE_GRAPH_SUBGROUP_INDEX 0
#define HEAD_INDEX_SUBGROUP_INDEX 1
#define MAX_DEGREE 32
#define GRAPH_CLUSTER_PREFIX "/anns/cluster_"
#define HEAD_INDEX_PREFIX "/anns/head_index"

using namespace derecho::cascade;
using namespace parlayANN;

AdjGraph convert_graph_to_adjgraph(Graph<unsigned int> &G);
Clusters get_clusters_from_adjgraph(AdjGraph &adj, int num_clusters);

/**
   This loads the prebuilt diskann graph on ssd into cascade according to the
   clusters.
   Each cluster is its own object pool and all of these object pools will be on
   the same subgroup. Hopefully they can each be a shard because of their
   affinity set regex.
 */
template <typename data_type>
void load_diskann_graph_into_cascade(
    ServiceClientAPI &capi,
    const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex,
    const Clusters &clusters, int max_degree) {
  int dimension = _pFlashIndex->get_data_dim();
  std::vector<int> non_max_neighbor_nodes;
  for (int i = 0; i < clusters.size(); i++) {
    std::string cluster_folder = GRAPH_CLUSTER_PREFIX + std::to_string(i);
    capi.template create_object_pool<PersistentCascadeStoreWithStringKey>(
        cluster_folder, WHOLE_GRAPH_SUBGROUP_INDEX, HASH, {}, "cluster[0-9]+");

    data_type *coord_ptr = new data_type[dimension * clusters[i].size()];
    std::vector<data_type *> tmp_coord_buffer;

    std::vector<std::pair<uint32_t, uint32_t *>> tmp_nbr_buffer;
    uint32_t *neighbors_ptr = new uint32_t[max_degree * clusters[i].size()];

    for (int j = 0; j < clusters[i].size(); j++) {
      tmp_coord_buffer.push_back(coord_ptr + j * dimension);
      tmp_nbr_buffer.emplace_back(0, neighbors_ptr + j * max_degree);
    }

    _pFlashIndex->read_nodes(
        clusters[i], tmp_coord_buffer,
        tmp_nbr_buffer); // after this, all data is in tmp_coord_buffer and
                         // tmp_nbr_buffer
    for (int j = 0; j < tmp_nbr_buffer.size(); j++) {
      if (tmp_nbr_buffer[j].first != 32)
        non_max_neighbor_nodes.push_back(j);
    }
    parlay::parallel_for(0, clusters[i].size(), [&](size_t j) {
      uint32_t vector_id = clusters[i][j];
      size_t num_byte_emb = sizeof(data_type) * dimension;
      size_t num_byte_neighbors =
          sizeof(unsigned int) * tmp_nbr_buffer[j].first;
      size_t num_byte_object = num_byte_emb + num_byte_neighbors;
      std::vector<std::byte> data(num_byte_object);
      std::memcpy(data.data(), tmp_coord_buffer[j], num_byte_emb);
      std::memcpy(data.data() + dimension * sizeof(data_type),
                  tmp_nbr_buffer[j].second, num_byte_neighbors);
      ObjectWithStringKey vector_data;
      vector_data.key = cluster_folder + "/vector_" + std::to_string(vector_id);
      vector_data.previous_version = INVALID_VERSION;
      vector_data.previous_version_by_key = INVALID_VERSION;
      vector_data.blob =
          Blob(reinterpret_cast<const uint8_t *>(data.data()), num_byte_object);
      auto result = capi.put(vector_data, false);
      for (auto &reply_future : result.get()) {
        auto reply = reply_future.second.get();
      }
    });
    std::cout << "Done with cluster " << i << "/" << clusters.size() - 1
              << std::endl;
    std::cout << "SAMPLE NEIGHBOR" << std::endl;
    for (int j = 0; j < tmp_nbr_buffer[10].first; j++) {
      std::cout << tmp_nbr_buffer[10].second[j] << std::endl;
    }
    delete[] coord_ptr;
    delete[] neighbors_ptr;
  }

  std::cout << "nodes with non max neighbors" << std::endl;
  for (const int &x : non_max_neighbor_nodes) {
    std::cout << x << std::endl;
  }
}

/**
   node_list is the list of nodes from the ssd index that shuld be saved in the
   head index

*/
void save_head_index_node_indices(std::vector<uint32_t> &node_list,
                                  const std::string &in_mem_index_path);




/**
   This function builds the head index from the current diskindex and then saves
   the head index to in_mem_index_path
*/
template <typename data_type>
void build_and_save_head_index(
    const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex,
			       uint64_t num_nodes_to_cache,
			       int R, int L, float alpha,
			       const std::string &in_mem_index_path) {
  std::vector<uint32_t> node_list;
  _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
  std::cout << "Finished Caching " << node_list.size();
  if (node_list.size() != num_nodes_to_cache) {
    throw std::runtime_error(
        "cached node list size diff from num_node_to_cache: " +
        std::to_string(node_list.size()) + " vs " +
        std::to_string(num_nodes_to_cache));
  }

  //
  uint64_t aligned_dim = ROUND_UP(
      _pFlashIndex->get_data_dim(),
      8); // round up the dimension to a multiple of 8, a lof of operations
          // indlucding index building requires that the allocated data for
          // coordinates of vectors need to be aligned this way
  

  data_type *_coord_cache_buf = nullptr; // this is where all the specified nodes' coordinate data will be read.
  // Allocate space for coordinate cache
  size_t coord_cache_buf_len = num_nodes_to_cache * aligned_dim;
  diskann::alloc_aligned((void **)&_coord_cache_buf,
                         coord_cache_buf_len * sizeof(data_type),
                         8 * sizeof(data_type));
  memset(_coord_cache_buf, 0, coord_cache_buf_len * sizeof(data_type));

  size_t BLOCK_SIZE = 8;
  size_t num_blocks = DIV_ROUND_UP(num_nodes_to_cache, BLOCK_SIZE);

  std::vector<size_t> failed_read_node_id;
  for (size_t block = 0; block < num_blocks; block++) {
    size_t start_idx = block * BLOCK_SIZE;
    size_t end_idx = (std::min)(num_nodes_to_cache, (block + 1) * BLOCK_SIZE);

    // Copy offset into buffers to read into
    std::vector<uint32_t> nodes_to_read;
    std::vector<data_type *> coord_buffers;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
    for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++) {
      nodes_to_read.push_back(node_list[node_idx]);
      coord_buffers.push_back(_coord_cache_buf + node_idx * aligned_dim);
      nbr_buffers.emplace_back(0, nullptr);
    }

    // issue the reads
    auto read_status = _pFlashIndex->read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

    for (size_t i = 0; i < read_status.size(); i++) {
      if (read_status[i] != true) {
        failed_read_node_id.push_back(node_list[start_idx + i]);
      }
    }
  }
  std::cout << "Failed to cache " << failed_read_node_id.size() << " nodes"
  << std::endl;
  if (failed_read_node_id.size() != 0) throw std::runtime_error("Failed to load " + std::to_string(failed_read_node_id.size()) + " nodes");

  // Once the data reading is done, _coord_cache_buf is the ptr to the coordinate data of the specified nodes.

  
  // now we build the head index with the data above.
  uint32_t filter_list_size = 0;

  auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                .with_filter_list_size(0)
                                .with_alpha(alpha)
                                .with_saturate_graph(false)
                                .with_num_threads(NUM_THREADS)
                                .build();
  auto filter_params = diskann::IndexFilterParamsBuilder()
                           .with_universal_label("")
                           .with_label_file("")
                           .with_save_path_prefix(in_mem_index_path)
                           .build();

  auto config =
      diskann::IndexConfigBuilder()
          .with_metric(diskann::Metric::L2)
          .with_dimension(_pFlashIndex->get_data_dim())
          .with_max_points(node_list.size())
          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
          .with_data_type(diskann_type_to_name<data_type>())
          .with_label_type("uint")
          .is_dynamic_index(false)
          .with_index_write_params(index_build_params)
          .is_enable_tags(false)
          .is_use_opq(false)
          .is_pq_dist_build(false)
          .with_num_pq_chunks(0)
          .build();
  auto index_factory = diskann::IndexFactory(config);
  auto index = index_factory.create_instance();
  std::vector<uint32_t> tags;
  index->build(_coord_cache_buf, node_list.size(), tags);
  index->save(in_mem_index_path.c_str());

  save_head_index_node_indices(node_list, in_mem_index_path);
}


/**
   this function loads the head index into cascade
*/
template <typename data_type>
void load_diskann_head_index_into_cascade(
    ServiceClientAPI &capi,
    const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex,
					  const std::string &in_mem_index_path,
					  int max_degree) {
  int num_points_head_index = _pFlashIndex->get_num_points() * HEAD_INDEX_PERCENTAGE;
  
  diskann::InMemGraphStore graph_store(num_points_head_index,
                                       max_degree);
  auto [nodes_read, start, num_frozen_points] =
    graph_store.load(in_mem_index_path, num_points_head_index);

  // if (nodes_read != _pFlashIndex->get_num_points()) {
    // throw std::runtime_error("number of nodes read not equal to num points " + std::to_string(nodes_read) + " , " + std::to_string(_pFlashIndex->get_num_points()));
  // }

  std::unique_ptr<diskann::Distance<data_type>> dist;
  dist.reset((diskann::Distance<data_type> *)diskann::get_distance_function<data_type>(diskann::Metric::L2));
  diskann::InMemDataStore<data_type> data_store(
						_pFlashIndex->get_num_points(), _pFlashIndex->get_data_dim(), std::move(dist));
  data_store.load(in_mem_index_path + ".data");


  uint32_t *node_list_data = nullptr;
  size_t npts, dim;
  diskann::load_bin<uint32_t>(in_mem_index_path + ".indices", node_list_data, npts, dim);
  if (dim != 1) throw std::runtime_error("The dimension for indices file doesn't make sense: " + std::to_string(dim));
  if (npts != num_points_head_index)
    throw std::runtime_error(
        "Number of points from graph and indices file is different " +
        std::to_string(num_points_head_index) + " " + std::to_string(npts));
  uint8_t *mapping = new uint8_t[npts * sizeof(uint32_t)];
  std::memcpy(mapping, node_list_data, npts * sizeof(uint32_t));
  

  // for (
  capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(
								      HEAD_INDEX_PREFIX, HEAD_INDEX_SUBGROUP_INDEX);
  
  for (int i = 0; i < npts; i++) {
    data_type vector_emb[_pFlashIndex->get_data_dim()];
    data_store.get_vector(i, vector_emb);

    auto neighbors = graph_store.get_neighbours(i);

    size_t num_byte_emb = _pFlashIndex->get_data_dim() * sizeof(data_type);
    size_t num_byte_nbr = neighbors.size() * sizeof(uint32_t);
    size_t num_byte_obj = num_byte_emb + num_byte_nbr;
    std::vector<std::byte> data(num_byte_obj);
    std::memcpy(data.data(), vector_emb, num_byte_emb);
    std::memcpy(data.data() + num_byte_emb, neighbors.data(), num_byte_nbr);
    ObjectWithStringKey vector_data;
    vector_data.key = HEAD_INDEX_PREFIX "/vector_" + std::to_string(i);
    vector_data.previous_version = INVALID_VERSION;
    vector_data.previous_version_by_key = INVALID_VERSION;
    vector_data.blob =
      Blob(reinterpret_cast<const uint8_t *>(data.data()), num_byte_obj);
    auto result = capi.put(vector_data, false);
    for (auto &reply_future : result.get()) {
      auto reply = reply_future.second.get();
    }
  }

  ObjectWithStringKey head_index_mapping;
  head_index_mapping.key = HEAD_INDEX_PREFIX "/mapping";
  head_index_mapping.previous_version = INVALID_VERSION;
  head_index_mapping.previous_version_by_key = INVALID_VERSION;

  head_index_mapping.blob =
    Blob(reinterpret_cast<const uint8_t *>(mapping), npts * sizeof(uint32_t));
  
  auto result = capi.put(head_index_mapping, false);
  for (auto &reply_future : result.get()) {
    auto reply = reply_future.second.get();
  }
  delete[] mapping;
  delete[] node_list_data;
  
  // for (
  // 
  // capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(
								      // HEAD_INDEX, HEAD_INDEX_SUBGROUP_INDEX);
}

template <typename data_type>
void convert_diskann_graph_to_adjgraph(
    const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex,
    AdjGraph &adj, int max_degree) {
  adj =
      std::move(std::vector<std::vector<int>>(_pFlashIndex->get_num_points()));
  parlay::parallel_for(
      0, _pFlashIndex->get_num_points(), [&](size_t node_index) {
        data_type *coord_ptr = new data_type[_pFlashIndex->get_data_dim()];
        std::vector<data_type *> tmp_coord_buffer;
        tmp_coord_buffer.push_back(coord_ptr);
        std::vector<std::pair<uint32_t, uint32_t *>> tmp_nbr_buffer;

        uint32_t *neighbors_ptr = new uint32_t[max_degree];
        tmp_nbr_buffer.emplace_back(0, neighbors_ptr);

        _pFlashIndex->read_nodes(std::vector<uint32_t>(1, node_index),
                                 tmp_coord_buffer, tmp_nbr_buffer);
        for (int i = 0; i < tmp_nbr_buffer[0].first; i++) {
          adj[node_index].push_back(*(tmp_nbr_buffer[0].second + i));
        }

        delete[] coord_ptr;
        delete[] neighbors_ptr;
      });
}
