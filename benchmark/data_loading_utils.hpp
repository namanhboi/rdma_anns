#include <cstring>
#include <immintrin.h> // needed to include this to make sure that the code compiles since in DiskANN/include/utils.h it uses this library.
#include "abstract_data_store.h"
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
#include "benchmark_dataset.hpp"
#include "../src/udl_path_and_index.hpp"
#include <filesystem>

#define NUM_THREADS 16

#define HEAD_INDEX_R 32
#define HEAD_INDEX_L 50
#define HEAD_INDEX_ALPHA 1.2
#define HEAD_INDEX_PERCENTAGE 0.01



// for now, this will be in memory
#define WHOLE_GRAPH_SUBGROUP_INDEX 1
#define HEAD_INDEX_SUBGROUP_INDEX 0
#define PQ_SUBGROUP_INDEX 2
// the pq subgroup will live in the same nodes as the whole graph subgroup
// need to make sure that this is the case in the layout.json file


#define MAX_DEGREE 64
// will need to revisit this later
#define GRAPH_CLUSTER_PREFIX "/anns/cluster_"
#define HEAD_INDEX_PREFIX "/anns/head_index"
#define PQ_PREFIX "/anns/pq"
// For < 10 billion vectors: all servers are members of the pq object pool only
// shard, meaning they will all have copies of the pq of all vectors
// For > 10 billion, then we can randomly shard the pq among the pg subgroup. 

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
  // for now, 
  for (int i = 0; i < clusters.size(); i++) {
    std::string cluster_folder = GRAPH_CLUSTER_PREFIX + std::to_string(i);
    capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(
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

template<typename data_type>
uint32_t get_num_nodes_head_index(
				  const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex) {
  return _pFlashIndex->get_num_points() * HEAD_INDEX_PERCENTAGE;
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
			       int R, int L, float alpha,
			       const std::string &in_mem_index_path) {
  uint32_t num_nodes_to_cache = get_num_nodes_head_index(_pFlashIndex);
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
    size_t end_idx = (std::min)(static_cast<size_t>(num_nodes_to_cache), (block + 1) * BLOCK_SIZE);

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
          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
          .with_dimension(_pFlashIndex->get_data_dim())
          .with_max_points(node_list.size())
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

  std::unique_ptr<diskann::Distance<data_type>> dist;
  dist.reset((diskann::Distance<data_type> *)diskann::get_distance_function<data_type>(diskann::Metric::L2));
  std::shared_ptr<diskann::InMemDataStore<data_type>> data_store =
      std::make_shared<diskann::InMemDataStore<data_type>>(
          _pFlashIndex->get_num_points(), _pFlashIndex->get_data_dim(),
							   std::move(dist));
  data_store->populate_data(_coord_cache_buf, node_list.size());
  std::cout <<"distance between node 120 and node 7 is " << data_store->get_distance(120, 7) << std::endl;
  std::cout << "distance between node 999 and node 0 is " << data_store->get_distance(999, 0) << std::endl;
  std::string query_file = "/home/nam/workspace/rdma_anns/extern/DiskANN/build/"
                           "data/sift/sift_query.fbin";

  data_type *query_data;
  size_t query_num;
  size_t query_dim;
  size_t query_aligned_dim;
  diskann::load_aligned_bin(query_file, query_data, query_num, query_dim,
                            query_aligned_dim);
  std::vector<uint32_t> search_id_results(10);
  index->search(query_data, 10, 20, search_id_results.data());
  for (auto id : search_id_results) {
    std::cout << "result " << id << std::endl;
  }  
  save_head_index_node_indices(node_list, in_mem_index_path);
}



template <typename data_type>
void load_diskann_in_mem_index(ServiceClientAPI &capi, const std::string &in_mem_index_path) {
  std::ifstream data_store_in;
  data_store_in.exceptions(std::ios::badbit | std::ios::failbit);
  data_store_in.open(in_mem_index_path + ".data", std::ios::binary);
  data_store_in.seekg(0, data_store_in.beg);


  uint32_t num_pts, dim;
  data_store_in.read((char *)&num_pts, sizeof(uint32_t));
  data_store_in.read((char *)&dim, sizeof(uint32_t));
  data_store_in.close();

  std::unique_ptr<diskann::Distance<data_type>> dist;
  dist.reset((diskann::Distance<data_type> *)diskann::get_distance_function<data_type>(diskann::Metric::L2));
  diskann::InMemDataStore<data_type> data_store(
						num_pts, dim, std::move(dist));
  data_store.load(in_mem_index_path + ".data");

  std::ifstream graph_store_in;
  graph_store_in.exceptions(std::ios::badbit | std::ios::failbit);
  graph_store_in.open(in_mem_index_path, std::ios::binary);
  graph_store_in.seekg(0, data_store_in.beg);

  uint64_t graph_expected_file_size;
  uint32_t max_deg;
  graph_store_in.read((char *)&graph_expected_file_size, sizeof(uint64_t));
  graph_store_in.read((char *)&max_deg, sizeof(uint32_t));
  graph_store_in.close();

  
  diskann::InMemGraphStore graph_store(num_pts, max_deg);
  auto [nodes_read, start, num_frozen_points] =
    graph_store.load(in_mem_index_path, num_pts);

  capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(
								      HEAD_INDEX_PREFIX, HEAD_INDEX_SUBGROUP_INDEX);
  uint32_t header[] = {num_pts, dim, max_deg};
  ObjectWithStringKey in_mem_graph_header;
  in_mem_graph_header.key = HEAD_INDEX_PREFIX "/header";
  in_mem_graph_header.blob = Blob(reinterpret_cast<const uint8_t*>(header), sizeof(uint32_t) * 3);

  capi.put_and_forget(in_mem_graph_header, false);

  parlay::parallel_for(0, num_pts, [&] (size_t i) {
  // for (uint32_t i = 0; i < num_pts; i++) {
    std::vector<uint32_t> nbrs = graph_store.get_neighbours(i);
    ObjectWithStringKey nbr_i;
    nbr_i.key = HEAD_INDEX_PREFIX "/nbr_" + std::to_string(i);
    // nbr_i.blob = Blob(
        // [&](uint8_t *buffer, const std::size_t size) {
          // uint32_t num_nbrs = nbrs.size();
          // std::memcpy(buffer, &num_nbrs, sizeof(uint32_t));
          // std::memcpy(buffer, nbrs.data(), sizeof(uint32_t) * num_nbrs);
          // return size;
        // },
		      // sizeof(uint32_t) * (nbrs.size() + 1));
    
    
    nbr_i.blob = Blob(reinterpret_cast<const uint8_t *>(nbrs.data()),
                      nbrs.size() * sizeof(uint32_t));
    nbr_i.previous_version = INVALID_VERSION;
    nbr_i.previous_version_by_key = INVALID_VERSION;

    capi.put_and_forget(nbr_i, false);
    
    data_type vector_data[dim];
    data_store.get_vector(i, vector_data);
    ObjectWithStringKey data_i;
    data_i.key = HEAD_INDEX_PREFIX "/emb_" + std::to_string(i);
    data_i.blob = Blob(reinterpret_cast<const uint8_t *>(vector_data),
                       dim * sizeof(data_type));
    data_i.previous_version = INVALID_VERSION;
    data_i.previous_version_by_key = INVALID_VERSION;
    capi.put_and_forget(data_i, false);
  });
}


/**
   Deprecated
*/
template <typename data_type>
void load_diskann_head_index_into_cascade(
    ServiceClientAPI &capi, const std::string &in_mem_index_path) {
  // capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(
								      // HEAD_INDEX_PREFIX, HEAD_INDEX_SUBGROUP_INDEX);
  
  std::cout << in_mem_index_path << std::endl;
  std::ifstream data_store_in;
  data_store_in.exceptions(std::ios::badbit | std::ios::failbit);
  std::cout << "hello done creating data store obj" << std::endl;
  data_store_in.open(in_mem_index_path + ".data", std::ios::binary);
  std::cout << "done opening " << in_mem_index_path + ".data";
  data_store_in.seekg(0, data_store_in.end);
  std::cout << "seeking";
  size_t expected_file_size_data = data_store_in.tellg();
  data_store_in.seekg(0, data_store_in.beg);
  uint8_t* data_store_data = new uint8_t[expected_file_size_data];
  data_store_in.read((char *)data_store_data, expected_file_size_data);
  std::cout << "Successfully read the data_store file with "
  << expected_file_size_data << " bytes" << std::endl;
  uint32_t num_pts, dim;
  std::memcpy(&num_pts, data_store_data, sizeof(uint32_t));
  std::memcpy(&dim, data_store_data + sizeof(uint32_t), sizeof(uint32_t));

    capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(
								      HEAD_INDEX_PREFIX, HEAD_INDEX_SUBGROUP_INDEX);
  ObjectWithStringKey data_store_obj;
  data_store_obj.key = HEAD_INDEX_PREFIX "/data_store";
  data_store_obj.blob = Blob(data_store_data, expected_file_size_data);
  data_store_obj.previous_version = INVALID_VERSION;
  data_store_obj.previous_version_by_key = INVALID_VERSION;

  auto data_store_put_res = capi.put(data_store_obj, false);
  for (auto &reply_future : data_store_put_res.get()) {
    auto reply = reply_future.second.get();
  }

  std::cout << "Successfully put the data_store file " << std::endl;

  std::ifstream graph_store_in;
  graph_store_in.exceptions(std::ios::badbit | std::ios::failbit);
  graph_store_in.open(in_mem_index_path, std::ios::binary);
  std::cout << "done reading the graph streo" ;
  graph_store_in.seekg(0, graph_store_in.beg);
  size_t expected_file_size;
  graph_store_in.read((char *)&expected_file_size, sizeof(size_t));
  uint8_t *graph_store_data =
      new uint8_t[expected_file_size - sizeof(size_t) +
                  sizeof(uint32_t)]; // don't include the
                                     // expected_file_size in this buffer,
                                     // replace it with the number of pts
  
  
  std::memcpy(graph_store_data, &num_pts, sizeof(uint32_t));
  graph_store_in.read((char *)(graph_store_data + sizeof(uint32_t)), expected_file_size - sizeof(size_t));
  
  ObjectWithStringKey graph_store_obj;
  graph_store_obj.key = HEAD_INDEX_PREFIX "/graph_store";
  graph_store_obj.blob = Blob(graph_store_data, expected_file_size);
  graph_store_obj.previous_version = INVALID_VERSION;
  graph_store_obj.previous_version_by_key = INVALID_VERSION;

  auto graph_store_put_res = capi.put(graph_store_obj, false);
  for (auto &reply_future : graph_store_put_res.get()) {
    auto reply = reply_future.second.get();
  }
  std::cout << "Successfully put the graph_store file " << std::endl;
  delete[] data_store_data;
  delete[] graph_store_data;
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




template<typename data_type>
void load_diskann_pq_into_cascade(
    ServiceClientAPI &capi,
    const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex) {
  capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(PQ_PREFIX
								      , PQ_SUBGROUP_INDEX);  
  parlay::parallel_for(0, _pFlashIndex->get_num_points(), [&](size_t i) {
    std::vector<uint8_t> pq_vector = _pFlashIndex->get_pq_vector(i);
    ObjectWithStringKey pq_vector_obj;
    pq_vector_obj.key = PQ_PREFIX "/vector_" + std::to_string(i);
    pq_vector_obj.previous_version = INVALID_VERSION;
    pq_vector_obj.previous_version_by_key = INVALID_VERSION;

    pq_vector_obj.blob =
        Blob(reinterpret_cast<const uint8_t *>(pq_vector.data()),
             pq_vector.size() * sizeof(uint8_t));
    
  
    auto result = capi.put(pq_vector_obj, false);
    for (auto &reply_future : result.get()) {
      auto reply = reply_future.second.get();
    }
  });
}



template <typename data_type>
void run_queries_head_index(std::unique_ptr<diskann::Index<data_type>> index,
                        const std::string &query_file, const std::string &gt_file) {
  int k = 10;
  int l = 100;
  BenchmarkDataset<data_type> dataset(query_file, gt_file);
  std::vector<uint32_t> query_res_ids;
  query_res_ids.resize(k * dataset.get_num_queries());

  for (int i = 0; i < dataset.get_num_queries(); i++) {
    index->search(dataset.get_query(i), k, l, query_res_ids.data() + i * k);
  }

  double recall = diskann::calculate_recall(
      dataset.get_num_queries(), dataset.gt_ids, dataset.gt_dists,
					    dataset.gt_dim, query_res_ids.data(), k, k);

  std::cout << "recall is " << recall << std::endl;
}

template <typename data_type>
std::unique_ptr<diskann::Index<data_type>>
get_index(const std::string &in_mem_index_path) {
  std::cout << "halo" << std::endl;
  std::ifstream data_store_in;
  std::cout << "wtf to  read the data_store file with ";
  data_store_in.exceptions(std::ios::badbit | std::ios::failbit);
  std::cout << "try to  read the data_store file with ";
  data_store_in.open(in_mem_index_path + ".data");
  std::cout << "suc open the data_store file with ";
  data_store_in.seekg(0, data_store_in.end);
  size_t expected_file_size_data = data_store_in.tellg();
  data_store_in.seekg(0, data_store_in.beg);
  uint8_t* data_store_data = new uint8_t[expected_file_size_data];
  data_store_in.read((char *)data_store_data, expected_file_size_data);
  std::cout << "Successfully read the data_store file with "
  << expected_file_size_data << " bytes" << std::endl;
  uint32_t num_pts, dim;
  std::memcpy(&num_pts, data_store_data, sizeof(uint32_t));
  std::memcpy(&dim, data_store_data + sizeof(uint32_t), sizeof(uint32_t));
  std::ifstream graph_store_in;
  graph_store_in.exceptions(std::ios::badbit | std::ios::failbit);
  graph_store_in.open(in_mem_index_path, std::ios::binary);
  graph_store_in.seekg(0, graph_store_in.beg);
  size_t expected_file_size;
  graph_store_in.read((char *)&expected_file_size, sizeof(size_t));
  uint8_t *graph_store_data =
      new uint8_t[expected_file_size - sizeof(size_t) +
                  sizeof(uint32_t)]; // don't include the
                                     // expected_file_size in this buffer,
                                     // replace it with the number of pts
  
  
  std::memcpy(graph_store_data, &num_pts, sizeof(uint32_t));
  graph_store_in.read((char *)(graph_store_data + sizeof(uint32_t)), expected_file_size - sizeof(size_t));

  std::cout << "Successfully read the graph_store file with "
  << expected_file_size << " bytes" << std::endl;
  
  std::unique_ptr<diskann::InMemGraphStore> graph_store = std::make_unique<diskann::InMemGraphStore>(num_pts,
												     (size_t)32 );
  auto [nodes_read, start, num_frozen_points] =
    graph_store->load(in_mem_index_path, num_pts);

  std::unique_ptr<diskann::Distance<data_type>> dist;
  dist.reset((diskann::Distance<data_type> *)diskann::get_distance_function<data_type>(diskann::Metric::L2));
  std::shared_ptr<diskann::InMemDataStore<data_type>> data_store = std::make_shared<diskann::InMemDataStore<data_type>>(num_pts
						, dim, std::move(dist));
  data_store->load(in_mem_index_path + ".data");

  auto index_build_params = diskann::IndexWriteParametersBuilder(50, 32)
                                .with_filter_list_size(0)
                                .with_alpha(1.2)
                                .with_saturate_graph(false)
                                .with_num_threads(16)
                                .build();
  auto filter_params = diskann::IndexFilterParamsBuilder()
                           .with_universal_label("")
                           .with_label_file("")
                           .build();

  //bruh this caused a whole ass wild goose chase
  auto search_params = diskann::IndexSearchParams(20, 16);

  auto config =
      diskann::IndexConfigBuilder()
          .with_metric(diskann::Metric::L2)
          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
          .with_dimension(dim)
          .with_max_points(num_pts) 
          .is_dynamic_index(false)
          .is_enable_tags(false)
          .is_pq_dist_build(false)
          .is_use_opq(false)
          .is_filtered(false)
          .with_num_pq_chunks(0)
          .with_num_frozen_pts(0)
          .with_data_type(diskann_type_to_name<data_type>())
          .with_index_write_params(index_build_params)
          .with_index_search_params(search_params)
          .build();
  std::shared_ptr<diskann::AbstractDataStore<data_type>> pq_data_store = data_store;
  std::unique_ptr<diskann::Index<data_type>> head_index = std::make_unique<diskann::Index<data_type>>(config, data_store, std::move(graph_store), pq_data_store, 353);
  delete[] graph_store_data;
  delete[] data_store_data;
  return head_index;
}

template <typename data_type>
Clusters get_clusters_from_diskann_graph(const std::string &index_path_prefix,
                                         uint32_t num_clusters) {
  AdjGraph adj;
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<data_type>> _pFlashIndex(
								 new diskann::PQFlashIndex<data_type>(reader, diskann::Metric::L2));
  
  int res = _pFlashIndex->load(omp_get_num_procs(), index_path_prefix.c_str());
  if (res != 0)
    throw std::runtime_error("error loading diskann data, error: " +
                             std::to_string(res));

  convert_diskann_graph_to_adjgraph<data_type>(_pFlashIndex, adj, MAX_DEGREE);
  
  return get_clusters_from_adjgraph(adj, num_clusters);
}

/**
   DEPRECATED: write which nodes belong to each cluster.
*/
void write_cluster_bin_file(const Clusters &clusters,
                            const std::string &output_clusters);
/**
   parse file from write_cluster_bin_file

*/
std::vector<std::vector<uint32_t>> parse_cluster_bin_file(const std::string &cluster_file);


/**
   for each node, write which cluster it belongs to
*/
void write_cluster_assignment_bin_file(const Clusters &clusters,
                            const std::string &output_cluster_assignment);

std::vector<uint8_t>
parse_cluster_assignment_bin_file(const std::string& cluster_assignment_bin_file);



/**
   for each cluster, write all the nodes that belong to it in a file:
   cluster_i_nodes.bin
*/
void write_cluster_nodes_bin_file(const Clusters &clusters,
                                      const std::string &output_folder);


void write_cluster_data_folder(const Clusters &clusters,
                               const std::string &output_folder);




std::vector<uint32_t>
parse_cluster_nodes_bin_file(const std::string &cluster_nodes_bin_file);
