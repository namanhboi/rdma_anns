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
#include "partition.h"
#include "disk_utils.h"
#define NUM_THREADS 16

#define HEAD_INDEX_R 32
#define HEAD_INDEX_L 50
#define HEAD_INDEX_ALPHA 1.2
#define HEAD_INDEX_PERCENTAGE 0.01


#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))


#define MAX_DEGREE 32

using namespace derecho::cascade;
using namespace parlayANN;

AdjGraph convert_graph_to_adjgraph(Graph<unsigned int> &G);
Clusters get_clusters_from_adjgraph(AdjGraph &adj, int num_clusters);

/** create object pools for udl1 and udl2 */
void create_object_pools(ServiceClientAPI &capi);

template <typename data_type>
void test_pq_flash(const std::string &index_path_prefix, const std::string &query_file, const std::string & gt_file) {

  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<data_type>> _pFlashIndex(
								 new diskann::PQFlashIndex<data_type>(reader, diskann::Metric::L2));

  int _ = _pFlashIndex->load(1, index_path_prefix.c_str());
  BenchmarkDataset<data_type> dataset(query_file, gt_file);
  std::vector<uint64_t> results(10);
  _pFlashIndex->cached_beam_search(dataset.get_query(0), 10, 20, results.data(),
                                   nullptr, 1);
}




/**
   TODO
   create the index files for each cluster.
   To make it easily testable, use the same format as diskann, and then to test
   it, use 1 cluster and write data to new file. Use diskann to load that file
   and if recall is the same then we are correct.

   To use the already existing functions to build the disk index from
   disk_utils from diskann, for each cluster, we need to create a base file as
   well as a graph file. After this, for each pair of base and graph files, we
   do create_disk_layout to create the final disk file for that cluster
*/

template <typename data_type>
void create_cluster_index_files(const Clusters &clusters,
                                const std::string &data_file,
                                const std::string &index_path_prefix,
                                const std::string &output_folder) {


  if (!std::filesystem::exists(output_folder)) {
    throw std::invalid_argument(output_folder + " doesn't exists");
  }
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<data_type>> _pFlashIndex(
							       new diskann::PQFlashIndex<data_type>(reader, diskann::Metric::L2));
  int _ = _pFlashIndex->load(1, index_path_prefix.c_str());

  std::string disk_index_file = index_path_prefix + "_disk.index";
  std::ifstream index_metadata(disk_index_file, std::ios::binary);
  
  uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
  // metadata, nc should be 1)
  READ_U32(index_metadata, nr);
  READ_U32(index_metadata, nc);

  uint64_t disk_nnodes;
  uint64_t disk_ndims; 
  READ_U64(index_metadata, disk_nnodes);
  READ_U64(index_metadata, disk_ndims);


  size_t medoid_id_on_file, _max_node_len, _disk_bytes_per_point, _nnodes_per_sector;
  READ_U64(index_metadata, medoid_id_on_file);
  READ_U64(index_metadata, _max_node_len);
  READ_U64(index_metadata, _nnodes_per_sector);
  index_metadata.close();

  _disk_bytes_per_point = disk_ndims * sizeof(data_type);
  
  uint32_t max_degree = ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;
  std::cout << "max degree is " << max_degree << std::endl;

  parlay::parallel_for(0, clusters.size(), [&](size_t cluster_id) {
    std::string cluster_graph_file = output_folder + "/" + "cluster_" +
                             std::to_string(cluster_id) + ".graph";
    size_t file_offset = 0;
    std::ofstream graph_out;
    graph_out.open(cluster_graph_file, std::ios::binary | std::ios::trunc);
    graph_out.seekp(file_offset, graph_out.beg);
    
    // need to writer header
    size_t index_size = 24;
    graph_out.write((char *)&index_size, sizeof(uint64_t));
    graph_out.write((char *)&max_degree, sizeof(uint32_t));

    uint32_t start = 29429; // not sure about these 2 variables tbh
    size_t num_frozen_pts = 0;
    graph_out.write((char *)&start, sizeof(uint32_t));
    graph_out.write((char *)&num_frozen_pts, sizeof(size_t));

    std::vector<data_type *> tmp_coord_buffer(clusters[cluster_id].size(),
                                              nullptr);
    
    std::vector<std::pair<uint32_t, uint32_t *>> tmp_nbr_buffer;
    uint32_t *neighbors_ptr =
      new uint32_t[max_degree * clusters[cluster_id].size()];

    for (int j = 0; j < clusters[cluster_id].size(); j++) {
      tmp_nbr_buffer.emplace_back(0, neighbors_ptr + j * max_degree);
    }
    _pFlashIndex->read_nodes(clusters[cluster_id], tmp_coord_buffer,
                             tmp_nbr_buffer);

    uint32_t max_observed_degree = 0;
    for (uint32_t i = 0; i < clusters[cluster_id].size(); i++) {
      uint32_t nnbrs = tmp_nbr_buffer[i].first;
      graph_out.write((char *)&nnbrs, sizeof(nnbrs));
      graph_out.write((char *)tmp_nbr_buffer[i].second,
                      nnbrs * sizeof(uint32_t));
      index_size += (size_t)(sizeof(uint32_t) * (nnbrs + 1));
      max_observed_degree = nnbrs > max_observed_degree ? nnbrs : max_observed_degree;
    }
    std::cout << "max observed_degree " << max_observed_degree << std::endl;
    graph_out.seekp(file_offset, graph_out.beg);
    graph_out.write((char *)&index_size, sizeof(uint64_t));
    graph_out.write((char *)&max_observed_degree, sizeof(uint32_t));
    graph_out.close();

    // create data bin file

    std::string cluster_ids_file = output_folder + "/" + "cluster_" +
                                   std::to_string(cluster_id) +
                                   "_ids_uint32_t.bin";
    std::ofstream cluster_ids_out;
    cluster_ids_out.open(cluster_ids_file, std::ios::binary | std::ios::trunc);
      
    uint32_t num_points_cluster = clusters[cluster_id].size();
    uint32_t const_one = 1; // needed so that we can parse bin file with diskann::load_bin
    cluster_ids_out.write((char *)&num_points_cluster, sizeof(uint32_t));
    cluster_ids_out.write((char *)&const_one, sizeof(uint32_t));
    cluster_ids_out.write((char *)clusters[cluster_id].data(),
                          sizeof(uint32_t) * num_points_cluster);
    cluster_ids_out.close();


    std::string cluster_data_file =
      output_folder + "/" + "cluster_" + std::to_string(cluster_id) + ".bin";

    retrieve_shard_data_from_ids<data_type>(data_file, cluster_ids_file,
                                            cluster_data_file);
    
    // now create disk file for cluster
    std::string cluster_disk_file = output_folder + "/" + "cluster_" +
                                    std::to_string(cluster_id) + "_disk.index";
    diskann::create_disk_layout<data_type>(cluster_data_file, cluster_graph_file,
                                           cluster_disk_file);

    delete[] neighbors_ptr;
  });
  
}



/**
   Loads the pq data from file to cacsade, table file loaded by udl
*/
void load_diskann_pq_into_cascade(ServiceClientAPI &capi,
                                  const std::string &pq_compressed_vectors,
                                  const Clusters &clusters);

/**
   This loads the prebuilt diskann graph on ssd into cascade persistent kv store
   for in memory search according to the clusters.
   The embedding and neighbor ids of vector j of clusters i will be in
   /anns/global/data/cluster_i_vec_j
 */
template <typename data_type>
void load_diskann_graph_into_cascade_ssd(
    ServiceClientAPI &capi,
    const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex,
    const Clusters &clusters, int max_degree) {
  int dimension = _pFlashIndex->get_data_dim();
  std::vector<int> non_max_neighbor_nodes;

  for (int i = 0; i < clusters.size(); i++) {
    std::string cluster_folder = UDL2_DATA_PREFIX_CLUSTER + std::to_string(i);
    data_type *coord_ptr = new data_type[dimension * clusters[i].size()];
    std::vector<data_type *> tmp_coord_buffer;

    std::vector<std::pair<uint32_t, uint32_t *>> tmp_nbr_buffer;
    uint32_t *neighbors_ptr = new uint32_t[max_degree * clusters[i].size()];

    for (int j = 0; j < clusters[i].size(); j++) {
      tmp_coord_buffer.push_back(coord_ptr + j * dimension);
      tmp_nbr_buffer.emplace_back(0, neighbors_ptr + j * max_degree);
    }

    _pFlashIndex->read_nodes(clusters[i], tmp_coord_buffer, tmp_nbr_buffer);

    for (uint32_t i = 0; i < 100; i++) {
      for (uint32_t j = 0; j < dimension; j++) {
        std::cout << tmp_coord_buffer[i][j] << " " << std::endl;
      }
      std::cout << std::endl;
    }
    std::unique_ptr<diskann::Distance<data_type>> dist_fn(
        (diskann::Distance<data_type> *)
            diskann::get_distance_function<data_type>(diskann::Metric::L2));
    std::cout << "sample distance "
    << dist_fn->compare(tmp_coord_buffer[0], tmp_coord_buffer[1],
                                32);
    
    std::cout << "cluster folder is " << cluster_folder << std::endl;
    
    parlay::parallel_for(0, clusters[i].size(), [&](size_t j) {
      uint32_t vector_id = clusters[i][j];
      size_t nbr_blob_size =
        sizeof(uint32_t) * tmp_nbr_buffer[j].first + sizeof(uint32_t);
      size_t emb_blob_size = sizeof(data_type) * dimension;
      size_t total_size = emb_blob_size + nbr_blob_size;

      Blob vec_blob(
          [&](uint8_t *buffer, const std::size_t size) {
            uint32_t offset = 0;
            std::memcpy(buffer + offset, tmp_coord_buffer[j], emb_blob_size);
            offset += emb_blob_size;
            std::memcpy(buffer + offset, &tmp_nbr_buffer[j].first,
                        sizeof(tmp_nbr_buffer[j].first));
            offset += sizeof(tmp_nbr_buffer[j].first);
            std::memcpy(buffer + offset, tmp_nbr_buffer[j].second,
                        sizeof(uint32_t) * tmp_nbr_buffer[j].first);
            return size;
          },
		    total_size);
      ObjectWithStringKey vec_obj;
      vec_obj.key = cluster_folder + "_vec_" + std::to_string(vector_id);
      vec_obj.previous_version = INVALID_VERSION;
      vec_obj.previous_version_by_key = INVALID_VERSION;
      vec_obj.blob = std::move(vec_blob);
      capi.put_and_forget(vec_obj, false);
    });
    std::cout << "Done with cluster " << i << "/" << clusters.size() - 1
              << std::endl;
    // std::cout << "SAMPLE NEIGHBOR" << std::endl;
    // for (int j = 0; j < tmp_nbr_buffer[10].first; j++) {
      // std::cout << tmp_nbr_buffer[10].second[j] << std::endl;
    // }
    delete[] coord_ptr;
    delete[] neighbors_ptr;
  }

  // std::cout << "nodes with non max neighbors" << std::endl;
  // for (const int &x : non_max_neighbor_nodes) {
    // std::cout << x << std::endl;
  // }
}



/**
   This loads the prebuilt diskann graph on ssd into cascade volatile kv store
   for in memory search according to the clusters.
   The embedding of vector j of clusters i will be in
   /anns/global/data/cluster_i_emb_j
   The neighbors of vector j of cluster j will be in
   /anns/global/data/cluster_i_nbr_j
   - First uint32_t of neighbors blob is number of neighbors, rest is the neighbor ids
 */
template <typename data_type>
void load_diskann_graph_into_cascade_in_mem(
    ServiceClientAPI &capi,
    const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex,
    const Clusters &clusters, int max_degree) {
  int dimension = _pFlashIndex->get_data_dim();
  std::vector<int> non_max_neighbor_nodes;

  for (int i = 0; i < clusters.size(); i++) {
    std::string cluster_folder = UDL2_DATA_PREFIX_CLUSTER + std::to_string(i);
    data_type *coord_ptr = new data_type[dimension * clusters[i].size()];
    std::vector<data_type *> tmp_coord_buffer;

    std::vector<std::pair<uint32_t, uint32_t *>> tmp_nbr_buffer;
    uint32_t *neighbors_ptr = new uint32_t[max_degree * clusters[i].size()];

    for (int j = 0; j < clusters[i].size(); j++) {
      tmp_coord_buffer.push_back(coord_ptr + j * dimension);
      tmp_nbr_buffer.emplace_back(0, neighbors_ptr + j * max_degree);
    }

    _pFlashIndex->read_nodes(clusters[i], tmp_coord_buffer, tmp_nbr_buffer);

    for (uint32_t i = 0; i < 100; i++) {
      for (uint32_t j = 0; j < dimension; j++) {
        std::cout << tmp_coord_buffer[i][j] << " " << std::endl;
      }
      std::cout << std::endl;
    }
    std::unique_ptr<diskann::Distance<data_type>> dist_fn(
        (diskann::Distance<data_type> *)
            diskann::get_distance_function<data_type>(diskann::Metric::L2));
    std::cout << "sample distance "
    << dist_fn->compare(tmp_coord_buffer[0], tmp_coord_buffer[1],
                                32);
    
    std::cout << "cluster folder is " << cluster_folder << std::endl;
    



    parlay::parallel_for(0, clusters[i].size(), [&](size_t j) {
      uint32_t vector_id = clusters[i][j];
      size_t nbr_blob_size =
        sizeof(uint32_t) * tmp_nbr_buffer[j].first + sizeof(uint32_t);

      Blob nbr_blob(
          [&](uint8_t *buffer, const std::size_t size) {
            std::memcpy(buffer, &tmp_nbr_buffer[j].first,
                        sizeof(tmp_nbr_buffer[j].first));
            std::memcpy(buffer + sizeof(tmp_nbr_buffer[j].first), tmp_nbr_buffer[j].second,
                        sizeof(uint32_t) * tmp_nbr_buffer[j].first);
            return size;
          },
		    nbr_blob_size);
      ObjectWithStringKey nbr_obj;
      nbr_obj.key = cluster_folder + "_nbr_" + std::to_string(vector_id);
      nbr_obj.previous_version = INVALID_VERSION;
      nbr_obj.previous_version_by_key = INVALID_VERSION;
      nbr_obj.blob = std::move(nbr_blob);
      capi.put_and_forget(nbr_obj, false);

      size_t emb_blob_size = sizeof(data_type) * dimension;
      Blob emb_blob([&](uint8_t *buffer, const std::size_t size) {
        std::memcpy(buffer, tmp_coord_buffer[j], emb_blob_size);
        return size;
          },
		    emb_blob_size);
      ObjectWithStringKey emb_obj;
      emb_obj.key = cluster_folder + "_emb_" + std::to_string(vector_id);
      emb_obj.previous_version = INVALID_VERSION;
      emb_obj.previous_version_by_key = INVALID_VERSION;
      emb_obj.blob = std::move(emb_blob);
      capi.put_and_forget(emb_obj, false);

      // size_t num_byte_emb = sizeof(data_type) * dimension;
      // size_t num_byte_neighbors =
      //     sizeof(uint32_t) * tmp_nbr_buffer[j].first;
      // size_t num_byte_object = num_byte_emb + num_byte_neighbors;
      // std::vector<std::byte> data(num_byte_object);
      // std::memcpy(data.data(), tmp_coord_buffer[j], num_byte_emb);
      // std::memcpy(data.data() + dimension * sizeof(data_type),
      //             tmp_nbr_buffer[j].second, num_byte_neighbors);
      // ObjectWithStringKey vector_data;
      // vector_data.key = cluster_folder + "/vector_" + std::to_string(vector_id);
      // vector_data.previous_version = INVALID_VERSION;
      // vector_data.previous_version_by_key = INVALID_VERSION;
      // vector_data.blob =
      //     Blob(reinterpret_cast<const uint8_t *>(data.data()), num_byte_object);
      // auto result = capi.put(vector_data, false);
      // for (auto &reply_future : result.get()) {
      //   auto reply = reply_future.second.get();
      // }
    });
    std::cout << "Done with cluster " << i << "/" << clusters.size() - 1
              << std::endl;
    // std::cout << "SAMPLE NEIGHBOR" << std::endl;
    // for (int j = 0; j < tmp_nbr_buffer[10].first; j++) {
      // std::cout << tmp_nbr_buffer[10].second[j] << std::endl;
    // }
    delete[] coord_ptr;
    delete[] neighbors_ptr;
  }

  // std::cout << "nodes with non max neighbors" << std::endl;
  // for (const int &x : non_max_neighbor_nodes) {
    // std::cout << x << std::endl;
  // }
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
void build_and_save_head_index(const std::string &index_path_prefix,
                               const std::string &head_index_path, int R = HEAD_INDEX_R, int L = HEAD_INDEX_L,
                               float alpha = HEAD_INDEX_ALPHA) {
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<data_type>> _pFlashIndex(
								 new diskann::PQFlashIndex<data_type>(reader, diskann::Metric::L2));
  int _ = _pFlashIndex->load(1, index_path_prefix.c_str());

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
      8); // round up the dimension to a multiple of 8, a lof of opertaions
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
                           .with_save_path_prefix(head_index_path)
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
  index->save(head_index_path.c_str());
  save_head_index_node_indices(node_list, head_index_path);
}

// template <typename data_type>
// void convert_diskann_graph_to_adjgraph(
//     const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex,
//     AdjGraph &adj, int max_degree) {
//   adj = std::vector<std::vector<int>>(_pFlashIndex->get_num_points());
//   parlay::parallel_for(
//       0, _pFlashIndex->get_num_points(), [&](size_t node_index) {
//         data_type *coord_ptr = new data_type[_pFlashIndex->get_data_dim()];
//         std::vector<data_type *> tmp_coord_buffer;
//         tmp_coord_buffer.push_back(coord_ptr);
//         std::vector<std::pair<uint32_t, uint32_t *>> tmp_nbr_buffer;

//         uint32_t *neighbors_ptr = new uint32_t[max_degree];
//         tmp_nbr_buffer.emplace_back(0, neighbors_ptr);

//         _pFlashIndex->read_nodes(std::vector<uint32_t>(1, node_index),
//                                  tmp_coord_buffer, tmp_nbr_buffer);
//         for (int i = 0; i < tmp_nbr_buffer[0].first; i++) {
//           adj[node_index].push_back(*(tmp_nbr_buffer[0].second + i));
//         }

//         delete[] coord_ptr;
//         delete[] neighbors_ptr;
//       });
// }


// template <typename data_type>
// void convert_diskann_graph_to_adjgraph(
//     const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex,
//     AdjGraph &adj, int max_degree) {
    
//   adj = std::vector<std::vector<int>>(_pFlashIndex->get_num_points());
//   std::cout << "number of points is " << _pFlashIndex->get_num_points() << std::endl;
  
//   // Try to avoid read_nodes entirely
//   for (uint64_t node_id = 0; node_id < _pFlashIndex->get_num_points(); node_id++) {
    
//     // Check if DiskANN has other methods like:
//     // auto neighbors = _pFlashIndex->get_neighbors(node_id);
//     // or
//     // std::vector<uint32_t> neighbors;
//     // _pFlashIndex->get_node_neighbors(node_id, neighbors);
    
//     // If those don't exist, we need to debug the read_nodes call
//     std::vector<uint32_t> single_node = {static_cast<uint32_t>(node_id)};
    
//     // Minimal test - see if this crashes
//     try {
//       // Test with minimal allocations
//       std::vector<data_type*> coord_buf = {nullptr};
//       std::vector<std::pair<uint32_t, uint32_t*>> nbr_buf;
      
//       // Allocate single neighbor buffer
//       uint32_t *neighbors = (uint32_t*)calloc(max_degree, sizeof(uint32_t));
//       if (!neighbors) {
//         std::cerr << "Failed to allocate memory" << std::endl;
//         continue;
//       }
      
//       nbr_buf.emplace_back(0, neighbors);
      
//       _pFlashIndex->read_nodes(single_node, coord_buf, nbr_buf);
      
//       // If we get here without crashing, copy the data
//       adj[node_id].assign(neighbors, neighbors + nbr_buf[0].first);
      
//       free(neighbors);
      
//     } catch (const std::exception& e) {
//       std::cerr << "Failed to read node " << node_id << ": " << e.what() << std::endl;
//       // Continue with empty adjacency list
//     }
    
//     if (node_id % 10000 == 0) {
//       std::cout << "Processed " << node_id << " nodes" << std::endl;
//     }
//   }
// }

template <typename data_type>
void convert_diskann_graph_to_adjgraph_batch(
    const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex,
						  AdjGraph &adj, int max_degree) {
  adj =
    std::vector<std::vector<int>>(_pFlashIndex->get_num_points());
  std::cout << "number of points is " << _pFlashIndex->get_num_points() << std::endl;
  uint64_t max_batch_size = 100;
  uint64_t num_read = 0;
  uint64_t total = _pFlashIndex->get_num_points();
  while (num_read < total) {
    if (num_read % 100000 == 0)
      std::cout << num_read << std::endl;
    uint64_t left = total - num_read;
    uint64_t batch_size = std::min(left, max_batch_size);
    std::vector<data_type*> tmp_coord_buffer;
    for (auto i = 0; i < batch_size; i++) {
      tmp_coord_buffer.push_back(nullptr);
    }
    std::vector<std::pair<uint32_t, uint32_t*>> tmp_nbr_buffer;
    uint32_t *neighbors_ptr = new uint32_t[max_degree * batch_size];
    for (size_t i = 0; i < batch_size; i++) {
      tmp_nbr_buffer.emplace_back(0, neighbors_ptr + i * max_degree);
    }
    std::vector<uint32_t> node_ids_to_read;
    for (size_t i = 0; i < batch_size; i++) {
      node_ids_to_read.push_back(num_read + i);
    }
    _pFlashIndex->read_nodes(node_ids_to_read, tmp_coord_buffer, tmp_nbr_buffer);
    for (size_t i = 0; i < batch_size; i++) {
      uint32_t node_id = num_read + i;
      adj[node_id].assign(tmp_nbr_buffer[i].second, tmp_nbr_buffer[i].second + tmp_nbr_buffer[i].first);
    }
    num_read+= batch_size;
    delete[] neighbors_ptr;
  }
}

template <typename data_type>
void convert_diskann_graph_to_adjgraph(
    const std::unique_ptr<diskann::PQFlashIndex<data_type>> &_pFlashIndex,
    AdjGraph &adj, int max_degree) {
  adj = std::vector<std::vector<int>>(_pFlashIndex->get_num_points());
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
  }
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
  
  int res = _pFlashIndex->load(1, index_path_prefix.c_str());
  std::cout << "done loading disk index" << std::endl;
  if (res != 0)
    throw std::runtime_error("error loading diskann data, error: " +
                             std::to_string(res));
  convert_diskann_graph_to_adjgraph<data_type>(_pFlashIndex, adj, _pFlashIndex->get_data_dim());
  std::cout << "done converting diskann graph to adjgraph" << std::endl;
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

Clusters
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



/**
   given a data file, and index path prefix for disk index files, create the
   cluster folder with all cluster related data including index files
*/
template <typename data_type>
void write_all_cluster_data(const std::string &data_file,
                  const std::string &index_path_prefix, const int num_clusters,
                  const std::string &clusters_folder,
                  const std::string &pq_vectors) {
  std::cout << "clustering the index" << std::endl;
  Clusters clusters = get_clusters_from_diskann_graph<data_type>(
								 index_path_prefix, num_clusters);
  
  std::cout << "writing clusters to file" << std::endl;

  write_cluster_data_folder(clusters, clusters_folder);

  create_cluster_index_files<data_type>(clusters, data_file, index_path_prefix,
                                        clusters_folder);
}


