/**
  This file is used to load in all the data necessary to run a benchmark.
*/
#include "data_loading_utils.hpp"
#include "defaults.h"
#include "defs.h"
#include "metis_io.h"
#include <boost/program_options.hpp>
#include <cascade/service_client_api.hpp>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <libaio.h>
#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <stdexcept>
#include <string>
#include <utils/euclidian_point.h>
#include <utils/graph.h>
#include <utils/point_range.h>
#include <vector>

/**
   this file currently load a prebuilt diskann index and run a clustering
   algorithm on it to create disjoint clusters. It then load these clusters onto
   cascade.
 */

namespace po = boost::program_options;
using namespace derecho::cascade;
using namespace parlayANN;


template <typename data_type>
void load_data_fs(ServiceClientAPI &capi, const std::string &data_file, const std::string &index_path_prefix,
		  const int num_clusters, const std::string &clusters_folder, const std::string &pq_vectors) {
  Clusters clusters = get_clusters_from_diskann_graph<data_type>(
								 index_path_prefix, num_clusters);
  write_cluster_data_folder(clusters, clusters_folder);

  create_cluster_index_files<data_type>(clusters, data_file, index_path_prefix,
                                        clusters_folder);
  // need to load pq data in as well
#ifdef PQ_KV
  load_diskann_pq_into_cascade(capi, pq_vectors, clusters);
#endif
}


template <typename data_type>
void load_data_kv(ServiceClientAPI &capi, const std::string &index_path_prefix,
               const int num_clusters, const std::string &clusters_folder) {
  Clusters clusters = get_clusters_from_diskann_graph<data_type>(
								 index_path_prefix, num_clusters);
  WriteClusters(clusters, "clusters.txt");
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<data_type>> _pFlashIndex(
							       new diskann::PQFlashIndex<data_type>(reader, diskann::Metric::L2));
  int _ = _pFlashIndex->load(omp_get_num_procs(), index_path_prefix.c_str());
#ifdef IN_MEM
  load_diskann_graph_into_cascade_in_mem(capi, _pFlashIndex, clusters,
                                         MAX_DEGREE);
#else
  load_diskann_graph_into_cascade_ssd(capi, _pFlashIndex, clusters, MAX_DEGREE);
#endif
  write_cluster_data_folder(clusters, clusters_folder);
}

template <typename data_type>
void test_head_index(const std::string &index_path_prefix,
                     const std::string &in_mem_index_path,
                     const std::string &query_file,
                     const std::string &gt_file) {
  // std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  // std::unique_ptr<diskann::PQFlashIndex<data_type>> _pFlashIndex(
  // new diskann::PQFlashIndex<data_type>(reader, diskann::Metric::L2));

  std::unique_ptr<diskann::Index<data_type>> head_index =
      get_index<data_type>(in_mem_index_path);

  run_queries_head_index<data_type>(std::move(head_index), query_file, gt_file);
}

int main(int argc, char **argv) {
  po::options_description desc("Program Input");
  std::string index_path_prefix;
  std::string dist_fn;
  std::string data_type;
  int num_clusters;
  std::string output_clusters;
  std::string clusters_folder;
  std::string head_index_path;
  std::string query_file;
  std::string gt_file;
  std::string data_file;
  std::string pq_vectors;

  desc.add_options()("help,h", "show help message")(
      "index_path_prefix,P",
      po::value<std::string>(&index_path_prefix)->required(),
      "Path to disk index file.")(
      "data_type,T", po::value<std::string>(&data_type)->required(),
      "Data type of vectors, could be float, uint8, int8")(
      "num_clusters,N", po::value<int>(&num_clusters)->required(),
      "Number of partiptions to divide the whole graph into")(
      "clusters_folder,O", po::value<std::string>(&clusters_folder)->required(),
      "Path to write the clusters")(
      "dist_fn,F", po::value<std::string>(&dist_fn)->required(),
      "Distance function, could be Euclidian or ?. RN only support Euclidian")(
      "head_index_path", po::value<std::string>(&head_index_path),
      "Path to in mem index, if not built then build it and save it there")(
      "query_file", po::value<std::string>(&query_file),
      "Path to in mem index, if not built then build it and save it there")(
      "gt_file", po::value<std::string>(&gt_file),
      "Path to in mem index, if not built then build it and save it there")(
      "data_file", po::value<std::string>(&data_file),
      "Path to in data file, like sift learn")(
      "pq_vectors", po::value<std::string>(&pq_vectors),
					       "Path to pq compressed vectors bin file");
  po::variables_map vm;

  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  po::notify(vm);
  if (dist_fn != "Euclidian")
    throw std::invalid_argument("only support euclidian");
  if (data_type != "uint8" && data_type != "int8" && data_type != "float")
    throw std::invalid_argument("wrong data_type");

  ServiceClientAPI &capi = ServiceClientAPI::get_service_client();
  create_object_pools(capi);

//udl1 only needs the object pool to be created and nothing else
#if !defined(TEST_UDL1)

  // if not test udl1 and udl2 then must be running fr, so we need to create head index
#if !defined(TEST_UDL2)
  if (data_type == "uint8") {
    build_and_save_head_index<uint8_t>(index_path_prefix, head_index_path);
  } else if (data_type == "int8") {
    build_and_save_head_index<int8_t>(index_path_prefix, head_index_path);
  } else if (data_type == "float") {
    build_and_save_head_index<float>(index_path_prefix, head_index_path);
  }  
#endif

#if  (defined(IN_MEM) || defined(DISK_KV))
  std::cout << "doing data loading" << std::endl;
  if (data_type == "uint8") {
    load_data_kv<uint8_t>(capi, index_path_prefix, num_clusters, clusters_folder);
  } else if (data_type == "int8") {
    load_data_kv<int8_t>(capi, index_path_prefix, num_clusters, clusters_folder);
  } else if (data_type == "float") {
    load_data_kv<float>(capi, index_path_prefix, num_clusters, clusters_folder);
  }
#elif defined(DISK_FS_DISKANN_WRAPPER) || defined(DISK_FS_DISTRIBUTED)
  if (data_type == "uint8") {
    load_data_fs<uint8_t>(capi, data_file,index_path_prefix, num_clusters, clusters_folder, pq_vectors);
  } else if (data_type == "int8") {
    load_data_fs<int8_t>(capi, data_file,index_path_prefix, num_clusters, clusters_folder, pq_vectors);
  } else if (data_type == "float") {
    load_data_fs<float>(capi, data_file,index_path_prefix, num_clusters, clusters_folder, pq_vectors);
  }
#endif
#endif  
  return 0;
  
}
