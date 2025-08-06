/**
  This file is used to load in all the data necessary to run a benchmark.
*/
#include "data_loading_utils.hpp"
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

namespace po = boost::program_options;
using namespace derecho::cascade;
using namespace parlayANN;

#define PROC_NAME "setup"

template <typename data_type>
void load_data(ServiceClientAPI &capi, const std::string &index_path_prefix,
               const int num_clusters, const std::string &output_clusters,
               const std::string &in_mem_index_path) {
  // AdjGraph adj;
  // std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  // std::unique_ptr<diskann::PQFlashIndex<data_type>> _pFlashIndex(
  // new diskann::PQFlashIndex<data_type>(reader, diskann::Metric::L2));
  // int res =
  // _pFlashIndex->load(omp_get_num_procs(), index_path_prefix.c_str());
  // if (res != 0)
  // throw std::runtime_error("error loading diskann data, error: " +
  // std::to_string(res));
  // convert_diskann_graph_to_adjgraph<data_type>(_pFlashIndex, adj,
  // MAX_DEGREE); Clusters clusters = get_clusters_from_adjgraph(adj,
  // num_clusters); WriteClusters(clusters, output_clusters);
  // load_diskann_graph_into_cascade<data_type>(capi, _pFlashIndex, clusters,
  // MAX_DEGREE);
  // if (!std::filesystem::exists(in_mem_index_path)) {
  // std::cout << "starting to build the in mem head index at " <<
  // in_mem_index_path << std::endl; build_and_save_head_index(_pFlashIndex,
  // HEAD_INDEX_R, HEAD_INDEX_L, HEAD_INDEX_ALPHA, in_mem_index_path);
  // }
  // load_diskann_head_index_into_cascade<data_type>(capi, in_mem_index_path);
  load_diskann_in_mem_index<data_type>(capi, in_mem_index_path);
  std::cout << "Done loading head index" << std::endl;
  // load_diskann_pq_into_cascade(capi, _pFlashIndex);
  // std::cout << "Done loading pq data" << std::endl;
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
  std::string in_mem_index_path;
  std::string query_file;
  std::string gt_file;

  desc.add_options()("help,h", "show help message")(
      "index_path_prefix,P",
      po::value<std::string>(&index_path_prefix)->required(),
      "Path to index to be loaded into Cascade. The graph will be partioned "
      "into num_clusters via the balanced graph partitioning algorithm. The "
      "non overlapping partitions will be loaded into object pools "
      "/anns/cluster_{i}, where i is the index of the partition. For the ith "
      "graph partition, we will load the data of the jth vector into "
      "/anns/cluster_{i}/vector_{j} and Embedding is from the base-path.")(
      "data_type,T", po::value<std::string>(&data_type)->required(),
      "Data type of vectors, could be float, uint8, int8")(
      "num_clusters,N", po::value<int>(&num_clusters)->required(),
      "Number of partiptions to divide the whole graph into")(
      "clusters_folder,O", po::value<std::string>(&clusters_folder)->required(),
      "Path to write the clusters")(
      "dist_fn,F", po::value<std::string>(&dist_fn)->required(),
      "Distance function, could be Euclidian or ?. RN only support Euclidian")(
      "in_mem_index_path", po::value<std::string>(&in_mem_index_path),
      "Path to in mem index, if not built then build it and save it there")(
      "query_file", po::value<std::string>(&query_file),
      "Path to in mem index, if not built then build it and save it there")(
      "gt_file", po::value<std::string>(&gt_file),
      "Path to in mem index, if not built then build it and save it there");

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

  Clusters clusters;
  if (data_type == "uint8") {
    Clusters clusters = get_clusters_from_diskann_graph<uint8_t>(
								 index_path_prefix, num_clusters);
  } else if (data_type == "int8") {
    Clusters clusters = get_clusters_from_diskann_graph<int8_t>(
								index_path_prefix, num_clusters);
  } else if (data_type == "float") {
    Clusters clusters = get_clusters_from_diskann_graph<float>(
								 index_path_prefix, num_clusters);
  }
  
  write_cluster_data_folder(clusters, clusters_folder);
  std::string cluster_assignment_bin_file =
    clusters_folder + "/cluster_assignment_all.bin";

  std::vector<uint8_t> cluster_assignment =
    parse_cluster_assignment_bin_file(cluster_assignment_bin_file);
  for (uint32_t cluster_id = 0; cluster_id < clusters.size(); cluster_id++) {
    for (uint32_t &node : clusters[cluster_id]) {
      assert(cluster_assignment[node] == cluster_id);
    }
    std::string cluster_nodes_bin_file = clusters_folder + "/cluster_" +
                                         std::to_string(cluster_id) +
                                         "_nodes.bin";
    std::vector<uint32_t> parsed_cluster_nodes =
      parse_cluster_nodes_bin_file(cluster_nodes_bin_file);
    assert(parsed_cluster_nodes == clusters[cluster_id]);
  }
  std::cout << "everything is correct" << std::endl;
  return 0;
}
