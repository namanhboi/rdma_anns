/*
  This file is used to load in all the data necessary to run a benchmark. This
  includes all graphs for all shards in KV store.
*/
#include "defs.h"
#include "metis_io.h"
#include "partition_prebuilt_graph.hpp"
#include <boost/program_options.hpp>
#include <cascade/service_client_api.hpp>
#include <cstddef>
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





int main(int argc, char **argv) {
  po::options_description desc("Program Input");
  std::string index_path_prefix;
  std::string dist_fn;
  std::string data_type;
  int num_clusters;
  std::string output_clusters;
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
      "output_clusters,O", po::value<std::string>(&output_clusters)->required(),
      "Path to write the clusters")(
      "dist_fn,F", po::value<std::string>(&dist_fn)->required(),
      "Distance function, could be Euclidian or ?. RN only support Euclidian");
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
  AdjGraph adj;
  if (data_type == "uint8") {
    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    std::unique_ptr<diskann::PQFlashIndex<uint8_t>> _pFlashIndex(
        new diskann::PQFlashIndex<uint8_t>(reader, diskann::Metric::L2));
    int res =
        _pFlashIndex->load(omp_get_num_procs(), index_path_prefix.c_str());
    if (res != 0)
      throw std::runtime_error("error loading diskann data, error: " +
                               std::to_string(res));
    convert_diskann_graph_to_adjgraph<uint8_t>(_pFlashIndex, adj, MAX_DEGREE);
    Clusters clusters = get_clusters_from_adjgraph(adj, num_clusters);
    WriteClusters(clusters, output_clusters);
    load_diskann_graph_into_cascade<uint8_t>(capi, _pFlashIndex, clusters,
                                             MAX_DEGREE);
  } else if (data_type == "int8") {
    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    std::unique_ptr<diskann::PQFlashIndex<int8_t>> _pFlashIndex(
        new diskann::PQFlashIndex<int8_t>(reader, diskann::Metric::L2));
    int res =
        _pFlashIndex->load(omp_get_num_procs(), index_path_prefix.c_str());
    if (res != 0)
      throw std::runtime_error("error loading diskann data, error: " +
                               std::to_string(res));
    convert_diskann_graph_to_adjgraph<int8_t>(_pFlashIndex, adj, MAX_DEGREE);
    Clusters clusters = get_clusters_from_adjgraph(adj, num_clusters);
    WriteClusters(clusters, output_clusters);
    load_diskann_graph_into_cascade<int8_t>(capi, _pFlashIndex, clusters,
                                            MAX_DEGREE);
  } else if (data_type == "float") {
    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    std::unique_ptr<diskann::PQFlashIndex<float>> _pFlashIndex(
        new diskann::PQFlashIndex<float>(reader, diskann::Metric::L2));
    int res =
        _pFlashIndex->load(omp_get_num_procs(), index_path_prefix.c_str());
    if (res != 0)
      throw std::runtime_error("error loading diskann data, error: " +
                               std::to_string(res));
    convert_diskann_graph_to_adjgraph<float>(_pFlashIndex, adj, MAX_DEGREE);
    Clusters clusters = get_clusters_from_adjgraph(adj, num_clusters);
    WriteClusters(clusters, output_clusters);
    load_diskann_graph_into_cascade<float>(capi, _pFlashIndex, clusters,
                                           MAX_DEGREE);
    build_and_save_head_index(_pFlashIndex, _pFlashIndex->get_num_points(), 32, 50, 1.2, "float", "in_mem_head_index");
  }
  
  // need to load data onto cascade, first is the graph, get the embedding and
  // neighors then load as business.

  return 0;
}
