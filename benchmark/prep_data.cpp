#include "data_loading_utils.hpp"
#include <boost/program_options.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <cascade/service_client_api.hpp>
#include <iostream>
#include <libaio.h>
#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <stdexcept>
#include <string>
#include <utils/euclidian_point.h>
#include <utils/graph.h>
#include <utils/point_range.h>



namespace po = boost::program_options;

int main(int argc, char **argv) {
  po::options_description desc("Program Input");
  bool built_head_index = false;
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
      "index_path_prefix,P", po::value<std::string>(&index_path_prefix),
      "Path to disk index file.")(
      "data_type,T", po::value<std::string>(&data_type),
      "Data type of vectors, could be float, uint8, int8")(
      "num_clusters,N", po::value<int>(&num_clusters),
      "Number of partiptions to divide the whole graph into")(
      "clusters_folder,O", po::value<std::string>(&clusters_folder),
      "Path to write the clusters")(
      "dist_fn,F", po::value<std::string>(&dist_fn),
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
      "Path to pq compressed vectors bin file")(
      "built_head_index", po::bool_switch(&built_head_index),
						"whether all the files for the clusters are built");
  

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
  if (built_head_index != false) {
    if (data_type == "uint8") {
      build_and_save_head_index<uint8_t>(index_path_prefix, head_index_path);
    } else if (data_type == "int8") {
      build_and_save_head_index<int8_t>(index_path_prefix, head_index_path);
    } else if (data_type == "float") {
      build_and_save_head_index<float>(index_path_prefix, head_index_path);
    }
  }
  if (data_type == "uint8") {
    write_all_cluster_data<uint8_t>(data_file,index_path_prefix, num_clusters, clusters_folder, pq_vectors);
  } else if (data_type == "int8") {
    write_all_cluster_data<int8_t>(data_file,index_path_prefix, num_clusters, clusters_folder, pq_vectors);
  } else if (data_type == "float") {
    write_all_cluster_data<float>(data_file,index_path_prefix, num_clusters, clusters_folder, pq_vectors);
  }
  return 0;
  
}
