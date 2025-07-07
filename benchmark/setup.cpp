/*
  This file is used to load in all the data necessary to run a benchmark. This includes all graphs for all shards in KV store.
*/
#include <string>
#include <vector>
#include <utils/graph.h>
#include <utils/euclidian_point.h>
#include <utils/point_range.h>
#include <boost/program_options.hpp>
#include <stdexcept>
#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <iostream>
#include <cascade/service_client_api.hpp>

namespace po = boost::program_options;
using namespace derecho::cascade;
using namespace parlayANN;

#define PROC_NAME "setup"

template<typename PointRange>
void load_graph_into_shard(ServiceClientAPI& capi, const Graph<unsigned int> &G, const PointRange& Points, std::string shard_folder, int subgroup_index) {

  using Point = typename PointRange::Point;
  using data_type = typename Point::distanceType;

  // capi.template 
  auto result_object_pool = capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(shard_folder, subgroup_index);
  for (auto& reply_future:result_object_pool.get()) {
    auto reply = reply_future.second.get();
  }
  std:: cout << "Number of points " << Points.size() << std::endl;
  parlay::parallel_for(0, Points.size(), [&] (size_t i) {
    ObjectWithStringKey emb;
    emb.key = shard_folder + "/emb_" + std::to_string(i);
    emb.previous_version = INVALID_VERSION;
    emb.previous_version_by_key = INVALID_VERSION;
    emb.blob = Blob(reinterpret_cast<const uint8_t*>(Points.location(i)), sizeof(data_type) * Points.dimension());
    auto result_emb = capi.put(emb, false);
    
    for (auto& reply_future:result_emb.get()) {
      auto reply = reply_future.second.get();
      // std::cout << "node(" << reply_future.first << ") replied with version:" << std::get<0>(reply)
      // << ",ts_us:" << std::get<1>(reply)
      // << std::endl;
    }
    ObjectWithStringKey neighbors;
    neighbors.key = shard_folder + "/neighbors_" + std::to_string(i);
    neighbors.previous_version = INVALID_VERSION;
    neighbors.previous_version_by_key = INVALID_VERSION;
    neighbors.blob = Blob(reinterpret_cast<const uint8_t*>(G[i].begin()), G[i].size() * sizeof(unsigned int));
    auto result_neighbors = capi.put(neighbors, false);
    for (auto& reply_future:result_neighbors.get()) {
      auto reply = reply_future.second.get();
    }    
  });
}





int main(int argc, char** argv) {
  po::options_description desc("Program Input");
  desc.add_options()
    ("help,h", "show help message") 
    ("graph-path,G", po::value<std::vector<std::string>>()->required(), "Path to graph segments to be loaded into Cascade. The segments will be loaded into object pools /anns/shard_{i}, where i is the index of the segment. For the ith graph segment, we will load the neighbors of the jth vector into /anns/shard_{i}/neighbors_{j} and the full length vector embedding in /anns/shard_{i}/emb_{j}. Embedding is from the base-path.")
    ("base-path,B", po::value<std::string>()->required(), "Path to the full vector data be loaded into Cascade.")
    ("data-type,T", po::value<std::string>()->required(), "Data type of vectors, could be float, uint8, int8")
    ("dist-func,F", po::value<std::string>()->required(), "Distance function, could be Euclidian or ?. RN only support Euclidian");
  po::variables_map vm;

  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  po::notify(vm);
  std::string dist_func = vm["dist-func"].as<std::string>();
  if (dist_func != "Euclidian") throw std::invalid_argument("only support euclidian");
  std::string data_type = vm["data-type"].as<std::string>();
  std::string base_path = vm["base-path"].as<std::string>();
  const std::vector<std::string> &graph_path_list = vm["graph-path"].as<std::vector<std::string>>();
  
  ServiceClientAPI &capi = ServiceClientAPI::get_service_client();

  if (data_type == "float") {
    using PR = PointRange<Euclidian_Point<float>>;
    PR Points(base_path.data());
    for (int i = 0; i < graph_path_list.size(); i++) {
#define GRAPH_FOLDER_PREFIX "/anns/shard_"
      std::string shard_folder = GRAPH_FOLDER_PREFIX + std::to_string(i);
      std::string graph_path = graph_path_list[i];
      load_graph_into_shard<PR>(capi,Graph<unsigned int>(graph_path.data()), Points, shard_folder, i);
    }
  } else if (data_type == "uint8") {
    using PR = PointRange<Euclidian_Point<uint8_t>>;
    PR Points(base_path.data());
    for (int i = 0; i < graph_path_list.size(); i++) {
#define GRAPH_FOLDER_PREFIX "/anns/shard_"
      std::string shard_folder = GRAPH_FOLDER_PREFIX + std::to_string(i);
      std::string graph_path = graph_path_list[i];
      load_graph_into_shard<PR>(capi,Graph<unsigned int>(graph_path.data()), Points, shard_folder, i);
    }
  } else if (data_type == "int8") {
    using PR = PointRange<Euclidian_Point<int8_t>>;
    PR Points(base_path.data());
    for (int i = 0; i < graph_path_list.size(); i++) {
#define GRAPH_FOLDER_PREFIX "/anns/shard_"
      std::string shard_folder = GRAPH_FOLDER_PREFIX + std::to_string(i);
      std::string graph_path = graph_path_list[i];
      load_graph_into_shard<PR>(capi,Graph<unsigned int>(graph_path.data()), Points, shard_folder, i);
    }
  }
  std::cout <<"done" << std::endl;

}
