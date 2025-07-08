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
#include "metis_io.h"
#include "partition_prebuilt_graph.hpp"
#include <cstddef>
namespace po = boost::program_options;
using namespace derecho::cascade;
using namespace parlayANN;

#define PROC_NAME "setup"
#define WHOLE_GRAPH_SUBGROUP_INDEX 0

// TODO still need to load the graph correctly: load the embedidng next to the neighbors id: /anns/shard_i/vector_j

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



struct vector_data {
  uint8_t *embedding;
  int dimension;
  edgeRange<unsigned int> neighbors;
};
/**
   Load each cluster (full length vector embedding and neighbor ids) onto a
   shard on Cascade. subgroup id is WHOLE_GRAPH_SUBGROUP_INDEX
   Key will be vector id, value will be as follows:
   - sizeof(data_type) * dimension bytes is the embedding
   - the next sizeof(unsigned int) * G[i].size() bytes are the ids of the
   neighbors.
   - will need to change from volatile to persistent datastore, also need to figure out how to pin certain kv pairs in cache (the ones close to starting point);
*/
template <typename PointRange>
void load_whole_graph(ServiceClientAPI &capi, const Graph<unsigned int> &G,
                      const PointRange &Points, const Clusters &clusters) {

  using Point = typename PointRange::Point;
  using data_type = typename Point::distanceType;


  int dimension = Points.dimension();

  if (clusters.size() > 255) {
    throw std::invalid_argument("Too many clusters mate, doesn't fit into 1 byte");
  }
  for (int i = 0; i < clusters.size(); i++) {
    std::string cluster_folder = "/anns/cluster_" + std::to_string(i);
    capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(cluster_folder, WHOLE_GRAPH_SUBGROUP_INDEX, HASH, {}, "cluster[0-9]+");
    parlay::parallel_for(0, clusters[i].size(), [&](size_t j) {
      uint32_t vector_id = clusters[i][j];
      size_t num_byte_emb = sizeof(data_type) * dimension;
      size_t num_byte_neighbors = sizeof(unsigned int) * G[vector_id].size();
      size_t num_byte_cluster_mapping = G[vector_id].size(); // contigous chunk of memory mapping a neighbor to its cluster.
      size_t num_byte_object = num_byte_emb + num_byte_neighbors ;
      std::vector<std::byte> data(num_byte_object);
      std::memcpy(data.data(), Points.location(i), num_byte_emb);
      std::memcpy(data.data() + dimension * sizeof(data_type),
                  G[vector_id].begin(), num_byte_neighbors);
      ObjectWithStringKey vector_data;
      vector_data.key = cluster_folder + "/vector_" + std::to_string(vector_id);
      vector_data.previous_version = INVALID_VERSION;
      vector_data.previous_version_by_key = INVALID_VERSION;
      vector_data.blob = Blob(reinterpret_cast<const uint8_t*>(data.data()), num_byte_object);
      auto result = capi.put(vector_data, false);
      for (auto &reply_future : result.get()) {
        auto reply = reply_future.second.get();
      }
    });
    std::cout << "Done with cluster " << i << "/" << clusters.size() - 1<< std::endl;
  }
}


int main(int argc, char** argv) {
  po::options_description desc("Program Input");
  desc.add_options()
    ("help,h", "show help message") 
    ("graph-path,G", po::value<std::string>()->required(), "Path to whole graph to be loaded into Cascade. The graph will be partioned into num_clusters via the balanced graph partitioning algorithm. The non overlapping partitions will be loaded into object pools /anns/cluster_{i}, where i is the index of the partition. For the ith graph partition, we will load the data of the jth vector into /anns/cluster_{i}/vector_{j} and Embedding is from the base-path.")
    ("base-path,B", po::value<std::string>()->required(), "Path to the full vector data be loaded into Cascade.")
    ("data-type,T", po::value<std::string>()->required(), "Data type of vectors, could be float, uint8, int8")
    ("num-clusters,N", po::value<int>()->required(), "Number of partitions to divide the whole graph into")
    ("output-clusters,OC", po::value<std::string>()->required(), "Path to write the clusters")
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
  std::string graph_path = vm["graph-path"].as<std::string>();
  Graph<unsigned int> G(graph_path.data());

  int num_clusters = vm["num-clusters"].as<int>();

  ServiceClientAPI &capi = ServiceClientAPI::get_service_client();
  std::cout << "Starting to build clusters" << std::endl;
  const Clusters &clusters = get_clusters_from_graph(G, num_clusters);
  std::cout << "Done building clusters" << std::endl;
  std::cout << "total " << clusters.size() << " clusters" << std::endl;
  WriteClusters(clusters, vm["output-clusters"].as<std::string>());
  if (data_type == "float") {
    using PR = PointRange<Euclidian_Point<float>>;
    PR Points(base_path.data());
    load_whole_graph(capi, G, Points, clusters);

  } else if (data_type == "uint8") {
    using PR = PointRange<Euclidian_Point<uint8_t>>;
    PR Points(base_path.data());
    load_whole_graph(capi, G, Points, clusters);
  } else if (data_type == "int8") {
    using PR = PointRange<Euclidian_Point<int8_t>>;
    PR Points(base_path.data());
    load_whole_graph(capi, G, Points, clusters);
  }
  std::cout <<"done" << std::endl;

}
