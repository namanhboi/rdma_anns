/**
   take as input a built vamana graph and partition it using graph partitioning into multiple non-overlapping grpah partitions saved to files.
*/

#include "partition_prebuilt_graph.hpp"

AdjGraph convert_graph_to_adjgraph(Graph<unsigned int> &G) {
  AdjGraph adj;
  for (int i = 0; i < G.size(); i++) {
    adj.emplace_back(G[i].begin(), G[i].end());
  }
  return adj;
}


Clusters get_clusters_from_adjgraph(AdjGraph &adj, int num_clusters) {
  double eps = 0.05; 
  return ConvertPartitionToClusters(PartitionAdjListGraph(adj, num_clusters, eps));
}


void save_head_index_node_indices(std::vector<uint32_t> &node_list,
                                  const std::string &in_mem_index_path) {
  const std::string filename = in_mem_index_path + ".indices";
  diskann::save_bin<uint32_t>(filename, node_list.data(), node_list.size(), 1);
  std::cout << "finished writing head index node indices\n";
}

