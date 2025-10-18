#include "graph_partitioning_utils.h"


std::vector<std::vector<uint32_t>> get_partitions_from_adjgraph(std::vector<std::vector<int>> &adj, int num_partitions) {
  double eps = 0.05;
  return ConvertPartitionToClusters(
				    PartitionAdjListGraph(adj, num_partitions, eps));
}


