/**
   take as input a built vamana graph and partition it using graph partitioning into multiple non-overlapping grpah partitions saved to files.
*/

#include "data_loading_utils.hpp"
#include <filesystem>
#include <limits>

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


void write_cluster_bin_file(const Clusters &clusters, const std::string &output_clusters) {
  std::ofstream cluster_file(output_clusters, std::ios::binary | std::ios::trunc);
  cluster_file.exceptions(std::ios::badbit | std::ios::failbit);
  uint32_t num_clusters = clusters.size();
  
  cluster_file.write(reinterpret_cast<const char *>(&num_clusters),
                     sizeof(num_clusters));
  for (uint32_t i = 0; i < clusters.size(); i++) {
    uint32_t cluster_size = clusters[i].size();
    cluster_file.write(reinterpret_cast<const char *>(&cluster_size),
                       sizeof(cluster_size));
  }
  for (uint32_t i = 0; i < clusters.size(); i++) {
    const std::vector<uint32_t> &cluster = clusters[i];
    cluster_file.write(reinterpret_cast<const char *>(cluster.data()),
                       sizeof(uint32_t) * cluster.size());
  }
  cluster_file.close();
}


void write_cluster_assignment_bin_file(
				      const Clusters &clusters, const std::string &output_cluster_assignment) {
  uint32_t num_pts = 0;
  for (const auto &cluster : clusters) {
    num_pts += cluster.size();
  }
  std::vector<uint8_t> cluster_assignment(num_pts);

  uint8_t cluster_id = 0;
  for (const auto &cluster : clusters) {
    for (const uint32_t &node_id : cluster) {
      cluster_assignment[node_id] = cluster_id;
    }
    cluster_id++;
  }
  std::ofstream cluster_assignment_out(output_cluster_assignment,
                                       std::ios::binary | std::ios::trunc);

  assert(clusters.size() < std::numeric_limits<uint8_t>::max());
  uint8_t num_clusters = clusters.size();
  cluster_assignment_out.exceptions(std::ios::badbit | std::ios::failbit);
  cluster_assignment_out.write((char *)&num_pts, sizeof(num_pts));
  cluster_assignment_out.write((char *)&num_clusters, sizeof(num_clusters));

  cluster_assignment_out.write((char *)cluster_assignment.data(),
                               sizeof(uint8_t) * cluster_assignment.size());
}


std::vector<std::vector<uint32_t>> parse_cluster_bin_file(const std::string &cluster_file) {

  std::ifstream cluster_file_in;
  cluster_file_in.exceptions(std::ios::failbit | std::ios::badbit);
  cluster_file_in.open(cluster_file.c_str());
  cluster_file_in.seekg(0, cluster_file_in.beg);

  uint32_t num_clusters;
  cluster_file_in.read((char *)&num_clusters, sizeof(num_clusters));
  std::vector<std::vector<uint32_t>> clusters(num_clusters);

  std::vector<uint32_t> cluster_sizes;
  for (uint32_t i = 0; i < num_clusters; i++) {
    uint32_t cluster_size;
    cluster_file_in.read((char *)&cluster_size, sizeof(cluster_size));
    cluster_sizes.push_back(cluster_size);
    clusters[i] = std::vector<uint32_t>(cluster_size);
  }
  for (uint32_t i = 0; i < num_clusters; i++) {
    cluster_file_in.read((char *)clusters[i].data(),
                         sizeof(uint32_t) * cluster_sizes[i]);
  }
  return clusters;
}

void write_cluster_nodes_bin_file(const Clusters &clusters,
                                      const std::string &output_folder) {
  for (uint32_t i = 0; i < clusters.size(); i++) {
    std::string cluster_file =
      output_folder + "/cluster_" + std::to_string(i) + "_nodes.bin";
    std::ofstream cluster_file_out(cluster_file,
                                   std::ios::binary | std::ios::trunc);

    uint32_t num_nodes_cluster = clusters[i].size();
    cluster_file_out.write((char *)&num_nodes_cluster,
                           sizeof(num_nodes_cluster));
    cluster_file_out.write((char *)clusters[i].data(),
                           clusters[i].size() * sizeof(uint32_t));
  }
}

void write_cluster_data_folder(const Clusters &clusters,
                               const std::string &output_folder) {
  namespace fs = std::filesystem;
  fs::path cluster_bin_folder(output_folder);
  if (!fs::exists(cluster_bin_folder)) {
    fs::create_directory(cluster_bin_folder);
  }
  write_cluster_nodes_bin_file(clusters, output_folder);
  std::string cluster_assignment_all_file =
    output_folder + "/cluster_assignment_all.bin";
  write_cluster_assignment_bin_file(clusters, cluster_assignment_all_file);
  write_cluster_nodes_bin_file(clusters, output_folder);
}


std::vector<uint8_t> parse_cluster_assignment_bin_file(
						       const std::string &cluster_assignment_bin_file) {
  std::ifstream in(cluster_assignment_bin_file, std::ios::binary);
  uint32_t num_nodes;
  uint8_t num_clusters;
  in.read((char *)&num_nodes, sizeof(num_nodes));
  in.read((char *)&num_clusters, sizeof(num_clusters));
  std::vector<uint8_t> cluster_assignment(num_nodes);
  in.read((char *)cluster_assignment.data(), sizeof(uint8_t) * num_nodes);
  return cluster_assignment;
}

std::vector<uint32_t>
parse_cluster_nodes_bin_file(const std::string &cluster_nodes_bin_file) {
  std::ifstream in(cluster_nodes_bin_file, std::ios::binary);
  uint32_t num_nodes;
  in.read((char *)&num_nodes, sizeof(num_nodes));

  std::vector<uint32_t> nodes(num_nodes);
  in.read((char*) nodes.data(), sizeof(uint32_t) * num_nodes);
  return nodes;
}


void create_object_pools(ServiceClientAPI &capi) {
  capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(
								      UDL1_OBJ_POOL, UDL1_SUBGROUP_INDEX, HASH, {});

  capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(
								      UDL2_OBJ_POOL, UDL2_SUBGROUP_INDEX, HASH, {}, AFFINITY_SET_REGEX);
}

