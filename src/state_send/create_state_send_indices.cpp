#include "aux_utils.h"
#include "disk_utils.h"
#include <stdexcept>

template <typename T>
void create_indices(const std::string &base_file,
                    const std::string &index_path_prefix,
                    const std::string &dist_metric, int num_partitions,
                    const std::string &output_index_path_prefix,
                    bool only_partition) {
  LOG(INFO) << num_partitions;
  std::string graph_path = index_path_prefix + "_graph";
  if (!file_exists(graph_path)) {
    write_graph_index_from_disk_index<T>(index_path_prefix, graph_path);
  }
  create_and_write_partitions_to_loc_files(graph_path, output_index_path_prefix,
                                           num_partitions);
  write_partitions_to_txt_files(output_index_path_prefix, num_partitions);
  if (!only_partition) {
    create_partition_assignment_file(output_index_path_prefix, num_partitions);
    create_partition_assignment_symlinks(output_index_path_prefix,
                                         num_partitions);
    create_base_files_from_tags<T>(base_file, output_index_path_prefix,
                                   num_partitions);
    create_graphs_from_tags(graph_path, output_index_path_prefix,
                            num_partitions);
    create_disk_indices<T>(output_index_path_prefix, num_partitions);
  }
}

/**
   ./create_state_send_indices uint8
/home/nam/big-ann-benchmarks/data/bigann/base.1B.u8bin.crop_nb_10000000
/home/nam/big-ann-benchmarks/data/bigann/pipeann_10M l2 2
/home/nam/big-ann-benchmarks/data/bigann/global_graph_partitions/pipeann_10M


./build/tests/search_memory_index uint8 10000000
/home/nam/big-ann-benchmarks/data/bigann/pipeann_10M_graph 0 0
/home/nam/big-ann-benchmarks/data/bigann/query.public.10K.u8bin
/home/nam/big-ann-benchmarks/data/bigann/bigann-10M 10 bruh 8 l2
/home/nam/big-ann-benchmarks/data/bigann/global_graph_partixtions/pipeann_10M
*/

int main(int argc, char **argv) {
  std::string data_type(argv[1]);
  std::string base_file(argv[2]);
  std::string index_path_prefix(argv[3]);
  std::string dist_metric(argv[4]);
  int num_partitions = std::atoi(argv[5]);
  std::string output_index_path_prefix(argv[6]);
  int only_partition = std::atoi(argv[7]);
  if (only_partition != 1 && only_partition != 0) {
    throw std::invalid_argument("only partition can only be 0 or 1");
  }
  pipeann::Metric m =
      dist_metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
  if (dist_metric != "l2" && m == pipeann::Metric::L2) {
    std::cout << "Metric " << dist_metric << " is not supported. Using L2"
              << std::endl;
  }

  if (data_type == "float") {
    create_indices<float>(base_file, index_path_prefix, dist_metric,
                          num_partitions, output_index_path_prefix, only_partition == 1);
  } else if (data_type == "uint8") {
    create_indices<uint8_t>(base_file, index_path_prefix, dist_metric,
                            num_partitions, output_index_path_prefix, only_partition == 1);
  } else if (data_type == "int8") {
    create_indices<int8_t>(base_file, index_path_prefix, dist_metric,
                           num_partitions, output_index_path_prefix, only_partition == 1);
  }
}
