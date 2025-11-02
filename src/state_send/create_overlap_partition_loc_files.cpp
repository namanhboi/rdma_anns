#include "disk_utils.h"
#include <cstdint>
#include <string>



/**
   $HOME/workspace/rdma_anns/build/src/state_send/create_overlap_partition_loc_files \ ~/big-ann-benchmarks/data/bigann/10M/base.1B.u8bin.crop_nb_10000000 uint8 2 0.2 \
~/big-ann-benchmarks/data/bigann/10M/overlap_clusters_2/pipeann_10M

./build/src/state_send/create_overlap_partition_loc_files /home/nd433/big-ann-benchmarks/data/bigann/base.1B.u8bin.crop_nb_100000000 uint8 2 0.2 /home/nd433/anngraphs/bigann/100M/global_overlap_partitions_2/pipeann_100M
*/
int main(int argc, char **argv) {
  std::string base_file(argv[1]);
  std::string data_type(argv[2]);  
  int num_partitions = std::atoi(argv[3]);
  double overlap = std::stod(argv[4]);
  std::string output_index_path_prefix(argv[5]);
  if (data_type== "uint8") {
    create_and_write_overlap_partitions_to_loc_files<uint8_t>(
						     base_file, num_partitions, overlap, output_index_path_prefix);
  } else if (data_type == "float") {
    create_and_write_overlap_partitions_to_loc_files<float>(
						     base_file, num_partitions, overlap, output_index_path_prefix);

  } else if (data_type == "int8") {
    create_and_write_overlap_partitions_to_loc_files<float>(
						     base_file, num_partitions, overlap, output_index_path_prefix);
    create_and_write_overlap_partitions_to_loc_files<int8_t>(
							     base_file, num_partitions, overlap, output_index_path_prefix);
  } else {
    throw std::invalid_argument("data type weird value");
  }
  sort_and_rewrite_partition_loc_files(output_index_path_prefix,
                                       num_partitions);
  create_overlap_partition_assignment_file(output_index_path_prefix,
                                           num_partitions);
  // std::vector<std::vector<uint8_t>> partition_assignment;
  // uint8_t num_partitions_test;
  // load_overlap_partition_assignment_file(
      // output_index_path_prefix + "_partition_assignment.bin",
					 // partition_assignment, num_partitions_test);
}
