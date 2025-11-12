#include "disk_utils.h"


int main(int argc, char **argv) {
  std::string data_type(argv[1]);
  std::string base_file(argv[2]);
  uint32_t num_partitions = std::atoi(argv[3]);
  std::string output_index_path_prefix(argv[4]);

  if (data_type == "float") {
    create_and_write_partitions_to_loc_files<float>(
						      base_file, output_index_path_prefix, num_partitions);
  } else if (data_type == "uint8") {
    create_and_write_partitions_to_loc_files<uint8_t>(
						      base_file, output_index_path_prefix, num_partitions);
  } else if (data_type == "int8") {
    create_and_write_partitions_to_loc_files<uint8_t>(
						      base_file, output_index_path_prefix, num_partitions);
  }
  create_partition_assignment_file(output_index_path_prefix, num_partitions);
  return 0;
}

