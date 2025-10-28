#include "disk_utils.h"
#include <string>






int main(int argc, char **argv) {
  std::string base_file(argv[1]);
  int num_partitions = std::atoi(argv[2]);
  double overlap = std::stod(argv[3]);
  std::string output_index_path_prefix(argv[4]);
  create_and_write_overlap_partitions_to_loc_files(
						   base_file, num_partitions, overlap, output_index_path_prefix);


  create_overlap_partition_assignment_file(output_index_path_prefix,
                                           num_partitions);
}
