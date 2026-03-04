#include "utils.h"
#include <stdexcept>
#include <string>
#include "disk_utils.h"

void convert(const std::string &overlap_partition_assignment_file) {
  if (!file_exists(overlap_partition_assignment_file)) {
    throw std::invalid_argument(overlap_partition_assignment_file + " doesn't exist");
  }
  std::vector<std::vector<uint8_t>> overlap_partition_assignment;
  uint8_t num_partitions;
  load_overlap_partition_assignment_file(overlap_partition_assignment_file,
                                         overlap_partition_assignment,
                                         num_partitions);
  std::vector<uint8_t> partition_assignment;
  for (auto &v : overlap_partition_assignment) {
    partition_assignment.insert(partition_assignment.end(), v.begin(), v.end());
  }
  size_t npts = partition_assignment.size();
  pipeann::save_bin<uint8_t>(overlap_partition_assignment_file,
                             partition_assignment.data(), npts, (size_t)1);
}


int main(int argc, char** argv) {
  std::string overlap_partition_assignment_file = argv[1];
  convert(overlap_partition_assignment_file);
  return 0;
}

