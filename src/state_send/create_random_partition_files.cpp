#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include "disk_utils.h"

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <index_path_prefix> <num_points> <num_partitions>" << std::endl;
    return 1;
  }

  // 1. Initialize variables from command line arguments
  std::string index_path_prefix = argv[1];
  uint32_t num_points = std::stoul(argv[2]);
  uint32_t num_partitions = std::stoul(argv[3]);

  if (num_partitions == 0) {
    std::cerr << "Error: num_partitions must be greater than 0." << std::endl;
    return 1;
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::vector<uint32_t> ids;
  ids.reserve(num_points);
  
  std::vector<std::vector<uint32_t>> partitions(num_partitions);
  uint32_t partition_size = num_points / num_partitions;
    
  for (uint32_t i = 0; i < num_points; i++) {
    ids.push_back(i);
  }
  
  std::shuffle(ids.begin(), ids.end(), g);
  
  for (uint32_t i = 0; i < num_points; i++) {
    // 2. Prevent Out-of-Bounds by capping the partition index at the last partition.
    // Any "remainder" points will be added to the final partition.
    uint32_t current_partition = std::min(i / partition_size, num_partitions - 1);
    
    // 3. Push the SHUFFLED id, not the loop index 'i'
    partitions[current_partition].push_back(ids[i]);
  }

  for (auto &partition : partitions) {
    std::sort(partition.begin(), partition.end());
  }

  write_partitions_to_loc_files(partitions, index_path_prefix);
  write_partitions_to_txt_files(index_path_prefix, num_partitions);
  create_partition_assignment_file(index_path_prefix, num_partitions);
  
  return 0;
}
