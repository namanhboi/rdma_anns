#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include "disk_utils.h"

int main(int argc, char **argv) {
  std::string index_path_prefix;
  uint32_t num_points, num_partitions;
  

  if (argc != 4) {
    std::cout << "Usage: <index_path_prefix (doesn't include 'partititon' in path name> <num_points> <num_partitions>" << std::endl;
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
    uint32_t current_partition = i / partition_size;
    partitions[current_partition].push_back(i);
  }
  write_partitions_to_loc_files(partitions, index_path_prefix);
  create_partition_assignment_file(index_path_prefix, num_partitions);
  return 0;
}

