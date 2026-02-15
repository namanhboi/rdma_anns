#include "disk_utils.h"
#include "utils.h"
#include <string>
#include <iostream>
#include <fstream>
#include <array>
#include <vector>

#define MAX_PARTITIONS 15


std::vector<uint32_t> get_ids(const std::string &partition_line) {
  std::vector<uint32_t> ids;
  std::string::size_type pos = 0;
  while (pos != std::string::npos) {
    std::string::size_type new_pos = partition_line.find(" ", pos + 1);
    int start = pos == 0 ? 0 : pos + 1;
    if (new_pos != std::string::npos) {
      ids.push_back(std::stoi(partition_line.substr(start, new_pos - pos)));
    }
    pos = new_pos;
  }
  return ids;
}


std::vector<std::vector<uint32_t>> get_partitions(std::string partition_txt_file) {
  std::vector<std::string> partition_lines;
  std::ifstream f(partition_txt_file);
  std::vector<std::vector<uint32_t>> partitions;
  std::string line;
  while (getline(f, line)) {
    partition_lines.push_back(line);
  }

  for (const auto &partition_line : partition_lines) {
    partitions.push_back(get_ids(partition_line));
  }
  return partitions;
}

void write_files(std::vector<std::vector<uint32_t>> &partitions,
                 std::string output_path_prefix) {
  write_partitions_to_loc_files(partitions, output_path_prefix);

  create_partition_assignment_file(output_path_prefix, partitions.size());
}


int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: <partition_txt_file> <output_path_prefix>" << std::endl;
    return 1;
  }
  std::string partition_txt_file = argv[1];
  std::string output_path_prefix = argv[2];

  auto get_num_points =
      [](const std::vector<std::vector<uint32_t>> &partitions) {
        uint32_t num = 0;
        for (const auto &partition : partitions) {
	  num+= partition.size();
        }
        return num;
      };
  
  std::vector<std::vector<uint32_t>> partitions =
    get_partitions(partition_txt_file);
  LOG(INFO) << "Done parsing file. There are " << partitions.size()
  << " partitions and " << get_num_points(partitions) << " points";
  write_files(partitions, output_path_prefix);
  return 0;
}
