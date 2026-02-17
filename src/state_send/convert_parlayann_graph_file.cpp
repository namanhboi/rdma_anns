#include "disk_utils.h"
#include "utils.h"
#include <stdexcept>

void read_metadata_parlayann_graph(const std::string &parlayann_graph_file,
                                   uint32_t &num_points, uint32_t &max_deg) {
  std::ifstream reader(parlayann_graph_file, std::ios::binary);
  reader.read((char*)(&num_points), sizeof(uint32_t));
  reader.read((char *)(&max_deg), sizeof(uint32_t));
}



int main(int argc, char **argv) {
  std::string source_parlayann_graph(argv[1]);
  std::string output_graph_file(argv[2]);


  std::vector<uint32_t> ids;
  uint32_t num_points, max_deg;
  read_metadata_parlayann_graph(source_parlayann_graph, num_points, max_deg);
  for (uint32_t i = 0; i < num_points; i++) {
    ids.push_back(i);
  }
  LOG(INFO) << "num pts" << num_points;
  write_graph_file_from_parlayann_graph_file(source_parlayann_graph, ids,
                                             output_graph_file);
  return 0;
  
}



