#include "disk_utils.h"
#include "utils.h"
#include <stdexcept>




int main(int argc, char **argv) {
  std::string source_parlayann_graph(argv[1]);
  std::string loc_file(argv[2]);
  std::string output_graph_file(argv[3]);


  size_t num_pts, dim;
  std::vector<uint32_t> ids;
  pipeann::load_bin<uint32_t>(loc_file, ids, num_pts, dim);
  if (dim != 1) {
    throw std::invalid_argument(
				"dim for loc file should be 1, weird dim value " + std::to_string(dim));
  }
  LOG(INFO) << "num pts" << num_pts;
  write_graph_file_from_parlayann_graph_file(source_parlayann_graph, ids,
                                             output_graph_file);
  return 0;
  
}



