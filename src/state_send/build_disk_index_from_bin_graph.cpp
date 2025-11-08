#include "aux_utils.h"
#include <stdexcept>

int main(int argc, char **argv) {
  std::string data_type (argv[1]);
  std::string base_file(argv[2]);
  std::string graph_file(argv[3]);
  std::string output_index_file(argv[4]);

  if (data_type == "float") {
    pipeann::create_disk_layout<float>(graph_file, base_file, "", "", "", false, output_index_file);
  } else if (data_type == "uint8") {
    pipeann::create_disk_layout<uint8_t>(graph_file, base_file, "", "", "",
                                         false, output_index_file);
  } else if (data_type == "int8") {
    pipeann::create_disk_layout<int8_t>(graph_file, base_file, "", "", "",
                                        false, output_index_file);
  } else {
    throw std::invalid_argument("data type weird");

  } 
}


