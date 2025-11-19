#include "disk_utils.h"
#include <string>


int main(int main, char **argv) {
  std::string data_type(argv[1]);
  std::string index_path_prefix(argv[2]);

  std::string graph_file_output = index_path_prefix + "_graph";
  if (data_type == "float") {
    write_graph_index_from_disk_index<float>(index_path_prefix,
                                             graph_file_output);
  } else if (data_type == "uint8") {
    write_graph_index_from_disk_index<uint8_t>(index_path_prefix,
                                             graph_file_output);

  } else if (data_type == "int8") {
    write_graph_index_from_disk_index<int8_t>(index_path_prefix,
                                             graph_file_output);
  }
  return 0;
}
