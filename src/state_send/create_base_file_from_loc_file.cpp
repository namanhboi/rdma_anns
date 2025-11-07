#include "disk_utils.h"


int main(int argc, char **argv) {
  std::string data_type(argv[1]);
  std::string base_file(argv[2]);
  std::string loc_file(argv[3]);
  std::string base_file_output_path(argv[4]);

  if (data_type == "float") {
    create_base_from_tag<float>(base_file, loc_file, base_file_output_path);
  } else if (data_type == "uint8") {
    create_base_from_tag<uint8_t>(base_file, loc_file, base_file_output_path);
  } else if (data_type == "int8") {
    create_base_from_tag<int8_t>(base_file, loc_file, base_file_output_path);
  } else {
    throw std::invalid_argument("data type has weird value");
  }
  return 0;
}
