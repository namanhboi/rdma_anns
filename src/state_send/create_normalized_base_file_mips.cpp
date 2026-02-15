/**
   create normalized base file and writes it to <base_file>_data.normalized.bin.
   Also creates max base norm file in <base_file>_max_base_norm.bin
*/
#include "utils.h"


int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: <base_file> <data_type>";
  }

  std::string base_file = argv[1];
  std::string data_type = argv[2];
  if (data_type != "float") {
    throw std::invalid_argument(
				"only support normalization for mips for float");
  }
  float max_norm_of_base = pipeann::prepare_base_for_inner_products<float>(
									   base_file, base_file + "_data.normalized.bin");
  std::string norm_file = base_file + "_max_base_norm.bin";
  pipeann::save_bin(norm_file, &max_norm_of_base, 1, 1);
  return 0;
}
