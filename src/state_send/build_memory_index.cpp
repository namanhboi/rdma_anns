#include "disk_utils.h"
#include "distance.h"
#include "utils.h"
#include <stdexcept>


int main(int argc, char** argv) {
  std::string data_type(argv[1]);
  std::string data_path(argv[2]);
  std::string tags_file(argv[3]);
  uint32_t R = std::stoul(argv[4]);
  uint32_t L = std::stoul(argv[5]);
  float alpha = std::stof(argv[6]);
  std::string output_path(argv[7]);
  uint32_t num_threads = std::stoul(argv[8]);
  std::string dist_metric(argv[9]);
  
  bool dynamic_index = false;
  bool single_file_index = false;


  pipeann::Metric m = pipeann::get_metric(dist_metric);
  auto is_normalized_file = [](const std::string &data_path) {
    std::string normalized_suffix = "_data.normalized.bin";
    if (data_path.length() < normalized_suffix.length()) {
      return false;
    }
    std::string data_path_suffix =
        data_path.substr(data_path.length() - normalized_suffix.length(),
                         normalized_suffix.length());
    return data_path_suffix == normalized_suffix;
  };
  if (m == pipeann::Metric::INNER_PRODUCT) {
    LOG(INFO) << "for inner product, we require that the input file has to be "
                 "normalized first, which means it must end in "
                 "_data.normalized.bin. This allows L2 to be used. You can "
                 "normalize a file with create_normalized_base_file_mips.cpp";
    if (data_type != "float") {
      throw std::invalid_argument("data_type must be float for mips");
    }
    if (!is_normalized_file(data_path)) {
      throw std::invalid_argument(
				  "file must be normalized aka ending in _data.normalized.bin");
    }
    m = pipeann::Metric::L2;
  }

  if (data_type == "float") {
    build_in_memory_index<float>(data_path, tags_file, R, L, alpha, output_path,
                                 num_threads, dynamic_index, single_file_index,
                                 m);
  } else if (data_type == "uint8") {
    build_in_memory_index<uint8_t>(data_path, tags_file, R, L, alpha, output_path,
                                 num_threads, dynamic_index, single_file_index,
                                 m);
  } else if (data_type == "int8") {
    build_in_memory_index<int8_t>(data_path, tags_file, R, L, alpha, output_path,
                                 num_threads, dynamic_index, single_file_index,
                                 m);
  } else {
    throw std::invalid_argument("data type werid");
  }

}
