#include "disk_utils.h"
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


  pipeann::Metric m = dist_metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
  if (dist_metric != "l2" && m == pipeann::Metric::L2) {
    std::cout << "Metric " << dist_metric << " is not supported. Using L2" << std::endl;
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
