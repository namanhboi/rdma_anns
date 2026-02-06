#include "omp.h"

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

template<typename T>
bool build_index(const char *dataFilePath, const char *indexFilePath, const char *indexBuildParameters,
                 pipeann::Metric m, bool singleFile) {
  return pipeann::build_disk_index<T>(dataFilePath, indexFilePath, indexBuildParameters, m, singleFile);
}

int main(int argc, char **argv) {
  if (argc != 11) {
    std::cout << "Usage: " << argv[0]
              << " <data_type (float/int8/uint8)>  <data_file.bin>"
                 " <index_path_prefix> <R>  <L>  <num_pq_chunks>  "
                 "<indexing_memory_budget>  <num_threads>"
                 " <similarity metric (cosine/l2/mips) case sensitive>."
                 " <single_file_index (0/1)>"
                 " See README for more information on parameters."
    << std::endl;
    return 1;
  }
  std::string data_type(argv[1]);
  // std::string data_path(argv[2]);
  // std::string index_path_prefix(argv[3]);
  uint32_t R = std::stoi(argv[4]);
  uint32_t L = std::stoi(argv[5]);
  uint32_t num_pq_chunks = std::stoi(argv[6]);
  double indexing_ram_budget = std::stod(argv[7]);
  uint32_t num_threads = std::stod(argv[8]);
  std::string dist_metric(argv[9]);
  bool single_file_index = (std::stoi(argv[10]) == 1);
  

  pipeann::Metric m = pipeann::get_metric(dist_metric);

  if (data_type == "float") {
    pipeann::build_disk_index<float>(argv[2], argv[3], R, L, num_pq_chunks,
                                     num_threads, indexing_ram_budget, m,
                                     single_file_index, nullptr, 1);
  }else if (data_type == std::string("int8")) {
      pipeann::build_disk_index<int8_t>(argv[2], argv[3], R, L, num_pq_chunks,
                                     num_threads, indexing_ram_budget, m,
                                     single_file_index, nullptr, 1);
  }else if (data_type == std::string("uint8")) {
    pipeann::build_disk_index<uint8_t>(argv[2], argv[3], R, L, num_pq_chunks,
                                       num_threads, indexing_ram_budget, m,
                                       single_file_index, nullptr, 1);
  }else {
    std::cout << "Error. wrong file type" << std::endl;
    return 1;
  }
}

