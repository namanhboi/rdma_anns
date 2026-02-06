#include "disk_utils.h"
#include "types.h"
#include <stdexcept>
#include "distance.h"




int main(int argc, char **argv) {
  if (argc != 6) {
    std::cout << "Arguments in order: <data_type> <base_file> "
                 "<index_path_prefix> <dist_metric> <num_pq_chunks>"
    << std::endl;
    return 1;
  }
  std::string data_type(argv[1]);
  std::string base_file(argv[2]);
  std::string index_path_prefix(argv[3]);
  std::string dist_metric(argv[4]);
  uint64_t num_pq_chunks = std::stoull(argv[5]);
  pipeann::Metric m = pipeann::get_metric(dist_metric);

  if (num_pq_chunks > MAX_NUM_PQ_CHUNKS) {
    throw std::invalid_argument("max pq chunk is " +
                                std::to_string(MAX_NUM_PQ_CHUNKS));
  }

  if (data_type == "float") {
    create_pq_data<float>(base_file, index_path_prefix, num_pq_chunks, m);
  } else if (data_type =="uint8"){
    create_pq_data<uint8_t>(base_file, index_path_prefix, num_pq_chunks, m);
  } else if (data_type =="int8") {
    create_pq_data<int8_t>(base_file, index_path_prefix, num_pq_chunks, m);
  } else {
    throw std::invalid_argument("Data type weird value");
  }
}


