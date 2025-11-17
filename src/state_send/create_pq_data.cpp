#include "disk_utils.h"
#include "types.h"
#include <stdexcept>




int main(int argc, char **argv) {
  std::string data_type(argv[1]);
  std::string base_file(argv[2]);
  std::string index_path_prefix(argv[3]);
  std::string dist_metric(argv[4]);
  uint64_t num_pq_chunks = std::stoull(argv[5]);
  pipeann::Metric m =
    dist_metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
  if (dist_metric != "l2" && m == pipeann::Metric::L2) {
    std::cout << "Metric " << dist_metric << " is not supported. Using L2" << std::endl;
  }

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


