#include "utils.h"
#include "pq_table.h"
#include <cstdint>

template <typename T>
void load_and_save(std::string pq_compressed_path, std::string pq_pivots_path) {
  size_t npts, nchunks;
  pipeann::get_bin_metadata(pq_compressed_path, npts, nchunks);

  
  pipeann::FixedChunkPQTable<T> pq_table(pipeann::Metric::L2);
  uint64_t nr, nc, offset = 0;
  {
    std::ifstream reader(pq_pivots_path, std::ios::binary | std::ios::ate);
    reader.seekg(0);
    pq_table.load_pq_pivots_with_dummy(reader, nchunks, 0);
  }
  pq_table.save_pq_pivots(pq_pivots_path.c_str());

}

int main(int argc, char **argv) {
  std::string data_type;
  std::string pq_compressed_path;
  std::string pq_pivots_path;
  if (argc != 4) {
    std::cout << "Use the pq compressed file for metadata for number of chunks "
                 "then convert the old pq pivot format to new one, overwriting "
                 "the pq pivot path file"
    << std::endl;
    std::cout << "Usage: <data_type> <pq_compressed_path>"
                 "<pq_pivots_path>";
    return 1;
  }
  if (data_type == "uint8") {
    load_and_save<uint8_t>(pq_compressed_path, pq_pivots_path);
  } else if (data_type == "int8") {
    load_and_save<int8_t>(pq_compressed_path, pq_pivots_path);
  } else if (data_type == "float") {
    load_and_save<float>(pq_compressed_path, pq_pivots_path);
  } 
  
}
