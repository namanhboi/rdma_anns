#pragma once
#include <immintrin.h> // needed to include this to make sure that the code compiles since in DiskANN/include/utils.h it uses this library.
#include <cstdint>
#include <stdexcept>
#include <string>
#include "utils.h" // this is diskann utils

/**
   this class is a container for query and groundtruth data.
*/
template<typename data_type>
class BenchmarkDataset {
public:
  uint32_t next_query = 0;

  data_type *query_data = nullptr;
  size_t query_num;
  size_t query_dim;
  size_t query_aligned_dim; // basically dim rounded to a multiple of 8

  size_t gt_num;
  size_t gt_dim; // this is the K, so for each point, gt_ids has the gt_dim nearest neighbors.
  uint32_t *gt_ids = nullptr;
  float *gt_dists = nullptr;
  

  BenchmarkDataset(const std::string &query_file,
                          const std::string &gt_file) {
    diskann::load_aligned_bin(query_file, query_data, query_num, query_dim,
                              query_aligned_dim);
    diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      throw std::runtime_error("number of queries doesn't match the ground truth");
    }
    std::cout << "Query dim " << query_dim << std::endl;
    std::cout << "GT dim " << gt_dim << std::endl;
  }
  uint64_t get_next_query_index() {
    auto query_index = next_query;
    next_query++;
    return query_index;
  }
  void reset() { this->next_query = 0; }

  const data_type *get_query(uint32_t query_index) {
    return query_data + query_index * query_aligned_dim;
  }

  size_t get_num_queries() { return query_num; }

  uint32_t get_dim() {
    return query_dim;
  }

  size_t get_query_size() {
    return this->query_dim * sizeof(data_type);
  }

  
  ~BenchmarkDataset() {
    diskann::aligned_free(query_data);
    delete[] gt_ids;
    delete[] gt_dists;
  }

};
