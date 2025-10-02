#pragma once
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

#include <unistd.h>

#include "tsl/robin_set.h"
#include "utils.h"

namespace pipeann {
  const size_t MAX_PQ_TRAINING_SET_SIZE = 256000;
  const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 1000000;
  const double PQ_TRAINING_SET_FRACTION = 0.1;
  const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
  const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;
  const uint32_t WARMUP_L = 20;

  double get_memory_budget(const std::string &mem_budget_str);
  double get_memory_budget(double search_ram_budget_in_gb);
  void add_new_file_to_single_index(std::string index_file, std::string new_file);

  size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim);

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at);

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at,
                          const tsl::robin_set<unsigned> &active_tags);

}  // namespace pipeann
