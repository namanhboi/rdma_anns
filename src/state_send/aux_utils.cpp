#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "aux_utils.h"
#include "percentile_stats.h"
#include "utils.h"

#include "ssd_partition_index.h"
#include "tsl/robin_set.h"
#include "utils.h"

#define NUM_KMEANS 15

namespace pipeann {

  double get_memory_budget(double search_ram_budget) {
    double final_index_ram_limit = search_ram_budget;
    if (search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB > THRESHOLD_FOR_CACHING_IN_GB) {  // slack for space used by
                                                                                           // cached nodes
      final_index_ram_limit = search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB;
    }
    return final_index_ram_limit * 1024 * 1024 * 1024;
  }

  double get_memory_budget(const std::string &mem_budget_str) {
    double search_ram_budget = atof(mem_budget_str.c_str());
    return get_memory_budget(search_ram_budget);
  }

  size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim) {
    size_t num_pq_chunks = (size_t) (std::floor)(uint64_t(final_index_ram_limit / (double) points_num));

    LOG(INFO) << "Calculated num_pq_chunks :" << num_pq_chunks;
    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

    LOG(INFO) << "Compressing " << dim << "-dimensional data into " << num_pq_chunks << " bytes per vector.";
    return num_pq_chunks;
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at) {
    double total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t tie_breaker = recall_at;
      if (gs_dist != nullptr) {
        float *gt_dist_vec = gs_dist + dim_gs * i;
        tie_breaker = recall_at - 1;
        while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);

      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or, unsigned recall_at,
                          const tsl::robin_set<unsigned> &active_tags) {
    double total_recall = 0;
    std::set<unsigned> gt, res;
    bool printed = false;
    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t tie_breaker = recall_at;
      unsigned active_points_count = 0;
      unsigned cur_counter = 0;
      while (active_points_count < recall_at && cur_counter < dim_gs) {
        if (active_tags.find(*(gt_vec + cur_counter)) != active_tags.end()) {
          active_points_count++;
        }
        cur_counter++;
      }
      if (active_tags.empty())
        cur_counter = recall_at;

      if ((active_points_count < recall_at && !active_tags.empty()) && !printed) {
        LOG(INFO) << "Warning: Couldn't find enough closest neighbors " << active_points_count << "/" << recall_at
                  << " from truthset for query # " << i << ". Will result in under-reported value of recall.";
        printed = true;
      }
      if (gs_dist != nullptr) {
        tie_breaker = cur_counter - 1;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);
      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return ((double) (total_recall / (num_queries))) * ((double) (100.0 / recall_at));
  }
};  // namespace pipeann
