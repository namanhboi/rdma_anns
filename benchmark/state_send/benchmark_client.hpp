#pragma once
#include <thread>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include "blockingconcurrentqueue.h"
#include "serialize_utils.hpp"
#include <vector>

//TagT is the type of the index of the nodes the graph
template <typename T, typename TagT = uint32_t> class BenchmarkClient {
  std::unordered_map<query_id_t, std::chrono::steady_clock::time_point> query_send_time;
  std::unordered_map<query_id_t, std::chrono::steady_clock::time_point>
      query_result_time;

public:

  virtual void issue_query(const T *query, const uint64_t k_search,uint32_t mem_L, const uint64_t l_search, const uint32_t beam_width) = 0;
  virtual void wait_results() = 0;
  virtual result_t<TagT> get_result() = 0;
};


template<typename T, typename TagT = uint32_t>
class LocalBenchmarkClient : public BenchmarkClient<T, TagT> {

  std::mutex results_mtx;
  std::unordered_map<query_id_t, result_t<TagT>> results;
  

public:
  LocalBenchmarkClient();
  void issue_query(const T *query, const uint64_t k_search,uint32_t mem_L, const uint64_t l_search, const uint32_t beam_width) override;
  void wait_results() override;
  result_t<TagT> get_result() override;
  ~LocalBenchmarkClient();
};
