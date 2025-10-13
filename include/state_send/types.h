/**
   includes types like searchstate, queryemb, results, etc and their
   serialization
 */
#pragma once
#include "neighbor.h"
#include "query_buf.h"
#include "timer.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include <chrono>

#define MAX_N_CMPS 16384
#define MAX_N_EDGES 512
#define MAX_PQ_CHUNKS 128
#define SECTOR_LEN 4096

static constexpr int kMaxVectorDim = 512;
static constexpr int maxKSearch = 256;

enum class ClientType : uint32_t { LOCAL = 0, TCP = 1, RDMA = 2 };

using fnhood_t = std::tuple<unsigned, unsigned, char *>;

// message type for the server, it then uses the correct
// deserialize_states/queries method to get the batch of states/queries. I say
// batch but most of the time its 1
enum class MessageType : uint32_t { QUERIES, STATES, RESULT };

enum class SearchExecutionState {
  FINISHED,
  TOP_CAND_NODE_OFF_SERVER,
  TOP_CAND_NODE_ON_SERVER
};

struct QueryStats {
  double total_us = 0; // total time to process query in micros
  double n_4k = 0;     // # of 4kB reads
  double n_ios = 0;    // total # of IOs issued
  double io_us = 0;    // total time spent in IO/waiting for its turn
  double head_us = 0;  // total time spent in in-memory index
  double cpu_us = 0;   // total time spent in CPU
  double n_cmps = 0;   // # cmps
  double n_hops = 0;   // # search hops

  size_t write_serialize(char *buffer) const;
  size_t get_serialize_size() const;
  static std::shared_ptr<QueryStats> deserialize(const char *buffer);
};
inline double get_percentile_stats(
    std::vector<std::shared_ptr<QueryStats>> stats, uint64_t len,
    float percentile,
    const std::function<double(const std::shared_ptr<QueryStats> &)>
        &member_fn) {
  std::vector<double> vals(len);
  for (uint64_t i = 0; i < len; i++) {
    vals[i] = member_fn(stats[i]);
  }

  std::sort(
      vals.begin(), vals.end(),
      [](const double &left, const double &right) { return left < right; });

  auto retval = vals[(uint64_t)(percentile * ((float)len))];
  vals.clear();
  return retval;
}

inline double
get_mean_stats(std::vector<std::shared_ptr<QueryStats>> stats, uint64_t len,
               const std::function<double(const std::shared_ptr<QueryStats> &)>
                   &member_fn) {
  double avg = 0;
  for (uint64_t i = 0; i < len; i++) {
    avg += member_fn(stats[i]);
  }
  return avg / ((double)len);
}

struct search_result_t {
  uint64_t query_id;
  uint64_t num_res;
  uint32_t node_id[maxKSearch];
  float distance[maxKSearch];
  std::shared_ptr<QueryStats> stats = nullptr;

  static std::shared_ptr<search_result_t> deserialize(const char *buffer);
  size_t write_serialize(char *buffer) const;
  size_t get_serialize_size() const;
};

/**
   includes both full embeddings and pq representation of query. Client uses
   this to send stuff to server. Server upon receving a query, makes a
   queryembedding struct to put into map and also make an empty state.
 */
template <typename T> struct QueryEmbedding {
  uint64_t query_id;
  uint64_t client_peer_id;
  uint64_t mem_l = 0;
  uint64_t l_search;
  uint64_t k_search;
  uint64_t beam_width;
  uint32_t dim;
  uint32_t num_chunks;
  bool record_stats;
  T query[kMaxVectorDim];
  float pq_dists[32768];

  static std::shared_ptr<QueryEmbedding> deserialize(const char *buffer);
  static std::vector<std::shared_ptr<QueryEmbedding>>
  deserialize_queries(const char *buffer, size_t size);

  size_t write_serialize(char *buffer) const;
  size_t get_serialize_size() const;
  static size_t write_serialize_queries(
      char *buffer,
      const std::vector<std::shared_ptr<QueryEmbedding>> &queries);

  static size_t get_serialize_size_queries(
      const std::vector<std::shared_ptr<QueryEmbedding>> &queries);
};

template <typename T, typename TagT = uint32_t>
struct alignas(SECTOR_LEN) SearchState {
  // buffer.
  char sectors[SECTOR_LEN * 128];
  uint8_t pq_coord_scratch[32768 * 32];

  T data_buf[ROUND_UP(1024 * kMaxVectorDim, 256)];
  float dist_scratch[512];

  std::shared_ptr<QueryEmbedding<T>> query_emb;

  uint64_t data_buf_idx;
  uint64_t sector_idx;

  // search state.
  std::vector<pipeann::Neighbor> full_retset;
  std::vector<pipeann::Neighbor> retset;
  tsl::robin_set<uint64_t> visited;

  std::vector<unsigned> frontier;

  std::vector<fnhood_t> frontier_nhoods;
  std::vector<IORequest> frontier_read_reqs;

  unsigned cur_list_size, cmps, k;
  uint64_t mem_l = 0, l_search, k_search, beam_width;

  uint64_t query_id;

  // all the partition/server ids that it has been through
  std::vector<uint8_t> partition_history;

  // stats
  pipeann::Timer query_timer, io_timer, cpu_timer;
  std::shared_ptr<QueryStats> stats = nullptr;

  // std::chrono::steady_clock::time_point start_time;
  // std::chrono::steady_clock::time_point end_time;

  // client information to notify of completion
  ClientType client_type;

  /// LOCAL
  TagT *res_tags;
  float *res_dists;
  std::shared_ptr<std::atomic<uint64_t>> completion_count;

  // TCP
  uint64_t client_peer_id;
  /*
    deserialize one search state
   */
  static SearchState *deserialize(const char *buffer);

  /**
     used by the handler to deserialize the blob into states to then send to
     the search threads
   */
  static std::vector<SearchState *> deserialize_states(const char *buffer,
                                                       size_t size);
  /**
     write the serialized form of this state into the buffer.
     Data to be serialized:
     - full_retset
       - retset
       - visited nodes
       - frontier
       - cur_list_size
       - k
       - k_search
       - l_search
       - beamwidth
       - cmps
   */
  size_t write_serialize(char *buffer) const;
  size_t get_serialize_size() const;

  static size_t
  write_serialize_states(char *buffer,
                         const std::vector<SearchState *> &states);

  static size_t
  get_serialize_size_states(const std::vector<SearchState *> &states);

  /**
     sort the full retset then create searchrseult
   */
  search_result_t get_search_result();
};
