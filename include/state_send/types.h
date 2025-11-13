/**
   includes types like searchstate, queryemb, results, etc and their
   serialization
 */
#pragma once
#include "blockingconcurrentqueue.h"
#include "neighbor.h"
#include "query_buf.h"
#include "timer.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include <chrono>
#include <limits>

#define MAX_N_CMPS 16384
#define MAX_N_EDGES 512
#define MAX_PQ_CHUNKS 128
#define SECTOR_LEN 4096

static constexpr int kMaxVectorDim = 512;
static constexpr int maxKSearch = 256;

enum class ClientType : uint32_t { LOCAL = 0, TCP = 1, RDMA = 2 };

enum class DistributedSearchMode : uint32_t {
  SCATTER_GATHER = 0,
  STATE_SEND = 1,
  SINGLE_SERVER = 2
};

inline std::string dist_search_mode_to_string(DistributedSearchMode mode) {
  if (mode == DistributedSearchMode::SCATTER_GATHER) {
    return "SCATTER_GATHER";
  } else if (mode == DistributedSearchMode::STATE_SEND) {
    return "STATE_SEND";
  } else if (mode == DistributedSearchMode::SINGLE_SERVER) {
    return "SINGLE_SERVER";
  } else {
    throw std::runtime_error("Weird dist search mode value");
  }
}

using fnhood_t = std::tuple<unsigned, unsigned, char *>;

// message type for the server, it then uses the correct
// deserialize_states/queries method to get the batch of states/queries. I say
// batch but most of the time its 1
enum class MessageType : uint32_t {
  QUERIES,
  STATES,
  RESULT,

  // sent by client to all servers during state send to tell them to
  // deallocate the memory from query embedding
  RESULT_ACK,

  RESULTS,
  RESULTS_ACK,

  POISON // used to kill the batchig thread
};

inline std::string message_type_to_string(MessageType msg_type) {
  switch (msg_type) {
  case MessageType::QUERIES:
    return "QUERIES";
  case MessageType::STATES:
    return "STATES";
  case MessageType::RESULT:
    return "RESULT";
  case MessageType::RESULT_ACK:
    return "RESULT_ACK";
  case MessageType::RESULTS:
    return "RESULTS";
  case MessageType::RESULTS_ACK:
    return "RESULTS_ACK";
  case MessageType::POISON:
    return "POISON";
  default:
    return "UNKNOWN";
  }
}
/**
   sent to a server to free data associated with query embedding during state
   send
 */
struct ack {
  uint64_t query_id;

  size_t write_serialize(char *buffer) const;

  size_t get_serialize_size() const;

  static ack deserialize(const char *buffer);
};

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
  uint64_t client_peer_id;
  uint32_t k_search;
  uint32_t num_res;
  uint32_t node_id[maxKSearch];
  float distance[maxKSearch];
  std::vector<uint8_t> partition_history;
  std::shared_ptr<QueryStats> stats = nullptr;
  bool is_final = false;

  static std::shared_ptr<search_result_t> deserialize(const char *buffer);
  size_t write_serialize(char *buffer) const;
  size_t get_serialize_size() const;

  static size_t write_serialize_results(
      char *buffer,

      const std::vector<std::shared_ptr<search_result_t>> &results);

  static size_t get_serialize_results_size(
      const std::vector<std::shared_ptr<search_result_t>> &results);

  static std::vector<std::shared_ptr<search_result_t>>
  deserialize_results(const char *buffer);
};


constexpr uint32_t MAX_NUM_RESULTS = 512;

struct combined_search_results_t {
  uint32_t num_expected_results = std::numeric_limits<uint32_t>::max();
  uint32_t num_current_results = 0;
  std::array<std::shared_ptr<search_result_t>, MAX_NUM_RESULTS> results;
  std::array<std::chrono::steady_clock::time_point, MAX_NUM_RESULTS>
      receive_time;

  combined_search_results_t() {}


  combined_search_results_t(
			    std::initializer_list<std::shared_ptr<search_result_t>> res_list, uint32_t num_expected_results) : num_expected_results(num_expected_results)
  {
    num_current_results = 0;
    for (const auto &res : res_list) {
      results[num_current_results] = res;
      receive_time[num_current_results] = std::chrono::steady_clock::now();
      num_current_results++;
    }
  }
  void add_result(std::shared_ptr<search_result_t> result);

  std::shared_ptr<search_result_t> merge_results();
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
  bool populated_pq_dists = false;
  T query[kMaxVectorDim];
  float pq_dists[32768];

  /**
     we don't send pq dists because its big, initialize it upon receiving query
     for the first time.
   */

  static void deserialize(const char *buffer, QueryEmbedding *query);
  static void deserialize_queries(const char *buffer, uint64_t num_queries,
                                  QueryEmbedding **queries);

  size_t write_serialize(char *buffer) const;
  size_t get_serialize_size() const;
  static size_t write_serialize_queries(
      char *buffer,
      const std::vector<std::shared_ptr<QueryEmbedding>> &queries);

  static size_t get_serialize_size_queries(
					   const std::vector<std::shared_ptr<QueryEmbedding>> &queries);


  static void reset(QueryEmbedding<T>*);
};

template <typename T, typename TagT = uint32_t>
struct alignas(SECTOR_LEN) SearchState {
  // buffer.
  char sectors[SECTOR_LEN * 4];
  uint8_t pq_coord_scratch[32768 * 32];

  T data_buf[ROUND_UP(512 * kMaxVectorDim, 256)];
  float dist_scratch[512];

  QueryEmbedding<T> *query_emb = nullptr;

  uint64_t data_buf_idx = 0;
  uint64_t sector_idx = 0;

  // search state.
  std::vector<pipeann::Neighbor> full_retset;
  pipeann::Neighbor retset[1024];
  tsl::robin_set<uint32_t> visited;

  std::vector<unsigned> frontier;

  std::vector<fnhood_t> frontier_nhoods;
  std::vector<IORequest> frontier_read_reqs;

  unsigned cur_list_size = 0, cmps = 0, k = 0;
  uint64_t mem_l = 0, l_search = 0, k_search = 0, beam_width = 0;
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

  // TCP
  uint64_t client_peer_id;
  /*
    deserialize one search state
   */
  static void deserialize(const char *buffer, SearchState* state);

  /**
     used by the handler to deserialize the blob into states to then send to
     the search threads
   */
  static void deserialize_states(const char *buffer, uint64_t num_states,
                                 uint64_t num_queries, SearchState **states,
                                 QueryEmbedding<T> **queries);
  
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

  static size_t write_serialize_states(
      char *buffer, const std::vector<std::pair<SearchState *, bool>> &states);

  static size_t get_serialize_size_states(
      const std::vector<std::pair<SearchState *, bool>> &states);

  /**
     sort the full retset then create searchrseult
   */
  std::shared_ptr<search_result_t> get_search_result();

  // void write_serialize_result(char *buffer) const;
  // void get_serialize_result_size(char *buffer) const;
  static void reset(SearchState * state);
};








template <typename T> class PreallocatedQueue {
private:
  moodycamel::BlockingConcurrentQueue<T *> queue;
  T *elements;
  uint64_t num_elements;
  std::function<void(T *)> reset_element;
public:
  PreallocatedQueue(uint64_t num_elements,
                    std::function<void(T *)> reset_element)
  : num_elements(num_elements), reset_element(reset_element) {
    elements = new T[num_elements];
    LOG(INFO) << "Allocation successful";
    for (uint64_t i = 0; i < num_elements; i++) {
      queue.enqueue(elements + i);
    }
  }

  /*
    result must be allocated before hand
  */
  void dequeue_exact(uint64_t num_elements, T** elements) {
    size_t num_dequeued = queue.wait_dequeue_bulk(elements, num_elements);
    while (num_dequeued < num_elements) {
      T **it = elements + num_dequeued;
      size_t left = num_elements - num_dequeued;
      size_t new_deq = queue.wait_dequeue_bulk(it, left);
      num_dequeued += new_deq;
    }
  }

  /**
     element must be in an "empty" state where its ready to be enqueued. for
     search state, it must be cleared.
  */
  void free(T *element) {
    if (reset_element != nullptr) {
      reset_element(element);
    }
    queue.enqueue(element);
  }
  uint64_t get_num_elements() const { return num_elements; }

  ~PreallocatedQueue() { delete[] elements; }
  PreallocatedQueue(const PreallocatedQueue &) = delete;
  PreallocatedQueue &operator=(const PreallocatedQueue &) = delete;
  PreallocatedQueue(PreallocatedQueue &&other) noexcept
      : queue(std::move(other.queue)), elements(other.elements) {
    other.elements = nullptr;
  }
  PreallocatedQueue &operator=(PreallocatedQueue &&other) noexcept {
    if (this != &other) {
      delete[] elements;
      queue = std::move(other.queue);
      elements = other.elements;
      other.elements = nullptr;
    }
    return *this;
  }
};
