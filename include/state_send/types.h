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
#include <stdexcept>
#include <variant>
#define MAX_N_CMPS 16384
#define MAX_N_EDGES 512
#define MAX_PQ_CHUNKS 128
#define SECTOR_LEN 4096


// void print_neighbor_vec(const std::vector<pipeann::Neighbor> &nbrs);

// template <typename T> std::string list_to_string(T *start, uint32_t num) {
  // std::string str = "";
  // for (uint32_t i = 0; i < num; i++) {
    // str += std::to_string(start[i]) + " " ;
  // }
  // return str;
// }

size_t write_data(char *buffer, const char *data, size_t size,
                  size_t &offset);

static constexpr int kMaxVectorDim = 512;
static constexpr int maxKSearch = 256;
// need to enforce this shit
static constexpr uint32_t MAX_L_SEARCH = 512;


constexpr uint64_t MAX_PRE_ALLOC_ELEMENTS = 100000;

// max for statesend and scatter gather that uses inter query balacing impl (that I
// made) called balance batch
// later when/if i implement the comparison to pipeann methods then the beam
// width of those will be higher but we don't care since it won't be using
// states
constexpr uint32_t BALANCE_BATCH_MAX_BEAMWIDTH = 8;
constexpr uint32_t MAX_NUM_NEIGHBORS = 128;
constexpr uint32_t MAX_NUM_PQ_CHUNKS = 64;

enum class ClientType : uint32_t { LOCAL = 0, TCP = 1, RDMA = 2 };

enum class DistributedSearchMode : uint32_t {
  SCATTER_GATHER = 0,
  STATE_SEND = 1,
  SINGLE_SERVER = 2,
  DISTRIBUTED_ANN = 3,
  STATE_SEND_CLIENT_GATHER = 4,
  SCATTER_GATHER_TOP_N = 5
};



inline std::string dist_search_mode_to_string(DistributedSearchMode mode) {
  if (mode == DistributedSearchMode::SCATTER_GATHER) {
    return "SCATTER_GATHER";
  } else if (mode == DistributedSearchMode::STATE_SEND) {
    return "STATE_SEND";
  } else if (mode == DistributedSearchMode::SINGLE_SERVER) {
    return "SINGLE_SERVER";
  } else if (mode == DistributedSearchMode::DISTRIBUTED_ANN) {
    return "DISTRIBUTED_ANN";
  } else if (mode == DistributedSearchMode::STATE_SEND_CLIENT_GATHER) {
    return "STATE_SEND_CLIENT_GATHER";
  } else if (mode == DistributedSearchMode::SCATTER_GATHER_TOP_N) {
    return "SCATTER_GATHER_TOP_N";
  } else {
    throw std::runtime_error("Weird dist search mode value");
  }
}
inline DistributedSearchMode
get_distributed_search_mode(std::string dist_search_mode_str) {
  if (dist_search_mode_str == "STATE_SEND") {
    return  DistributedSearchMode::STATE_SEND;
  } else if (dist_search_mode_str == "SCATTER_GATHER") {
    return  DistributedSearchMode::SCATTER_GATHER;
  } else if (dist_search_mode_str == "SINGLE_SERVER") {
    return  DistributedSearchMode::SINGLE_SERVER;
  } else if (dist_search_mode_str == "DISTRIBUTED_ANN") {
    return  DistributedSearchMode::DISTRIBUTED_ANN;
  } else if (dist_search_mode_str == "STATE_SEND_CLIENT_GATHER"){
    return  DistributedSearchMode::STATE_SEND_CLIENT_GATHER;
  } else if (dist_search_mode_str == "SCATTER_GATHER_TOP_N") {
    return DistributedSearchMode::SCATTER_GATHER_TOP_N;
  }else {
    throw std::invalid_argument("Dist search mode has weird value " +
                                dist_search_mode_str);
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


  DISTRIBUTED_ANN_RESULTS,
  SCORING_QUERIES,

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
  case MessageType::DISTRIBUTED_ANN_RESULTS:
    return "DISTRIBUTED_ANN_RESULTS";
  case MessageType::SCORING_QUERIES:
    return "SCORING_QUERIES";
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
  FRONTIER_OFF_SERVER,
  FRONTIER_ON_SERVER,
  FRONTIER_EMPTY
};

enum class UpdateFrontierValue {
  // these 3 value categories are mutually exclusive
  FRONTIER_EMPTY_NO_OFF_SERVER,
  // there is no unexplored neighbor starting from index k

  FRONTIER_HAS_ON_SERVER,
  // frontier is not empty and
  // there is at least some on server nodes to explore

  FRONTIER_EMPTY_ONLY_OFF_SERVER,
  // frontier is empty because all nodes visited by state update frontier are
  // off server
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
  double n_inter_partition_hops = 0;

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
  uint64_t k_search;
  uint64_t num_res;
  uint32_t node_id[MAX_L_SEARCH * 2];
  float distance[MAX_L_SEARCH * 2];
  std::vector<uint8_t> partition_history;
  std::vector<uint32_t> partition_history_hop_idx;
  std::shared_ptr<QueryStats> stats = nullptr;

  // used for state_send_client_gather, the number of results to expect = size of partition history?
  bool is_final_result = false;

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

  static size_t get_max_num_res() { return MAX_L_SEARCH * 2; }
  static void reset(search_result_t*);
};

struct client_gather_results_t {
  std::vector<std::shared_ptr<search_result_t>> results;
  int final_result_idx = -1;
  client_gather_results_t(const std::shared_ptr<search_result_t> &res, bool &_all_results_arrived) {
    results.push_back(res);
    if (res->is_final_result)
      final_result_idx = 0;
    _all_results_arrived = all_results_arrived();
  }
  inline bool all_results_arrived() {
    if (final_result_idx == -1)
      return false;

    // we expect this many because in between 2 partition history entry, there
    // is a hop happening, which also means a result that is sent to the client
    // Additionally there is the final result which is sent, so the number of
    // results to expect = partition_history size
    uint32_t num_results_to_expect =
      results[final_result_idx]->partition_history.size();
    // LOG(INFO) << "NUM_RESULTS TO EXPECT " << num_results_to_expect;
    // LOG(INFO) << "NUM_RESULTS " << results.size();
    // LOG(INFO) << "final res partition history";
    // for (size_t i = 0; i < results[final_result_idx]->partition_history.size();
    //      i++) {
    //   std::cout << "(" << (uint32_t)results[final_result_idx]->partition_history[i] << ", " << results[final_result_idx]->partition_history_hop_idx[i] << "),";
    // }
    // std::cout << std::endl;
    return num_results_to_expect == results.size();
  }

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
  float query_norm = 0.0; // used for mips at reranking step
  bool record_stats;
  bool populated_pq_dists = false;
  // don't serialize this one since each time we receive a query we
  // need to populate the table. 
  bool normalized = false;  
  void *distributed_ann_state_ptr = nullptr;
  alignas(8 * sizeof(T)) T query[kMaxVectorDim];
  alignas(256) float pq_dists[32768];
  // alignment same as pipeann init_query_buf to speed up distance calc?
  // haven't benchmarked so don't know fs
  // https://github.com/thustorage/PipeANN/blob/main/include/ssd_index.h#L102

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
  // buffer to copy disk sector data into
  char sectors[SECTOR_LEN * BALANCE_BATCH_MAX_BEAMWIDTH];
  uint8_t pq_coord_scratch[MAX_NUM_NEIGHBORS * MAX_NUM_PQ_CHUNKS];
  
  // copy full embedding here to do dist comparison with full emb query
  T data_buf[kMaxVectorDim];
  
  float dist_scratch[MAX_NUM_NEIGHBORS];

  QueryEmbedding<T> *query_emb = nullptr;

  // max is the beam width, for beam width 1 this will always be 0
  uint64_t sector_idx = 0;

  // search state.
  std::vector<pipeann::Neighbor> full_retset;
  pipeann::Neighbor retset[1024];
  // tsl::robin_set<uint32_t> visited;

  std::vector<unsigned> frontier;

  std::vector<fnhood_t> frontier_nhoods;
  std::vector<IORequest> frontier_read_reqs;

  unsigned cur_list_size = 0, cmps = 0, k = 0;
  uint64_t mem_l = 0, l_search = 0, k_search = 0, beam_width = 0;
  uint64_t query_id;

  // all the partition/server ids that it has been through
  std::vector<uint8_t> partition_history;
  std::vector<uint32_t> partition_history_hop_idx;

  // stats
  pipeann::Timer query_timer, io_timer, cpu_timer;
  std::shared_ptr<QueryStats> stats = nullptr;

  // std::chrono::steady_clock::time_point start_time;
  // std::chrono::steady_clock::time_point end_time;

  // client information to notify of completion
  ClientType client_type;

  // TCP
  uint64_t client_peer_id;


  bool should_send_emb = false;

  // // since enqueuing both the state and the result is atomic, we actually don't need this i think
  // bool sent_state = false;
  // // used for state_send_client_gather
  // bool need_to_send_result_when_send_state = false;

  /*
    deserialize one search state
   */
  static void deserialize(const char *buffer, SearchState *state);

  /**
     used by the handler to deserialize the blob into states to then send to
     the search threads
   */
  static void deserialize_states(const char *buffer, uint64_t num_states,
                                 uint64_t num_queries, SearchState **states,
                                 QueryEmbedding<T> **queries);

  std::shared_ptr<search_result_t>
  get_search_result(DistributedSearchMode dist_search_mode) const;
  size_t write_search_result(DistributedSearchMode dist_search_mode, char * buffer) const;
  /**
     this doesn't serialize the query embedding but does serialize the stats
   */
  size_t write_serialize(char *buffer) const;
  /**
     gets the serialize size without query embedding
   */
  size_t get_serialize_size() const;

  /**
     [num states] [num_queries] [states] [queries]
   */
  static size_t write_serialize_states(
				       char *buffer, SearchState ** states, size_t num_states);

  static size_t get_serialize_size_states(
					  SearchState ** states, size_t num_states);

  // void write_serialize_result(char *buffer) const;
  // void get_serialize_result_size(char *buffer) const;
  static void reset(SearchState *state);
};

template <typename T> class PreallocatedQueue {
private:
  moodycamel::BlockingConcurrentQueue<T *> queue;
  T *elements = nullptr;
  char* additional_data_block = nullptr;
  uint64_t num_elements;
  std::function<void(T *)> reset_element;

public:
  PreallocatedQueue() : num_elements(0), reset_element(nullptr) {};
  PreallocatedQueue(uint64_t num_elements,
                    std::function<void(T *)> reset_element)
      : num_elements(num_elements), reset_element(reset_element) {
    elements = new T[num_elements];
    for (uint64_t i = 0; i < num_elements; i++) {
      queue.enqueue(elements + i);
    }
  }

  // used to support Region
  void allocate_and_assign_additional_block(
      size_t block_size_per_element,
					    std::function<void(T *, char *, void*)> assign_block) {
    additional_data_block = new char[block_size_per_element * num_elements];
    for (size_t i = 0; i < num_elements; i++) {
      assign_block(elements + i,
                   additional_data_block + block_size_per_element * i, this);
    }
  }
  

  /*
    result must be allocated before hand
  */
  void dequeue_exact(uint64_t num_elements, T **elements) {
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

  ~PreallocatedQueue() {
    delete[] elements;
    delete[] additional_data_block;
  }
  
  PreallocatedQueue(const PreallocatedQueue &) = delete;
  PreallocatedQueue &operator=(const PreallocatedQueue &) = delete;
  PreallocatedQueue(PreallocatedQueue &&other) noexcept
      : queue(std::move(other.queue)), elements(other.elements),
        num_elements(other.num_elements),
        reset_element(std::move(other.reset_element)), additional_data_block(other.additional_data_block)
  {
    other.elements = nullptr;
    other.num_elements = 0;
    other.reset_element = nullptr;
    other.additional_data_block = nullptr;
  }
  PreallocatedQueue &operator=(PreallocatedQueue &&other) noexcept {
    if (this != &other) {
      delete[] elements;
      delete[] additional_data_block;
      queue = std::move(other.queue);
      elements = other.elements;
      additional_data_block = other.additional_data_block;
      num_elements = other.num_elements;
      reset_element = std::move(other.reset_element);
    
      other.elements = nullptr;
      other.num_elements = 0;
      other.reset_element = nullptr;
      other.additional_data_block = nullptr;
    }
    return *this;
  }
};


/**
   RAII wrapper for a single item taken from a preallocated queue of pointers to
   that type
*/
template <typename T> class PreallocSingleManager {
private:
  PreallocatedQueue<T> &_queue;
  T *item;
  PreallocSingleManager(const PreallocSingleManager<T> &);
  PreallocSingleManager &operator=(const PreallocSingleManager<T> &);
public:
  PreallocSingleManager(PreallocatedQueue<T> &queue) : _queue(queue) {
    T* arr[1];
    _queue.dequeue_exact(1, arr);
    item = arr[0];
  }

  T *get_item() {
    return item;
  }

  ~PreallocSingleManager() { _queue.free(this->item); }
};




namespace distributedann {
  constexpr uint32_t MAX_BEAM_WIDTH_DISTRIBUTED_ANN = 64;
  constexpr uint32_t MAX_L_DISTRIBUTED_ANN = 1024; // 64 nodes * 64 neighbors each
  
template <typename T> struct scoring_query_t {
  uint64_t query_id; // scoring for which query
  uint32_t num_node_ids;
  uint32_t node_ids[MAX_BEAM_WIDTH_DISTRIBUTED_ANN];
  float threshold;
  uint32_t L;
  bool record_stats = false;
  QueryEmbedding<T> *query_emb = nullptr;
  void *distributed_ann_state_ptr = nullptr; // NEED TO DO THIS HABIBI
  uint64_t client_peer_id = std::numeric_limits<uint64_t>::max();

  // don't deserialize the query_emb
  static void deserialize(const char *buffer, scoring_query_t *);

  
  static void deserialize_scoring_queries(const char *buffer,
                                          uint64_t num_scoring_queries,
                                          uint64_t num_query_embs,
                                          scoring_query_t **scoring_queries,
                                          QueryEmbedding<T> **queries);
  

  // doesnt serialize the query embedding
  size_t write_serialize(char *buffer) const;
  size_t get_serialize_size() const;

  static size_t write_serialize_scoring_queries(
      char *buffer,
      const std::vector<std::pair<scoring_query_t *, bool>> &queries);

  static size_t get_serialize_size_scoring_queries(
						   const std::vector<std::pair<scoring_query_t *, bool>> &queries);

  static void reset(scoring_query_t *);
};



/**
   used to store result for both the head index and also the scoring service
 */
template <typename T> struct result_t {
  uint64_t query_id;
  size_t num_full_nbrs = 0;
  std::pair<uint32_t, float> sorted_full_nbrs[MAX_BEAM_WIDTH_DISTRIBUTED_ANN];
  size_t num_pq_nbrs = 0;
  std::pair<uint32_t, float> sorted_pq_nbrs[MAX_L_DISTRIBUTED_ANN];
  std::shared_ptr<QueryStats> stats = nullptr;
  void *distributed_ann_state_ptr = nullptr;
  uint64_t client_peer_id = std::numeric_limits<uint64_t>::max();


  static void deserialize(const char *buffer, result_t *);
  static void deserialize_results(const char *buffer, uint64_t num_results,
                                  result_t **results);

  size_t write_serialize(char *buffer) const;
  size_t get_serialize_size() const;

  static size_t
  get_serialize_size_results(const std::vector<result_t *> &results);

  static size_t write_serialize_results(char * buffer, const std::vector<result_t *> &results);
  // static size_t get_serialize_size_results();
  static void reset(result_t<T> *);
};


template <typename T, typename TagT = uint32_t>
struct DistributedANNState : SearchState<T, TagT> {
  moodycamel::BlockingConcurrentQueue<result_t<T> *> result_queue;


  static void reset(DistributedANNState<T, TagT>* state);
};


  enum DistributedANNTaskType { HEAD_INDEX, SCORING_QUERY };
  
  template <typename T> struct DistributedANNTask {
    DistributedANNTaskType task_type;
    std::variant<QueryEmbedding<T> *, scoring_query_t<T> *> task;
  };

}; // namespace distributedann
