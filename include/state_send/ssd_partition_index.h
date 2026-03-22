#pragma once

#include "blockingconcurrentqueue.h"
#include "cached_io.h"
#include "communicator.h"
#include "concurrentqueue.h"
#include "index.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "linux_aligned_file_reader.h"
#include "neighbor.h"
#include "pq_table.h"
#include "query_buf.h"
#include "tsl/robin_set.h"
#include "types.h"
#include "utils.h"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <set>
#include <string>
#include <unordered_set>
#include <variant>

#define MAX_N_CMPS 16384
#define MAX_N_EDGES 512
#define MAX_PQ_CHUNKS 128

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

#define MAX_WORKER_THREADS 64

constexpr uint64_t MAX_ELEMENTS_HANDLER = 500;

namespace {
inline void aggregate_coords(const unsigned *ids, const uint64_t n_ids,
                             const uint8_t *all_coords, const uint64_t ndims,
                             uint8_t *out) {
  for (uint64_t i = 0; i < n_ids; i++) {
    memcpy(out + i * ndims, all_coords + ids[i] * ndims,
           ndims * sizeof(uint8_t));
  }
}

inline void prefetch_chunk_dists(const float *ptr) {
  _mm_prefetch((char *)ptr, _MM_HINT_NTA);
  _mm_prefetch((char *)(ptr + 64), _MM_HINT_NTA);
  _mm_prefetch((char *)(ptr + 128), _MM_HINT_NTA);
  _mm_prefetch((char *)(ptr + 192), _MM_HINT_NTA);
}

inline void pq_dist_lookup(const uint8_t *pq_ids, const uint64_t n_pts,
                           const uint64_t pq_nchunks, const float *pq_dists,
                           float *dists_out) {
  _mm_prefetch((char *)dists_out, _MM_HINT_T0);
  _mm_prefetch((char *)pq_ids, _MM_HINT_T0);
  _mm_prefetch((char *)(pq_ids + 64), _MM_HINT_T0);
  _mm_prefetch((char *)(pq_ids + 128), _MM_HINT_T0);

  prefetch_chunk_dists(pq_dists);
  memset(dists_out, 0, n_pts * sizeof(float));
  for (uint64_t chunk = 0; chunk < pq_nchunks; chunk++) {
    const float *chunk_dists = pq_dists + 256 * chunk;
    if (chunk < pq_nchunks - 1) {
      prefetch_chunk_dists(chunk_dists + 256);
    }
    for (uint64_t idx = 0; idx < n_pts; idx++) {
      uint8_t pq_centerid = pq_ids[pq_nchunks * idx + chunk];
      dists_out[idx] += chunk_dists[pq_centerid];
    }
  }
}
} // namespace

constexpr int max_requests = 1000;
/**
  job is to manage search threads which advance the search states, eventually
  either sending them to other servers or send to client
 */
template <typename T, typename TagT = uint32_t> class SSDPartitionIndex {

  // concurernt hashmap
  libcuckoo::cuckoohash_map<uint64_t, QueryEmbedding<T> *> query_emb_map;

public:
  /**
     state of a beam search execution
   */
  void state_compute_dists(SearchState<T, TagT> *state, const unsigned *ids,
                           const uint64_t n_ids, float *dists_out);

  void state_print(SearchState<T, TagT> *state);
  void state_reset(SearchState<T, TagT> *state);

  /**
     called at the end of compute and add to retset and explore frontier. This
     is so that  issue_next_io_batch can read the frontier and issue the reads


the updated frontier will only contain nodes that are on-server/can be read

     RETURNS: whether all the nodes iterated over to fill the beam are off
     server, if so then return true, else false. In case where we have reached
     the end of the retset, return false.

ONLY RETURN TRUE IF WE MUST SEND THE STATE.

   */
  UpdateFrontierValue state_update_frontier(SearchState<T, TagT> *state);

  void state_compute_and_add_to_retset(SearchState<T, TagT> *state,
                                       const unsigned *node_ids,
                                       const uint64_t n_ids);

  void state_issue_next_io_batch(SearchState<T, TagT> *state, void *ctx);

  /**
     advances the state based on whatever is in frontier_nhoods, which is the
     result of reading what's in the frontier.
     It also updates the frontier after exploring what's in frontier_nhoods.
     Based on the state of the frontier, it can return the corresponding
     SearchExecutionState
   */
  SearchExecutionState state_explore_frontier(SearchState<T, TagT> *state);

  bool state_search_ends(SearchState<T, TagT> *state);

  void apply_tags_to_result(uint32_t *node_id, uint64_t num_res);

  // static void write_serialize_query(const T *query_emb,
  //                                   const uint64_t k_search,
  //                                   const uint64_t mem_l,
  //                                   const uint64_t l_search,
  //                                   const uint64_t beam_width, char *buffer)
  //                                   {
  //
  // }

  /*
    Traces back the nodes that state_update_frontier also went through. Return
    the partition_id of the first one that is offserver. If none are off server
    then return this server's partition_id.
   */
  uint8_t state_top_cand_random_partition(SearchState<T, TagT> *state);
  // bool state_is_top_cand_off_server(SearchState<T, TagT> *state);

  void state_print_detailed(SearchState<T, TagT> *state);
  void query_emb_print(std::shared_ptr<QueryEmbedding<T>> query_emb);

  bool state_io_finished(SearchState<T, TagT> *state);

  /*
    If dist search mode doesn't require any gathering then sort the distances.
    This is because for gather, the client will sort the distance anyways.

    In case of inner product, we have to conver the distance to l2 because of
    the normalization. TODO: need to implement the distance conversion
    correctly, will probably involve client in some way.
   */
  void state_finalize_distance(SearchState<T, TagT> *state);

  bool state_should_send_emb(SearchState<T, TagT> *state,
                             uint64_t server_peer_id);

  size_t state_get_serialize_result_size(SearchState<T, TagT> *state);
  size_t state_write_result(SearchState<T, TagT> *state, char *buffer);

  size_t states_get_serialize_result_sizes(SearchState<T, TagT> **states,
                                           size_t num_states);
  size_t states_write_results(SearchState<T, TagT> **states, size_t num_states,
                              char *buffer);

private:
  class CounterThread {
  private:
    SSDPartitionIndex *parent;
    std::atomic<bool> running = false;
    std::thread real_thread;

    std::stringstream cached_csv_output;
    // cached_ofstream csv_output;
    std::ofstream csv_output;
    uint64_t sleep_duration_ms;

    void write_header_csv();
    void write_one_row_to_csv();
    void main_loop();

  public:
    CounterThread(SSDPartitionIndex *parent, const std::string &log_output_path,
                  uint64_t sleep_duration_ms);
    void start();
    void join();
    void signal_stop();
  };
  std::unique_ptr<CounterThread> counter_thread;
  bool use_counter_thread;

private:
  bool use_logging;

  inline uint64_t get_timestamp_ns() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  }

private:
  static constexpr uint64_t max_queries_balance = 128;
  uint64_t num_queries_balance = 0;
  bool record_stats;

  class SearchThread {
  private:
    SSDPartitionIndex *parent;
    std::thread real_thread;
    // id used so that parent can send queries round robin
    uint64_t thread_id;
    std::atomic<bool> running{false};
    void *ctx = nullptr;

    moodycamel::ConsumerToken search_thread_consumer_token;

    std::atomic<uint64_t> number_concurrent_queries = 0;

    std::atomic<uint64_t> number_own_states = 0;
    std::atomic<uint64_t> number_foreign_states = 0;
    void process_state(SearchExecutionState s,
                               SearchState<T, TagT> *state);

    /**
       main loop that runs the search. This version only balances batch_size
       queries at a time.
     */
    void main_loop_batch();
    friend class CounterThread;

  public:
    /**
       will run the search, won't contain a queue/have any way of directly
       issueing a query. Instead, wait on io requests
     */
    SearchThread(SSDPartitionIndex *parent, uint64_t thread_id);
    void start();
    void signal_stop();
    void join();
  };

  // class ScoringThread : SearchThread {
  // private:
  //   void process_state(SearchExecutionState s,
  //                      SearchState<T, TagT> *state) override;
  // };

  moodycamel::ConcurrentQueue<SearchState<T, TagT> *> global_state_queue;

  moodycamel::ProducerToken client_state_prod_token;
  moodycamel::ProducerToken server_state_prod_token;

  std::vector<std::unique_ptr<SearchThread>> search_threads;
  uint32_t num_search_threads;

  // these 2 are only used for distributedann
  uint32_t num_orchestration_threads;
  uint32_t num_scoring_threads;

  std::atomic<int> current_search_thread_id = 0;

private:
  /*
    Distributedann stuff
   */
  bool is_orchestration_server; 

  /*
    returns 3 values, FINISHED, FRONTIER_OFF_SERVER (issue rpc to scoring
    server), FRONTIER_EMPTY, in which case we loop until its FINISHED. Can't
    return FRONTIER_ON_SERVER ecause orchestration server can't store any index
    data.

   */
  SearchExecutionState
  state_explore_frontier_orchestration(SearchState<T, TagT> *state);

  /*
    return true if frontier is not empty (meaning we can send stuff to scoring
    server), else return false
   */
  bool state_update_frontier_orchestration(SearchState<T, TagT> *state);


  /*
    send the states to scoring servers based on node_id
   */
  void state_send_scoring_queries_distributedann(SearchState<T, TagT> *state);

  uint64_t get_random_scoring_server_id();

  class OrchestrationThread {
  private:
    SSDPartitionIndex *parent;
    std::thread real_thread;
    // id used so that parent can send queries round robin
    uint64_t thread_id;
    std::atomic<bool> running{false};
    void *ctx = nullptr;
    moodycamel::ConsumerToken state_consumer_token;
    moodycamel::ConsumerToken result_consumer_token;
    std::atomic<uint64_t> number_concurrent_queries = 0;

    std::atomic<uint64_t> number_own_states = 0;
    std::atomic<uint64_t> number_foreign_states = 0;    
    // main loop poll on this queue to get search result which contains pointer
    // to io request which contains pointer to the state.
    // Use this state pointer to add the results back into the state
    moodycamel::ConcurrentQueue<search_result_t *> computation_result_queue;


    void add_states_to_batch(SearchState<T, TagT> **states, size_t num_states);

    void process_state(SearchExecutionState s,
                       SearchState<T, TagT> *state);

    // each iteration of this loop tries to advance one of the state in the
    // balancing batch.
    // Every computation (mem index, centroid, frontier RPC requests) require an
    // rpc to a scoring server which will return a search_result_t.
    // Once we receive the search_result_t, we have the pointer to the IORequest
    // which we set the finished member to true.
    // We then check if all io is finished for the state.
    // If not, continue, if yes then advance the state by incorporating the
    // search_result_t into the retset and full retset of the state. The
    // function that does this can be called distributedann_explore_frontier.
    // This function returns the execution state of the SearchState, then we can
    // handle each value via bool process_state(SearchState) *s). Returns true
    // if state is finished, false is not
    // Added search_result field to IORequest to link search_result_t to that
    // IORequest
    void main_loop_batch();

  public:
    OrchestrationThread(SSDPartitionIndex<T, TagT> *parent, uint64_t thread_id);
    void enqueue_computation_result(search_result_t *);
    void start();
    void signal_stop();
    void join();
  };

  /*
    takes scoring states from global state queue and compute their results.
    If frontier is empty then do mem index (need to add mem_k) or centroid, if
    not then progress state with issue io, then state explore frontier.

almost the same as search thread but progress state for only 1 step. Need to
handle the enable loc bullshit
   */

  void state_explore_frontier_scoring_simple(SearchState<T, TagT> *state);

  class ScoringThread {
  private:
    SSDPartitionIndex *parent;
    std::thread real_thread;
    // id used so that parent can send queries round robin
    uint64_t thread_id;
    std::atomic<bool> running{false};
    void *ctx = nullptr;

    moodycamel::ConsumerToken search_thread_consumer_token;

    std::atomic<uint64_t> number_concurrent_queries = 0;

    std::atomic<uint64_t> number_own_states = 0;
    std::atomic<uint64_t> number_foreign_states = 0;    
    void process_state(SearchExecutionState s,
                       SearchState<T, TagT> *state);
    void main_loop_batch();

    void add_states_to_batch(SearchState<T, TagT> **states, size_t num_states);
    
    
  public:
    ScoringThread(SSDPartitionIndex<T, TagT> *parent, uint64_t thread_id);
    void start();
    void signal_stop();
    void join();
  };

  std::vector<std::unique_ptr<OrchestrationThread>> orchestration_threads;
  std::vector<std::unique_ptr<ScoringThread>> scoring_threads;
private:
  /**
     handles commmunication with servers and clients, note that if you enqueue a
     state, it will be deleted by btaching thread, don't double delete it
     yourself.
   */
  class BatchingThread {
  private:
    PreallocatedQueue<Region> preallocated_region_queue;
    SSDPartitionIndex *parent;

    std::thread real_thread;
    std::atomic<bool> running = false;

    std::unordered_map<uint64_t,
                       std::unique_ptr<std::vector<SearchState<T, TagT> *>>>
        state_queue;
    std::unordered_map<uint64_t,
                       std::unique_ptr<std::vector<SearchState<T, TagT> *>>>
        result_queue;
    std::condition_variable msg_queue_cv;
    std::mutex msg_queue_mutex; // also manages is_peer_client

    std::vector<uint64_t> get_num_msg_peers() {
      std::scoped_lock lock(msg_queue_mutex);
      std::vector<uint64_t> num_msg_peer;
      for (const auto &peer_id : parent->communicator->get_other_peer_ids()) {
        uint64_t num_msg = 0;
        if (state_queue.contains(peer_id)) {
          num_msg = state_queue[peer_id]->size();
        }
        if (result_queue.contains(peer_id)) {
          num_msg += result_queue[peer_id]->size();
        }
        num_msg_peer.push_back(num_msg);
      }
      return num_msg_peer;
    }

    void main_loop();
    friend class CounterThread;

  public:
    BatchingThread(SSDPartitionIndex *parent);
    void push_client_result_to_batch(SearchState<T, TagT> *state);
    /**
       for state send, just send the state, for client gather, send result to
       client as well
     */
    void push_state_to_batch(SearchState<T, TagT> *state,
                             uint64_t recipient_peer_id);
    void start();
    void join();
    void signal_stop();
  };

  std::unique_ptr<BatchingThread> batching_thread;

public:
  /**
     starts all search and io threads
   */
  void start();

  /**
     shutdown all search and io threads
   */
  void shutdown();

public:
  /**
     contains code for general setup stuff for the pipeann disk index.
     is_local determines whether we want to spin up the communication layer or
     if we just want to search the index directly without going through tcp.
     numpartitions > 1 must mean that we use is_local = false;
     is_local = true and num_parittion = 1 is fine, this just means that we send
     query and receive results via tcp.
   */
  SSDPartitionIndex(pipeann::Metric m, uint8_t partition_id,
                    uint32_t num_search_threads,
                    uint32_t num_orchestration_threads,
                    uint32_t num_scoring_threads,
                    std::shared_ptr<AlignedFileReader> &fileReader,
                    std::unique_ptr<P2PCommunicator> &communicator,
                    DistributedSearchMode dist_search_mode,
                    SearchThreadMode search_thread_mode,
      pipeann::IndexBuildParameters *parameters = nullptr,
      uint64_t max_queries_balance = 8, bool use_batching = false,
      uint64_t max_batch_size = 0, bool use_counter_thread = false,
      std::string counter_csv = "", uint64_t counter_sleep_ms = 500,
		    bool use_logging = false, const std::string &log_file = "", bool use_mem_index = false);
  ~SSDPartitionIndex();

  // returns region of `node_buf` containing [COORD(T)]
  inline T *offset_to_node_coords(const char *node_buf) {
    return (T *)node_buf;
  }

  // returns region of `node_buf` containing [NNBRS][NBR_ID(uint32_t)]
  inline unsigned *offset_to_node_nhood(const char *node_buf) {
    return (unsigned *)(node_buf + data_dim * sizeof(T));
  }

  // obtains region of sector containing node
  inline char *offset_to_node(const char *sector_buf, uint32_t node_id) {
    return offset_to_loc(sector_buf, id2loc(node_id));
  }

  // sector # on disk where node_id is present
  inline uint64_t node_sector_no(uint32_t node_id) {
    return loc_sector_no(id2loc(node_id));
  }

  inline uint64_t u_node_offset(uint32_t node_id) {
    return u_loc_offset(id2loc(node_id));
  }

  // unaligned offset to location
  inline uint64_t u_loc_offset(uint64_t loc) {
    return loc * max_node_len; // compacted store.
  }

  inline uint64_t u_loc_offset_nbr(uint64_t loc) {
    return loc * max_node_len + data_dim * sizeof(T);
  }

  inline char *offset_to_loc(const char *sector_buf, uint64_t loc) {
    return (char *)sector_buf +
           (nnodes_per_sector == 0 ? 0
                                   : (loc % nnodes_per_sector) * max_node_len);
  }

  // avoid integer overflow when * SECTOR_LEN.
  inline uint64_t loc_sector_no(uint64_t loc) {
    return 1 + (nnodes_per_sector > 0
                    ? loc / nnodes_per_sector
                    : loc * DIV_ROUND_UP(max_node_len, SECTOR_LEN));
  }

  inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) {
    return (sector_no - 1) * nnodes_per_sector + sector_off;
  }

  static constexpr uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();

  // TODO:load function needs load the cluster index mapping file.
  libcuckoo::cuckoohash_map<uint32_t, uint32_t>
      id2loc_; // id -> loc (start from 0)

  // mapping from node id to actual index in order it was stored on
  // disk
  TagT id2loc(uint32_t id) {
    if (!enable_locs)
      return id;

    uint32_t loc = 0;
    if (id2loc_.find(id, loc)) {
      return loc;
    } else {
      LOG(ERROR) << "id " << id << " not found in id2loc";
      crash();
      return kInvalidID;
    }
  }

  libcuckoo::cuckoohash_map<uint32_t, TagT> tags;
  TagT id2tag(uint32_t id) {
#ifdef NO_MAPPING
    return id; // use ID to replace tags.
#else
    TagT ret;
    if (tags.find(id, ret)) {
      return ret;
    } else {
      return id;
    }
#endif
  }
  int load(const char *index_prefix, bool new_index_format = true);

  void load_mem_index(pipeann::Metric metric, const size_t query_dim,
                      const std::string &mem_index_path);

  void load_tags(const std::string &tag_file, size_t offset = 0);

  std::vector<uint32_t> get_init_ids() {
    return std::vector<uint32_t>(this->medoids,
                                 this->medoids + this->num_medoids);
  }

  std::pair<uint8_t *, uint32_t> get_pq_config() {
    return std::make_pair(this->data.data(), (uint32_t)this->n_chunks);
  }

  uint64_t get_num_frozen_points() { return this->num_frozen_points; }

  uint64_t get_frozen_loc() { return this->frozen_location; }

  uint8_t get_partition_assignment(uint32_t node_id) {
    if (dist_search_mode != DistributedSearchMode::STATE_SEND &&
        dist_search_mode != DistributedSearchMode::STATE_SEND_CLIENT_GATHER &&
        dist_search_mode != DistributedSearchMode::DISTRIBUTED_ANN)
      return my_partition_id;
    return partition_assignment[node_id];
  }

  uint64_t get_data_dim() { return this->data_dim; }

private:
  // loading functions
  void load_disk_index(const std::string &disk_index_file, bool new_index_format);
  /*load the partition_ids_uint32.bin file */
  void load_locs(const std::string& loc_file);

  /*load pq data and setup the table*/
  void load_pq(const std::string &pq_compressed_vectors, const std::string &pq_table_bin);

  /*used for mips to load max norm file*/
  void load_max_norm(const std::string &max_norm_file);

  /*used by statesend + client gather and distributedann orchestration server*/
  void load_partition_assignment(const std::string &partition_assignment_file);
  


private:
  uint8_t my_partition_id;
  // index info
  // nhood of node `i` is in sector: [i / nnodes_per_sector]
  // offset in sector: [(i % nnodes_per_sector) * max_node_len]
  // nnbrs of node `i`: *(unsigned*) (buf)
  // nbrs of node `i`: ((unsigned*)buf) + 1
  uint64_t max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

  uint64_t global_graph_num_points = 0;
  // data info
  uint64_t num_points = 0;
  uint64_t init_num_pts = 0;
  uint64_t num_frozen_points = 0;
  uint64_t frozen_location = 0;
  uint64_t data_dim = 0;
  uint64_t aligned_dim = 0;
  uint64_t size_per_io = 0;

  double query_norm;

  std::string _disk_index_file;

  std::shared_ptr<AlignedFileReader> &reader;

  // PQ data
  // n_chunks = # of chunks ndims is split into
  // data: uint8_t * n_chunks
  // chunk_size = chunk size of each dimension chunk
  // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
  std::vector<uint8_t> data;
  uint64_t chunk_size;
  uint64_t n_chunks;
  pipeann::FixedChunkPQTable<T> pq_table;

  // distance comparator
  pipeann::Metric metric;
  std::shared_ptr<pipeann::Distance<T>> dist_cmp;

  // used only for inner product search to re-scale the result value
  // (due to the pre-processing of base during index build)
  float _max_base_norm = 0.0f;

  // Are we dealing with normalized data? This will be true
  // if distance == COSINE and datatype == float. Required
  // because going forward, we will normalize vectors when
  // asked to search with COSINE similarity. Of course, this
  // will be done only for floating point vectors.
  bool data_is_normalized = false;

  // medoid/start info
  uint32_t *medoids =
      nullptr; // by default it is just one entry point of graph, we
  // can optionally have multiple starting points
  size_t num_medoids = 1; // by default it is set to 1

  // test the estimation efficacy.
  uint32_t beamwidth, l_index, range, maxc;
  float alpha;
  // assumed max thread, only the first nthreads are initialized.

  bool load_flag = false;   // already loaded.
  
  // each of the variables below correspond to a load function
  bool enable_mem_index = false;
  bool enable_disk_index = false;
  bool enable_tags = false; 
  bool enable_locs = false; 
  bool enable_pq = false;
  bool enable_max_norm = false;
  bool enable_partition_assignment=false;
  

  std::atomic<uint64_t> cur_id, cur_loc;
  static constexpr uint32_t kMaxElemInAPage = 16;

  std::atomic<uint64_t> current_search_thread_index{0};

  std::vector<uint8_t> partition_assignment;

  std::unique_ptr<pipeann::Index<T, TagT>> mem_index_;

  bool use_batching = false;
  uint64_t max_batch_size = 0;

  std::atomic<uint64_t> num_foreign_states_global_queue = 0;
  std::atomic<uint64_t> num_new_states_global_queue = 0;

private:
  DistributedSearchMode dist_search_mode;
  SearchThreadMode search_thread_mode;
  // section is for commmunication
  std::unique_ptr<P2PCommunicator> &communicator;
  std::vector<uint64_t> other_peer_ids;

private:
  PreallocatedQueue<SearchState<T>> preallocated_state_queue;
  PreallocatedQueue<QueryEmbedding<T>> preallocated_query_emb_queue;
  PreallocatedQueue<search_result_t> preallocated_result_queue;

private:
  /**
     notify based on client peer id
   */
  void notify_client_tcp(SearchState<T, TagT> *search_state);

  /**
     STATE WILL BE DELETED
   */
  void notify_client(SearchState<T, TagT> *search_state);

private:
  /**
     STATE WILL BE DELETED
     serialize the data from state, can delete after function call. need to
     check that the top candidate node is offserver first as precondition.
   */
  void send_state(SearchState<T, TagT> *search_state,
                  uint64_t recipient_peer_id);

private:
  std::atomic<uint64_t> msg_received_id = 0;

  /*
    used to store the pointers in handler
   */
  std::array<SearchState<T, TagT> *, MAX_ELEMENTS_HANDLER> state_scratch;
  std::array<QueryEmbedding<T> *, MAX_ELEMENTS_HANDLER> query_scratch;

public:
  /**
   * will be registered to the communicator by the server cpp file.
   * Need to construct the states and enqueue them onto the search thread
   * from these handler
   */
  void receive_handler(const char *buffer, size_t size);

private:
  void compute_pq_dists(float *pq_dists, uint8_t *pq_coord_scratch,
                        const unsigned *ids, const uint64_t n_ids,
                        float *dists_out) {
    ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks,
                       pq_coord_scratch);
    ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                     dists_out);
  }
};
