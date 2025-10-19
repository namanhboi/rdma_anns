#pragma once

#include "blockingconcurrentqueue.h"
#include "concurrentqueue.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "linux_aligned_file_reader.h"
#include "neighbor.h"
#include "parameters.h"
#include "pq_table.h"
#include "query_buf.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include <condition_variable>
#include <cstdint>
#include <immintrin.h>
#include <set>
#include <string>
#include <variant>
#include "communicator.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "types.h"
#include "index.h"
#include <unordered_set>

#define MAX_N_CMPS 16384
#define MAX_N_EDGES 512
#define MAX_PQ_CHUNKS 128
#define SECTOR_LEN 4096

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

#define MAX_SEARCH_THREADS 64

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
  libcuckoo::cuckoohash_map<uint64_t, std::shared_ptr<QueryEmbedding<T>>>
      query_emb_map;
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
   */
  void state_update_frontier(SearchState<T, TagT> *state);

  void state_compute_and_add_to_retset(SearchState<T, TagT> *state,
                                       const unsigned *node_ids,
                                       const uint64_t n_ids);

  bool state_issue_next_io_batch(SearchState<T, TagT> *state, void *ctx);
  
    /**
       advances the state based on whatever is in frontier_nhoods, which is the
       result of reading what's in the frontier.
       It also updates the frontier after exploring what's in frontier_nhoods.
       Based on the state of the frontier, it can return the corresponding SearchExecutionState
     */
    SearchExecutionState state_explore_frontier(SearchState<T, TagT> *state);

  bool state_search_ends(SearchState<T, TagT> *state);

  void apply_tags_to_result(std::shared_ptr<search_result_t> result);

  // static void write_serialize_query(const T *query_emb,
  //                                   const uint64_t k_search,
  //                                   const uint64_t mem_l,
  //                                   const uint64_t l_search,
  //                                   const uint64_t beam_width, char *buffer)
  //                                   {
  // 
  // }
  
  uint8_t state_top_cand_partition(SearchState<T, TagT> *state);


  void state_print_detailed(SearchState<T, TagT> *state);
  void query_emb_print(std::shared_ptr<QueryEmbedding<T>> query_emb);
  
private:
  static constexpr uint64_t max_queries_balance = 128;
  uint64_t num_queries_balance = 0;
  bool record_stats;

  class SearchThread {
    SSDPartitionIndex *parent;
    std::thread real_thread;
    // id used so that parent can send queries round robin
    uint64_t thread_id;
    std::atomic<bool> running{false};
    void *ctx = nullptr;

    moodycamel::ConsumerToken search_thread_consumer_token;

    moodycamel::ConcurrentQueue<SearchState<T, TagT> *> thread_state_queue;

    /**
       main loop that runs the search. This version balances all queries at
       once, resulting in poor qps
     */
    void main_loop_balance_all();

    /**
       main loop that runs the search. This version only balances batch_size queries at a
       time. 
     */
    void main_loop_batch();
    friend class SSDPartitionIndex; // to access ctx of class to send io
  public:
    /**
       will run the search, won't contain a queue/have any way of directly
       issueing a query. Instead, wait on io requests
     */
    SearchThread(SSDPartitionIndex *parent, uint64_t thread_id);
    void push_state(SearchState<T, TagT> *new_state);
    void start();
    void signal_stop();
    void join();
  };

  moodycamel::ConcurrentQueue<SearchState<T, TagT> *> global_state_queue;
  moodycamel::ProducerToken client_state_prod_token;
  moodycamel::ProducerToken server_state_prod_token;

  std::vector<std::unique_ptr<SearchThread>> search_threads;
  uint32_t num_search_threads;
  std::atomic<int> current_search_thread_id = 0;

private:
  /**
     handles commmunication with servers and clients, note that if you enqueue a
     state, it will be deleted by btaching thread, don't double delete it yourself.
   */
  class BatchingThread {
  private:
    SSDPartitionIndex *parent;

    std::thread real_thread;
    std::atomic<bool> running = false;


    std::unordered_map<uint64_t,
                       std::unique_ptr<std::vector<SearchState<T, TagT> *>>>
        msg_queue;
    std::unordered_set<uint64_t> peer_client_ids;
    std::condition_variable msg_queue_cv;
    std::mutex msg_queue_mutex; // also manages is_peer_client
    

    void main_loop();
  public:
    BatchingThread(SSDPartitionIndex *parent);
    void push_result_to_batch(SearchState<T, TagT> *state);
    void push_state_to_batch(SearchState<T, TagT> *state);
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
                    uint32_t num_partitions, uint32_t num_search_threads,
                    std::shared_ptr<AlignedFileReader> &fileReader,
                    std::unique_ptr<P2PCommunicator> &communicator,
                    DistributedSearchMode dist_search_mode, bool tags = false,
                    pipeann::Parameters *parameters = nullptr,
                    uint64_t max_queries_balance = 8, bool enable_locs = true,
                    bool use_batching = false, uint64_t max_batch_size = 0);
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
    // num partition = 1 means that there is no mapping file
    if (!enable_locs)
      return id;
    
    if (num_partitions == 1) {
      return id;
    }
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

  // load compressed data, and obtains the handle to the disk-resident index
  // also loads in the paritition index mapping file if num_partitions > 1
  int load(const char *index_prefix, bool new_index_format = true,
           const char *cluster_assignment_file = nullptr);

  void load_mem_index(pipeann::Metric metric, const size_t query_dim,
                      const std::string &mem_index_path);

  void load_tags(const std::string &tag_file, size_t offset = 0);

  std::vector<uint32_t> get_init_ids() {
    return std::vector<uint32_t>(this->medoids,
                                 this->medoids + this->num_medoids);
  }

  // computes PQ dists between src->[ids] into fp_dists (merge, insert)
  void compute_pq_dists(const uint32_t src, const uint32_t *ids,
                        float *fp_dists, const uint32_t count,
                        uint8_t *aligned_scratch = nullptr);

  std::pair<uint8_t *, uint32_t> get_pq_config() {
    return std::make_pair(this->data.data(), (uint32_t)this->n_chunks);
  }

  uint64_t get_num_frozen_points() { return this->num_frozen_points; }

  uint64_t get_frozen_loc() { return this->frozen_location; }

  inline uint8_t get_cluster_assignment(uint32_t node_id) {
    if (num_partitions == 1) return my_partition_id;
    return cluster_assignment[node_id];
  }

  uint64_t get_data_dim() {return this->data_dim;}

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

  uint32_t num_partitions;

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
  std::shared_ptr<pipeann::Distance<T>> dist_cmp;

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
  bool enable_tags = false; // support for tags and dynamic indexing
  bool enable_locs = true; // support for loc files

  std::atomic<uint64_t> cur_id, cur_loc;
  static constexpr uint32_t kMaxElemInAPage = 16;

  std::atomic<uint64_t> current_search_thread_index{0};

  std::vector<uint8_t> cluster_assignment;


  std::unique_ptr<pipeann::Index<T, TagT>> mem_index_;

  bool use_batching = false;
  uint64_t max_batch_size = 0;
private:
  DistributedSearchMode dist_search_mode;
  // section is for commmunication
  std::unique_ptr<P2PCommunicator> &communicator;
  
private:
  /**
     notify based on client peer id
   */
  void notify_client_tcp(SearchState<T, TagT> *search_state);
  /**
     writes results to the res_tags and res_dists that client specified and
     increment completion count. Deallocates the search_state as well.
  */
  void notify_client_local(SearchState<T, TagT> *search_state);

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
  void send_state(SearchState<T, TagT> *search_state);
  
public:
  /**
   * will be registered to the communicator by the server cpp file.
   * Need to construct the states and enqueue them onto the search thread
   * from these handler
   */
  void receive_handler(const char* buffer, size_t size);

public:
  /**
     right now we search the disk index directly without going through the in
     mem index. After query is done, increment completion_count
   */
  void search_ssd_index_local(
      const T *query_emb, const uint64_t query_id, const uint64_t k_search,
      const uint32_t mem_L, const uint64_t l_search, TagT *res_tags,
      float *res_dists, const uint64_t beam_width,
			      std::shared_ptr<std::atomic<uint64_t>> completion_count);

  // void search_ssd_index(const T *query_emb, const uint64_t k_search,
                        // const uint32_t mem_L, const uint64_t l_search,
                        // const uint64_t beam_width, const uint64_t peer_id);
};
