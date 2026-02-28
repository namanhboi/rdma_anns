#include "blockingconcurrentqueue.h"
#include "communicator.h"
#include "concurrentqueue.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "types.h"
#include <condition_variable>
#include <unordered_set>
#include "distance.h"

constexpr uint32_t MAX_ELEMENTS_HANDLER_CLIENT  =256;

template <typename T> class StateSendClient {
private:
  std::unique_ptr<P2PCommunicator> communicator;

  libcuckoo::cuckoohash_map<uint64_t, std::chrono::steady_clock::time_point>
      query_send_time;
  libcuckoo::cuckoohash_map<uint64_t, std::chrono::steady_clock::time_point>
      query_result_time;
  std::atomic<uint64_t> num_results_received={0};


  libcuckoo::cuckoohash_map<uint64_t, search_result_t*> results;


  /////// used for scatter gather only
  libcuckoo::cuckoohash_map<
      uint64_t,
      std::vector<std::pair<uint8_t, std::chrono::steady_clock::time_point>>>
      sub_query_result_time;
  libcuckoo::cuckoohash_map<
      uint64_t,
      std::vector<std::pair<uint8_t, search_result_t*>>>
      sub_query_results;
  ///////////

  // used to store state send client gather results
  libcuckoo::cuckoohash_map<uint64_t, client_gather_results_t>
      client_gather_results;

  // id in the communicator json file containing all the ip addresses
  uint64_t my_id;
  std::vector<uint64_t> other_peer_ids;

  std::atomic<uint64_t> current_round_robin_peer_index{0};

  std::atomic<uint64_t>
      query_id={0}; // the search thread gets the query id via this

private:
  class ClientThread {
  private:
    std::thread real_thread;
    uint64_t my_thread_id; // used for round robin querying the server



    void main_loop();
    StateSendClient *parent;
    std::atomic<bool> running{false};
  public:
    ClientThread(uint64_t id, StateSendClient *parent);
    void start();
    void signal_stop();
    void join();
  };

  std::vector<std::unique_ptr<ClientThread>> client_threads;
  int num_client_threads;

  std::atomic<uint64_t> current_client_thread_id = {0};

private:
  moodycamel::BlockingConcurrentQueue<std::shared_ptr<QueryEmbedding<T>>>
      concurrent_query_queue;

private:
  /**
     for distributedann comparison
   */
  uint8_t num_partitions;
  std::vector<std::vector<uint8_t>> partition_assignment;
  uint8_t get_random_partition_assignment(uint32_t node_id) {
    static thread_local std::random_device dev;
    static thread_local std::mt19937 gen(dev());

    std::uniform_int_distribution<uint8_t> distrib(
						   0, partition_assignment[node_id].size() - 1);
    return partition_assignment[node_id][distrib(gen)];
  }
  
  
  class OrchestrationThread {
  private:
    std::thread real_thread;
    StateSendClient *parent;
    std::atomic<bool> running{false};
    moodycamel::ConsumerToken ctok;

    /**
       using the state's frontier, send data
     */
    size_t
    send_scoring_queries(const distributedann::DistributedANNState<T> *state,
                         const std::shared_ptr<QueryEmbedding<T>> &query,
                         float threshold,
                         std::vector<uint8_t> &partitions_with_emb);
    
    void
    search_query(std::shared_ptr<QueryEmbedding<T>> query, search_result_t* result);
    void main_loop();
  public:
    OrchestrationThread(StateSendClient *parent);
    void start();
    void signal_stop();
    void join();
  };

  uint32_t num_orchestration_threads;
  std::vector<std::unique_ptr<OrchestrationThread>> orchestration_threads;


  // number of states = number of orchestration threads since 1 thread will work
  // on a query at a time
  PreallocatedQueue<distributedann::DistributedANNState<T>> prealloc_states;

  // used to avoid dynamically allocating result when recived message from
  // handler
  PreallocatedQueue<distributedann::result_t<T>> prealloc_result;


  /*
    used to store the pointers in handler
   */
  std::array<distributedann::result_t<T> *, MAX_ELEMENTS_HANDLER_CLIENT>
      handler_result_scratch;
public:
  void distributed_ann_receive_handler(const char *buffer, size_t size);
    
private:
  class ResultReceiveThread {
  private:
    
    std::thread real_thread;
    uint64_t my_thread_id;

    // used for STATE_SEND, DISTRIBUTED_ANN, SINGLE_SERVER
    void process_singular_result(search_result_t* res);

    // used for SCATTER_GATHER
    void process_scatter_gather_result(search_result_t* res);

    // used by STATE_SEND_CLIENT_GATHER
    void process_state_send_client_gather_result(search_result_t* res);
    
    void main_loop();
    std::atomic<bool> running{false};

    StateSendClient *parent;
    // moodycamel::ConsumerToken ctok;
  public:
    ResultReceiveThread(StateSendClient *parent);
    void start();
    void signal_stop();
    void join();
  };
  std::unique_ptr<ResultReceiveThread> result_thread;
private:
  DistributedSearchMode dist_search_mode;

  // top_n for SCATTER_GATHER_TOP_N, other_peer_ids.size() for SCATTER_GATHER
  uint32_t num_results_to_expect = std::numeric_limits<uint32_t>::max();

  // moodycamel::ProducerToken ptok;
  moodycamel::BlockingConcurrentQueue<search_result_t *> result_queue;

private:
  size_t dim;
  T *partition_medoids;
  pipeann::Distance<T> *distance_fn; 
  
public:
  StateSendClient(const uint64_t id,
                  const std::vector<std::string> &address_list,
                  int num_worker_threads,
                  DistributedSearchMode dist_search_mode, uint64_t dim,
                  const std::string &partition_assignment_file, uint32_t top_n,
                  const std::string &medoid_file);
  PreallocatedQueue<search_result_t> prealloc_result_queue;
  PreallocatedQueue<Region> prealloc_region_queue;
  
  void start();

  uint64_t search(const T *query_emb, const uint64_t k_search, const uint64_t mem_l,
		  const uint64_t l_search, const uint64_t beam_width, bool record_stats);

  void wait_results(const uint64_t num_results);
  search_result_t* get_result(const uint64_t query_id);

  std::chrono::steady_clock::time_point
  get_query_send_time(const uint64_t query_id);

  std::chrono::steady_clock::time_point
  get_query_result_time(const uint64_t query_id);
  

  /*
    logs the time received and also save the result for comparison later
   */
  void receive_result_handler(const char *buffer, size_t size);

  /**
     send acks based on partition history of result
  */
  void send_acks(const search_result_t *result);

  void clear_results(const std::vector<search_result_t *> &results_used);

  void shutdown();

};


