#include "blockingconcurrentqueue.h"
#include "communicator.h"
#include "concurrentqueue.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "types.h"
#include <condition_variable>
#include <unordered_set>


constexpr uint32_t MAX_ELEMENTS_HANDLER_CLIENT  =256;

template <typename T> class StateSendClient {
private:
  std::unique_ptr<P2PCommunicator> communicator;

  libcuckoo::cuckoohash_map<uint64_t, std::chrono::steady_clock::time_point>
      query_send_time;
  libcuckoo::cuckoohash_map<uint64_t, std::chrono::steady_clock::time_point>
      query_result_time;
  std::atomic<uint64_t> num_results_received={0};


  libcuckoo::cuckoohash_map<uint64_t, std::shared_ptr<search_result_t>> results;


  /////// used for scatter gather only
  libcuckoo::cuckoohash_map<
      uint64_t,
      std::vector<std::pair<uint8_t, std::chrono::steady_clock::time_point>>>
      sub_query_result_time;
  libcuckoo::cuckoohash_map<
      uint64_t,
      std::vector<std::pair<uint8_t, std::shared_ptr<search_result_t>>>>
      sub_query_results;
  ///////////



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
    
    std::shared_ptr<search_result_t>
    search_query(std::shared_ptr<QueryEmbedding<T>> query);
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
  uint64_t dim;

  // moodycamel::ProducerToken ptok;
  moodycamel::BlockingConcurrentQueue<std::shared_ptr<search_result_t>>
      result_queue;
public:
  StateSendClient(const uint64_t id,
                  const std::vector<std::string> &address_list,
                  int num_worker_threads,
                  DistributedSearchMode dist_search_mode, uint64_t dim,
                  const std::string &partition_assignment_file);

  // StateSendClient(const uint64_t id,
                  // const std::vector<std::string> &address_list,
                  // int num_orchestration_thread,
                  // const std::string &partition_assignment_file,
                  // DistributedSearchMode dist_search_mode, uint64_t dim);

  /**
     shuts down every thing correctly,

   */

  void start();

  uint64_t search(const T *query_emb, const uint64_t k_search, const uint64_t mem_l,
		  const uint64_t l_search, const uint64_t beam_width, bool record_stats);

  void wait_results(const uint64_t num_results);
  std::shared_ptr<search_result_t> get_result(const uint64_t query_id);

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
  void send_acks(std::shared_ptr<search_result_t> result);

  void shutdown();

};


