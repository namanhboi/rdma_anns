#include "blockingconcurrentqueue.h"
#include "communicator.h"
#include "concurrentqueue.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "types.h"

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
    moodycamel::BlockingConcurrentQueue<std::shared_ptr<QueryEmbedding<T>>>
        concurrent_query_queue;
    void main_loop();
    StateSendClient *parent;
    std::atomic<bool> running{false};
  public:
    ClientThread(uint64_t id, StateSendClient *parent);
    void push_query(std::shared_ptr<QueryEmbedding<T>> query);
    void start();
    void signal_stop();
    void join();
  };

  std::vector<std::unique_ptr<ClientThread>> client_threads;
  int num_client_threads;

  std::atomic<uint64_t> current_client_thread_id={0};

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
  StateSendClient(const uint64_t id, const std::string &communicator_json,
                  int num_client_thread, DistributedSearchMode dist_search_mode,
                  uint64_t dim);

  StateSendClient(const uint64_t id, const std::vector<std::string> &address_list,
                  int num_client_thread, DistributedSearchMode dist_search_mode,
                  uint64_t dim);  
  /**
     shuts down every thing correctly,
   */

  void start_result_thread();
  void start_client_threads();

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


  void send_acks(std::shared_ptr<search_result_t> result);

  void shutdown();

};


