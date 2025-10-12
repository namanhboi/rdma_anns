#include "communicator.h"
#include "blockingconcurrentqueue.h"
#include "types.h"
#include "libcuckoo/cuckoohash_map.hh"

template<typename T>
class StateSendClient {
private:
  std::unique_ptr<P2PCommunicator> communicator;
  libcuckoo::cuckoohash_map<uint64_t, std::chrono::steady_clock::time_point>
      query_send_time;
  libcuckoo::cuckoohash_map<uint64_t, std::chrono::steady_clock::time_point>
      query_result_time;
  std::atomic<uint64_t> num_results_received;

  libcuckoo::cuckoohash_map<uint64_t, std::shared_ptr<int>> results;


  // id in the communicator json file containing all the ip addresses
  uint64_t my_id;
  std::vector<uint64_t> other_peer_ids;


  std::atomic<uint64_t> current_round_robin_peer_index{0};

  std::atomic<uint64_t>
      query_id; // the search thread gets the query id via this

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

    // idnex into other_peer_ids
    uint64_t current_round_robin_peer_index;
  public:
    ClientThread(uint64_t id, StateSendClient *parent);
    void push_query(std::shared_ptr<QueryEmbedding<T>> query);
    void start();
    void signal_stop();
    void join();
  };

  std::vector<std::unique_ptr<ClientThread>> client_threads;
  int num_client_threads;


  std::atomic<uint64_t> current_client_thread_id;

private:
  uint64_t dim;
public:
  StateSendClient(const uint64_t id, const std::string &communicator_json, int num_client_thread, uint64_t dim);
  ~StateSendClient();


  void start_result_thread();
  void start_client_threads();

  void search(const T *query_emb, const uint64_t k_search,
              const uint64_t mem_l, const uint64_t l_search,
              const uint64_t beam_width);

  void wait_results(const uint64_t num_results);

  /*
    logs the time received and also save the result for comparison later
   */
  void receive_result_handler(const char *buffer, size_t size);
  void shutdown();
};

