#include "communicator.h"



template<typename T>
class StateSendClient {
private:
  std::unique_ptr<P2PCommunicator> communicator;
  std::unordered_map<uint64_t, std::chrono::steady_clock::time_point>
      query_send_time;
  std::unordered_map<uint64_t, std::chrono::steady_clock::time_point>
      query_result_time;


  // id in the communicator json file containing all the ip addresses
  uint64_t my_id;
  std::vector<uint64_t> other_peer_ids;

  // idnex into other_peer_ids
  std::atomic<uint64_t> current_round_robin_peer_index;
public:
  StateSendClient(const uint64_t id, const std::string &communicator_json);
  void start();
  /**
     search is responsible for creating the query id and sending the state constructed from the query to the server
  */
  void search(const T *query_emb, const uint64_t k_search, const uint64_t mem_l,
              const uint64_t l_search, const uint64_t beam_width);
  void wait_results();
  void receive_handler(const char *buffer, size_t size);
  void signal_stop();
};

