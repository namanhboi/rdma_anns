#include "state_send_client.h"



template <typename T>
StateSendClient<T>::StateSendClient(const uint64_t id,
                                    const std::string &communicator_json)
: my_id(id) {
  communicator = std::make_unique<ZMQP2PCommunicator>(my_id, communicator_json);
  std::cout << "Done with constructor for statesendclient" << std::endl;
  other_peer_ids = communicator->get_other_peer_ids();
}

template <typename T> void StateSendClient<T>::start() {
  communicator->start_recv_thread();
}


template <typename T>
void StateSendClient<T>::search(const T *query_emb, const uint64_t k_search,
                                const uint64_t mem_l, const uint64_t l_search,
                                const uint64_t beam_width) {
  uint64_t server_peer_id_index =
    this->current_round_robin_peer_index.fetch_add(1) % other_peer_ids.size();
  uint64_t server_peer_id = other_peer_ids[server_peer_id_index];
    
}

template <typename T> void StateSendClient<T>::wait_results() {
    
}


template <typename T>
void StateSendClient<T>::receive_handler(const char *buffer, size_t size) {}


template <typename T> void StateSendClient<T>::signal_stop() {
  communicator->stop_recv_thread();
}


