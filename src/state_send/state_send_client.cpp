#include "state_send_client.h"
#include "types.h"
#include <chrono>
#include <thread>

template <typename T>
StateSendClient<T>::ClientThread::ClientThread(uint64_t id,
                                               StateSendClient<T> *parent)
    : my_thread_id(id), parent(parent) {}

template <typename T> void StateSendClient<T>::ClientThread::main_loop() {
  std::vector<std::shared_ptr<QueryEmbedding<T>>> batch_of_queries;
  constexpr int max_batch_size = 16;
  batch_of_queries.reserve(max_batch_size);
  while (running) {
    size_t num_dequeued = concurrent_query_queue.wait_dequeue_bulk(
        batch_of_queries.begin(), max_batch_size);
    assert(num_dequeued == batch_of_queries.size());
    // check for poison pill
    for (auto i = 0; i < num_dequeued; i++) {
      if (batch_of_queries[i] == nullptr) {
        batch_of_queries.erase(batch_of_queries.begin() + i);
        break;
      }
    }
    if (batch_of_queries.size() == 0) {
      assert(running == false);
      break;
    }
    size_t region_size =
        sizeof(MessageType::QUERIES) +
        QueryEmbedding<T>::get_serialize_size_queries(batch_of_queries);
    MessageType msg_type = MessageType::QUERIES;
    Region r;
    r.length = region_size;
    r.addr = new char[region_size];

    size_t offset = 0;
    std::memcpy(r.addr, &msg_type, sizeof(msg_type));
    offset += sizeof(msg_type);

    QueryEmbedding<T>::write_serialize_queries(r.addr + offset,
                                               batch_of_queries);
    uint32_t server_peer_id =
        parent->current_round_robin_peer_index.fetch_add(1) %
        parent->other_peer_ids.size();

    parent->communicator->send_to_peer(parent->other_peer_ids[server_peer_id],
                                       r);
    

    batch_of_queries.clear();
  }
}

template <typename T>
void StateSendClient<T>::ClientThread::push_query(
    std::shared_ptr<QueryEmbedding<T>> query) {
  parent->query_send_time.insert_or_assign(query->query_id, std::chrono::steady_clock::now());
  concurrent_query_queue.enqueue(query);
}

template <typename T>
uint64_t StateSendClient<T>::search(const T *query_emb, const uint64_t k_search,
                                const uint64_t mem_l, const uint64_t l_search,
                                const uint64_t beam_width) {
  std::shared_ptr<SearchState<T>> state = std::make_shared<SearchState<T>>();
  uint64_t query_id = this->query_id.fetch_add(1);
  std::shared_ptr<QueryEmbedding<T>> query =
      std::make_shared<QueryEmbedding<T>>();

  query->query_id = query_id;
  query->client_peer_id = my_id;
  query->mem_l = mem_l;
  query->l_search = l_search;
  query->k_search = k_search;
  query->beam_width = beam_width;
  query->dim = this->dim;
  query->num_chunks = 0;
  std::memcpy(query->query, query_emb, sizeof(T) * query->dim);
  uint64_t next_client_thread_id =
      current_client_thread_id.fetch_add(1) % num_client_threads;
  client_threads[next_client_thread_id]->push_query(query);
  return query_id;
}

template <typename T> void StateSendClient<T>::ClientThread::start() {
  real_thread = std::thread(&StateSendClient<T>::ClientThread::main_loop, this);
  running = true;
}

template <typename T> void StateSendClient<T>::ClientThread::signal_stop() {
  running = false;
  concurrent_query_queue.enqueue(nullptr);
}

template <typename T> void StateSendClient<T>::ClientThread::join() {
  if (real_thread.joinable())
    real_thread.join();
}


template <typename T>
StateSendClient<T>::StateSendClient(const uint64_t id,
                                    const std::string &communicator_json,
                                    int num_client_threads, uint64_t dim)
    : my_id(id), num_client_threads(num_client_threads), dim(dim) {
  communicator = std::make_unique<ZMQP2PCommunicator>(my_id, communicator_json);
  std::cout << "Done with constructor for statesendclient" << std::endl;
  other_peer_ids = communicator->get_other_peer_ids();
  communicator->register_receive_handler(
      [this](const char *buffer, size_t size) {
        this->receive_result_handler(buffer, size);
      });

  for (uint64_t i = 0; i < num_client_threads; i++) {
    client_threads.emplace_back(std::make_unique<ClientThread>(i, this));
  }
}

template <typename T> void StateSendClient<T>::start_result_thread() {
  communicator->start_recv_thread();
}

template <typename T> void StateSendClient<T>::start_client_threads() {
  for (auto &client_thread : client_threads) {
    client_thread->start();
  }
}

// template <typename T>
// void StateSendClient<T>::search(const T *query_emb, const uint64_t k_search,
//                                 const uint64_t mem_l, const uint64_t
//                                 l_search, const uint64_t beam_width) {
//   uint64_t server_peer_id_index =
//     this->current_round_robin_peer_index.fetch_add(1) %
//     other_peer_ids.size();
//   uint64_t server_peer_id = other_peer_ids[server_peer_id_index];
// }

template <typename T>
void StateSendClient<T>::wait_results(const uint64_t num_results) {
  while (num_results_received != num_results) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

template <typename T>
std::shared_ptr<search_result_t> StateSendClient<T>::get_result(const uint64_t query_id) {
  return results.find(query_id);
}

template <typename T>
double StateSendClient<T>::get_query_latency_milli(const uint64_t query_id) {
  auto sent = query_send_time.find(query_id);
  auto received = query_result_time.find(query_id);
  std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(received - sent);

  double lat = static_cast<double>(elapsed.count()) / 1000.0;
  return lat;
}


template <typename T>
std::chrono::steady_clock::time_point
StateSendClient<T>::get_query_send_time(const uint64_t query_id) {
  return query_send_time.find(query_id);
}

template <typename T>
std::chrono::steady_clock::time_point
StateSendClient<T>::get_query_result_time(const uint64_t query_id) {
  return query_result_time.find(query_id);
}

template <typename T>
void StateSendClient<T>::receive_result_handler(const char *buffer,
                                                size_t size) {
  size_t offset = 0;
  MessageType msg_type;
  std::memcpy(&msg_type, buffer, sizeof(msg_type));
  assert(msg_type == MessageType::RESULT);
  offset += sizeof(msg_type);

  std::shared_ptr<search_result_t> res =
    search_result_t::deserialize(buffer + offset);
  query_result_time.insert_or_assign(res->query_id,
                                     std::chrono::steady_clock::now());
  results.insert_or_assign(res->query_id, res);
  num_results_received.fetch_add(1);
}

template <typename T> void StateSendClient<T>::shutdown() {
  communicator->stop_recv_thread();
  for (auto &client_thread : client_threads) {
    client_thread->signal_stop();
  }
  for (auto &client_thread : client_threads) {
    client_thread->join();
  }
}


template <typename T> StateSendClient<T>::~StateSendClient() {
  shutdown();
}


template class StateSendClient<uint8_t>;
template class StateSendClient<int8_t>;
template class StateSendClient<float>;
