#include "state_send_client.h"
#include "types.h"
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>
#include <unordered_set>

template <typename T>
StateSendClient<T>::ClientThread::ClientThread(uint64_t id,
                                               StateSendClient<T> *parent)
    : my_thread_id(id), parent(parent) {}

template <typename T> void StateSendClient<T>::ClientThread::main_loop() {
  std::vector<std::shared_ptr<QueryEmbedding<T>>> batch_of_queries;
  constexpr int max_batch_size = 1;
  batch_of_queries.reserve(max_batch_size + 1);
  // std::cout << "main loop started for client thread " << std::endl;
  while (running) {
    batch_of_queries.resize(max_batch_size + 1);
    size_t num_dequeued = concurrent_query_queue.wait_dequeue_bulk(
        batch_of_queries.begin(), max_batch_size);
    batch_of_queries.resize(num_dequeued);
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
    if (parent->dist_search_mode == DistributedSearchMode::STATE_SEND ||
        parent->dist_search_mode == DistributedSearchMode::SINGLE_SERVER) {
      uint32_t server_peer_id =
          parent->current_round_robin_peer_index.fetch_add(1) %
          parent->other_peer_ids.size();
      for (const auto &query : batch_of_queries) {
        parent->query_send_time.insert_or_assign(
            query->query_id, std::chrono::steady_clock::now());
      }
      parent->communicator->send_to_peer(parent->other_peer_ids[server_peer_id],
                                         r);
    } else if (parent->dist_search_mode ==
               DistributedSearchMode::SCATTER_GATHER) {

      for (const auto &query : batch_of_queries) {
        parent->query_send_time.insert_or_assign(
            query->query_id, std::chrono::steady_clock::now());
      }
      for (uint64_t i = 0; i < parent->other_peer_ids.size(); i++) {
        if (i == parent->other_peer_ids.size() - 1) {
          // don't need to make an additional copy of r
          // std::cout << "sending query" <<std::endl;
          parent->communicator->send_to_peer(parent->other_peer_ids[i], r);
        } else {
          Region r_copy;
          r_copy.length = r.length;
          r_copy.context = r.context;
          r_copy.lkey = r.lkey;
          r_copy.addr = new char[r_copy.length];
          std::memcpy(r_copy.addr, r.addr, r.length);
          parent->communicator->send_to_peer(parent->other_peer_ids[i], r_copy);
        }
      }
    }
  }
}

template <typename T>
void StateSendClient<T>::ClientThread::push_query(
    std::shared_ptr<QueryEmbedding<T>> query) {

  // std::cout << "pushed query " << query->query_id << std::endl;
  concurrent_query_queue.enqueue(query);
}

template <typename T>
uint64_t
StateSendClient<T>::search(const T *query_emb, const uint64_t k_search,
                           const uint64_t mem_l, const uint64_t l_search,
                           const uint64_t beam_width, bool record_stats) {
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
  query->record_stats = record_stats;
  std::memcpy(query->query, query_emb, sizeof(T) * query->dim);
  uint64_t next_client_thread_id =
      current_client_thread_id.fetch_add(1) % num_client_threads;
  client_threads[next_client_thread_id]->push_query(query);
  return query_id;
}

template <typename T> void StateSendClient<T>::ClientThread::start() {
  running = true;
  real_thread = std::thread(&StateSendClient<T>::ClientThread::main_loop, this);
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
                                    int num_client_threads,
                                    DistributedSearchMode dist_search_mode,
                                    uint64_t dim)
    : my_id(id), num_client_threads(num_client_threads), dim(dim),
      dist_search_mode(dist_search_mode) {
  communicator = std::make_unique<ZMQP2PCommunicator>(my_id, communicator_json);
  // std::cout << "Done with constructor for statesendclient" << std::endl;
  other_peer_ids = communicator->get_other_peer_ids();
  communicator->register_receive_handler(
      [this](const char *buffer, size_t size) {
        this->receive_result_handler(buffer, size);
      });

  for (uint64_t i = 0; i < num_client_threads; i++) {
    client_threads.emplace_back(std::make_unique<ClientThread>(i, this));
  }
  result_thread = std::make_unique<ResultReceiveThread>(this);
}

template <typename T>
StateSendClient<T>::StateSendClient(const uint64_t id,
                                    const std::vector<std::string> &address_list,
                                    int num_client_threads,
                                    DistributedSearchMode dist_search_mode,
                                    uint64_t dim)
    : my_id(id), num_client_threads(num_client_threads), dim(dim),
      dist_search_mode(dist_search_mode) {
  communicator = std::make_unique<ZMQP2PCommunicator>(my_id, address_list);
  // std::cout << "Done with constructor for statesendclient" << std::endl;
  other_peer_ids = communicator->get_other_peer_ids();
  communicator->register_receive_handler(
      [this](const char *buffer, size_t size) {
        this->receive_result_handler(buffer, size);
      });

  for (uint64_t i = 0; i < num_client_threads; i++) {
    client_threads.emplace_back(std::make_unique<ClientThread>(i, this));
  }
  result_thread = std::make_unique<ResultReceiveThread>(this);
}

template <typename T> void StateSendClient<T>::start_result_thread() {
  communicator->start_recv_thread();
  result_thread->start();
}

template <typename T> void StateSendClient<T>::start_client_threads() {
  // std::cout << "starting " << client_threads.size() << " threads" << std::endl;
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
    // std::cout << num_results_received << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  num_results_received = 0;
}

template <typename T>
std::shared_ptr<search_result_t>
StateSendClient<T>::get_result(const uint64_t query_id) {
  return results.find(query_id);
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
  // LOG(INFO) << "bruh";
  size_t offset = 0;
  MessageType msg_type;
  std::memcpy(&msg_type, buffer, sizeof(msg_type));
  assert(msg_type == MessageType::RESULT || msg_type == MessageType::RESULTS);
  offset += sizeof(msg_type);
  if (msg_type == MessageType::RESULT) {

    std::shared_ptr<search_result_t> res =
      search_result_t::deserialize(buffer + offset);
    this->result_queue.enqueue(res);
  } else if (msg_type == MessageType::RESULTS) {
    auto results = search_result_t::deserialize_results(buffer + offset);
    this->result_queue.enqueue_bulk(results.begin(), results.size());
  }
  // uint64_t query_id;
  // std::memcpy(&msg_type, buffer, sizeof(msg_type));

}

template <typename T> void StateSendClient<T>::shutdown() {
  communicator->stop_recv_thread();
  for (auto &client_thread : client_threads) {
    client_thread->signal_stop();
  }
  for (auto &client_thread : client_threads) {
    client_thread->join();
  }
  result_thread->signal_stop();
  result_thread->join();
}

template <typename T>
StateSendClient<T>::ResultReceiveThread::ResultReceiveThread(
    StateSendClient *parent)
: parent(parent) {}

template <typename T> void StateSendClient<T>::ResultReceiveThread::start() {
  running = true;
  real_thread =
      std::thread(&StateSendClient<T>::ResultReceiveThread::main_loop, this);
}

template <typename T> void StateSendClient<T>::ResultReceiveThread::join() {
  if (real_thread.joinable()) {
    real_thread.join();
  }
}

template <typename T>
void StateSendClient<T>::ResultReceiveThread::signal_stop() {
  running = false;
  parent->result_queue.enqueue(nullptr);
}

template <typename T>
void StateSendClient<T>::send_acks(std::shared_ptr<search_result_t> result) {
  std::unordered_set<uint8_t> server_peer_ids(
					      result->partition_history.cbegin(), result->partition_history.cend());
  for (const auto &server_peer_id : server_peer_ids) {
    Region r;
    MessageType msg_type = MessageType::RESULT_ACK;
    ack a;
    a.query_id = result->query_id;
    r.length = sizeof(msg_type) + a.get_serialize_size();
    r.addr = new char[r.length];
    size_t offset = 0;
    std::memcpy(r.addr + offset, &msg_type, sizeof(msg_type));
    offset += sizeof(msg_type);
    a.write_serialize(r.addr + offset);
    communicator->send_to_peer(server_peer_id, r);
  }
}


template <typename T>
void StateSendClient<T>::ResultReceiveThread::main_loop() {
  while (running) {
    std::shared_ptr<search_result_t> res;
    parent->result_queue.wait_dequeue(res);
    // LOG(INFO) << "hello";
    if (res == nullptr) {
      assert(!running);
      break;
    }
    if (parent->dist_search_mode == DistributedSearchMode::SCATTER_GATHER) {
      // LOG(INFO) << "HELLO";
      if (res->partition_history.size() != 1) {
        throw std::runtime_error("partition history size not 1: " +
                                 std::to_string(res->partition_history.size()));
      }
      // LOG(INFO) << "result received ";
      // sub_query_result_time.insert_or_assign(res->query_id,
      // std::chrono::steady_clock::now());
      // parent->sub_query_result_time.upsert(
      //     res->query_id,
      //     [res](std::vector<std::pair<
      //               uint8_t, std::chrono::steady_clock::time_point>> &vec) {
      //       vec.push_back(
      //           {res->partition_history[0], std::chrono::steady_clock::now()});

      //       return false;
      //     },
      //     std::vector<
      //         std::pair<uint8_t, std::chrono::steady_clock::time_point>>{
      //         {res->partition_history[0],
      //         std::chrono::steady_clock::now()}});
      // size_t num_res = 1;
      // parent->sub_query_results.upsert(
      //     res->query_id,
      //     [res, &num_res](
      //         std::vector<std::pair<uint8_t,
      //         std::shared_ptr<search_result_t>>>
      //             &vec) {
      //       vec.push_back({res->partition_history[0], res});
      //       num_res = vec.size();
      //       return false;
      //     },
      //     std::vector<std::pair<uint8_t, std::shared_ptr<search_result_t>>>{
      //         {res->partition_history[0], res}});

      
      // if (num_res == parent->other_peer_ids.size()) {
      //   std::shared_ptr<search_result_t> combined_res =
      //       combine_results(parent->sub_query_results.find(res->query_id));
      //   parent->results.insert_or_assign(combined_res->query_id, combined_res);

      //   parent->query_result_time.insert_or_assign(
      //       res->query_id, std::chrono::steady_clock::now());
      //   parent->num_results_received.fetch_add(1);
      // }
      bool all_results_arrived = false;
      parent->combined_search_results.upsert(
          res->query_id,
          [res, &all_results_arrived](
              std::shared_ptr<combined_search_results_t> &combined_results) {
            combined_results->add_result(res);
            if (combined_results->num_current_results ==
                combined_results->num_expected_results) {
              all_results_arrived = true;
            }
          },
          std::shared_ptr<combined_search_results_t>(
              new combined_search_results_t({res},
                                            parent->other_peer_ids.size())));
      if (all_results_arrived) {
        std::shared_ptr<search_result_t> combined =
            parent->combined_search_results.find(res->query_id)
                ->merge_results();
        parent->results.insert_or_assign(res->query_id, combined);
      }
    } else if (parent->dist_search_mode == DistributedSearchMode::SINGLE_SERVER) {
      // LOG(INFO) << "result received " << res->query_id;
      parent->results.insert_or_assign(res->query_id, res);
      // for (auto i = 0; i < res->num_res; i++) {
      // LOG(INFO) << res->node_id[i];
      // }
      // LOG(INFO) << res->num_res;
      parent->query_result_time.insert_or_assign(
          res->query_id, std::chrono::steady_clock::now());
      parent->num_results_received.fetch_add(1);
      parent->send_acks(res);
    } else if (parent->dist_search_mode == DistributedSearchMode::STATE_SEND) {
      // need to way to record all the states that had been sent by all servers
      // across all steps and also need to sort it at the end
      bool all_results_arrived = false;
      parent->combined_search_results.upsert(
          res->query_id,
          [res, &all_results_arrived](
              std::shared_ptr<combined_search_results_t> &combined_results) {
            combined_results->add_result(res);
            if (res->is_final == true) {
              combined_results->num_expected_results = res->partition_history.size();
            }
            if (combined_results->num_current_results ==
                combined_results->num_expected_results) {
              all_results_arrived = true;
            }
          },
          std::shared_ptr<combined_search_results_t>(
              new combined_search_results_t(
					    {res}, std::numeric_limits<uint32_t>::max())));

      if (all_results_arrived) {
        std::shared_ptr<search_result_t> combined =
            parent->combined_search_results.find(res->query_id)
                ->merge_results();
        parent->results.insert_or_assign(res->query_id, combined);
        parent->query_result_time.insert_or_assign(
						   res->query_id, std::chrono::steady_clock::now());
        parent->num_results_received.fetch_add(1);
        parent->send_acks(combined);
      }
    } else {
      throw std::invalid_argument("Weird DistributedSearchMode value");
    }
  }
}

template class StateSendClient<uint8_t>;
template class StateSendClient<int8_t>;
template class StateSendClient<float>;
