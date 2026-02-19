#include "state_send_client.h"
#include "disk_utils.h"
#include "neighbor.h"
#include "ssd_partition_index.h"
#include "types.h"
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>
#include <unordered_set>

// max num threads basically cuz one thread uses a state at a time
namespace distributedann {
constexpr uint32_t MAX_PREALLOC_STATES = 128;
constexpr uint32_t MAX_PREALLOC_RESULTS = 2048;

}; // namespace distributedann

template <typename T>
StateSendClient<T>::ClientThread::ClientThread(uint64_t id,
                                               StateSendClient<T> *parent)
    : my_thread_id(id), parent(parent) {}

template <typename T> void StateSendClient<T>::ClientThread::main_loop() {
  std::vector<std::shared_ptr<QueryEmbedding<T>>> batch_of_queries;
  constexpr int max_batch_size = 1;
  batch_of_queries.reserve(max_batch_size + 1);
  auto timeout = std::chrono::microseconds(100);
  // std::cout << "main loop started for client thread " << std::endl;
  while (running) {
    batch_of_queries.resize(max_batch_size + 1);
    size_t num_dequeued =
        parent->concurrent_query_queue.wait_dequeue_bulk_timed(
            batch_of_queries.begin(), max_batch_size, timeout);
    if (num_dequeued == 0)
      continue;

    batch_of_queries.resize(num_dequeued);
    assert(num_dequeued == batch_of_queries.size());
    // check for poison pill
    for (auto i = 0; i < num_dequeued; i++) {
      if (batch_of_queries[i] == nullptr) {
        batch_of_queries.erase(batch_of_queries.begin() + i);
        break;
      }
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
      // for (const auto &query : batch_of_queries) {
      // parent->query_send_time.insert_or_assign(
      // query->query_id, std::chrono::steady_clock::now());
      // }
      parent->communicator->send_to_peer(parent->other_peer_ids[server_peer_id],
                                         r);
    } else if (parent->dist_search_mode ==
               DistributedSearchMode::SCATTER_GATHER) {

      // for (const auto &query : batch_of_queries) {
      // parent->query_send_time.insert_or_assign(
      // query->query_id, std::chrono::steady_clock::now());
      // }
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
  concurrent_query_queue.enqueue(query);
  // LOG(INFO) << "QUERIED";
  query_send_time.insert_or_assign(query->query_id,
                                   std::chrono::steady_clock::now());
  return query_id;
}

template <typename T> void StateSendClient<T>::ClientThread::start() {
  LOG(INFO) << "Started client thread";
  running = true;
  real_thread = std::thread(&StateSendClient<T>::ClientThread::main_loop, this);
}

template <typename T> void StateSendClient<T>::ClientThread::signal_stop() {
  running = false;
  // concurrent_query_queue.enqueue(nullptr);
}

template <typename T> void StateSendClient<T>::ClientThread::join() {
  if (real_thread.joinable())
    real_thread.join();
}

template <typename T>
StateSendClient<T>::StateSendClient(
    const uint64_t id, const std::vector<std::string> &address_list,
    int num_worker_threads, DistributedSearchMode dist_search_mode,
    uint64_t dim, const std::string &partition_assignment_file)
    : my_id(id), dim(dim), dist_search_mode(dist_search_mode) {
  communicator = std::make_unique<ZMQP2PCommunicator>(my_id, address_list);
  // std::cout << "Done with constructor for statesendclient" << std::endl;
  other_peer_ids = communicator->get_other_peer_ids();
  prealloc_states = PreallocatedQueue<distributedann::DistributedANNState<T>>(
      distributedann::MAX_PREALLOC_STATES,
      distributedann::DistributedANNState<T>::reset);
  prealloc_result = PreallocatedQueue<distributedann::result_t<T>>(
      distributedann::MAX_PREALLOC_RESULTS, distributedann::result_t<T>::reset);
  if (dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
    if (!file_exists(partition_assignment_file)) {
      throw std::invalid_argument("partition_assignment_file doesn't exist");
    }
    num_orchestration_threads = num_worker_threads;
    communicator->register_receive_handler(
        [this](const char *buffer, size_t size) {
          this->distributed_ann_receive_handler(buffer, size);
        });
    for (uint64_t i = 0; i < num_orchestration_threads; i++) {
      orchestration_threads.emplace_back(
          std::make_unique<OrchestrationThread>(this));
    }
    load_partition_assignment_file(partition_assignment_file,
                                   partition_assignment, num_partitions);
    LOG(INFO) << "Done loading partition assignment file";
  } else {
    num_client_threads = num_worker_threads;
    communicator->register_receive_handler(
        [this](const char *buffer, size_t size) {
          this->receive_result_handler(buffer, size);
        });
    for (uint64_t i = 0; i < num_client_threads; i++) {
      client_threads.emplace_back(std::make_unique<ClientThread>(i, this));
    }
  }

  result_thread = std::make_unique<ResultReceiveThread>(this);
}

template <typename T> void StateSendClient<T>::start() {
  communicator->start_recv_thread();
  if (dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
    for (auto &orchestration_thread : orchestration_threads) {
      orchestration_thread->start();
      LOG(INFO) << "STARTED ORCHESTRATION THREAD";
    }
  } else {
    for (auto &client_thread : client_threads) {
      client_thread->start();
    }
  }
  result_thread->start();
}

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

std::shared_ptr<search_result_t>
combine_results_client_gather(const client_gather_results_t &all_results) {
  // throw std::runtime_error("Needs to impl");
  std::shared_ptr<search_result_t> combined_res =
      std::make_shared<search_result_t>();
  combined_res->query_id = all_results.results[0]->query_id;
  combined_res->k_search = all_results.results[0]->k_search;

  std::vector<std::pair<uint32_t, float>> node_id_dist;
  for (const auto &result : all_results.results) {
    for (auto i = 0; i < result->num_res; i++) {
      node_id_dist.emplace_back(result->node_id[i], result->distance[i]);
    }
  }
  std::sort(node_id_dist.begin(), node_id_dist.end(),
            [](auto &left, auto &right) { return left.second < right.second; });
  combined_res->num_res = std::min(combined_res->k_search, node_id_dist.size());
  for (auto i = 0; i < combined_res->num_res; i++) {
    combined_res->node_id[i] = node_id_dist[i].first;
    combined_res->distance[i] = node_id_dist[i].second;
  }
  // need to combine stats as well, for now lets just do the mean of each
  // category
  combined_res->stats = std::make_shared<QueryStats>();
  size_t count = 0;

  for (const auto &res : all_results.results) {
    if (res->stats != nullptr) {
      count++;
      combined_res->stats->total_us += res->stats->total_us;
      combined_res->stats->n_4k += res->stats->n_4k;
      combined_res->stats->n_ios += res->stats->n_ios;
      combined_res->stats->io_us += res->stats->io_us;
      combined_res->stats->head_us += res->stats->head_us;
      combined_res->stats->cpu_us += res->stats->cpu_us;
      combined_res->stats->n_cmps += res->stats->n_cmps;
      combined_res->stats->n_hops += res->stats->n_hops;
    }
  }
  return combined_res;
}

std::shared_ptr<search_result_t> combine_results_scatter_gather(
    const std::vector<std::pair<uint8_t, std::shared_ptr<search_result_t>>>
        &vec_res) {
  std::shared_ptr<search_result_t> combined_res =
      std::make_shared<search_result_t>();
  combined_res->query_id = vec_res[0].second->query_id;
  combined_res->k_search = vec_res[0].second->k_search;
  std::vector<std::pair<uint32_t, float>> node_id_dist;
  for (const auto &[cluster_id, res] : vec_res) {
    for (auto i = 0; i < res->num_res; i++) {
      node_id_dist.emplace_back(res->node_id[i], res->distance[i]);
      // LOG(INFO) << res ->node_id[i];
    }
  }
  std::sort(node_id_dist.begin(), node_id_dist.end(),
            [](auto &left, auto &right) { return left.second < right.second; });
  combined_res->num_res = std::min(combined_res->k_search, node_id_dist.size());
  // LOG(INF) << "num_res " << combined_res->num_res;
  for (auto i = 0; i < combined_res->num_res; i++) {
    combined_res->node_id[i] = node_id_dist[i].first;
    combined_res->distance[i] = node_id_dist[i].second;
  }
  // need to combine stats as well, for now lets just do the mean of each
  // category
  combined_res->stats = std::make_shared<QueryStats>();
  size_t count = 0;

  for (const auto &[cluster_id, res] : vec_res) {
    if (res->stats != nullptr) {
      count++;
      combined_res->stats->total_us += res->stats->total_us;
      combined_res->stats->n_4k += res->stats->n_4k;
      combined_res->stats->n_ios += res->stats->n_ios;
      combined_res->stats->io_us += res->stats->io_us;
      combined_res->stats->head_us += res->stats->head_us;
      combined_res->stats->cpu_us += res->stats->cpu_us;
      combined_res->stats->n_cmps += res->stats->n_cmps;
      combined_res->stats->n_hops += res->stats->n_hops;
    }
  }
  return combined_res;
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
  if (dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
    for (auto &orchestration_thread : orchestration_threads) {
      orchestration_thread->signal_stop();
    }
    for (auto &orchestration_thread : orchestration_threads) {
      orchestration_thread->join();
    }
  } else {
    for (auto &client_thread : client_threads) {
      client_thread->signal_stop();
    }
    for (auto &client_thread : client_threads) {
      client_thread->join();
    }
  }
  result_thread->signal_stop();
  result_thread->join();
  communicator->stop_recv_thread();
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
void StateSendClient<T>::ResultReceiveThread::process_singular_result(
    const std::shared_ptr<search_result_t> &res) {
  parent->results.insert_or_assign(res->query_id, res);
  // for (auto i = 0; i < res->num_res; i++) {
  // LOG(INFO) << res->node_id[i];
  // }
  // LOG(INFO) << res->num_res;
  parent->query_result_time.insert_or_assign(res->query_id,
                                             std::chrono::steady_clock::now());
  parent->num_results_received.fetch_add(1);
  parent->send_acks(res);
}


template <typename T>
void StateSendClient<T>::ResultReceiveThread::process_scatter_gather_result(
    const std::shared_ptr<search_result_t> &res) {
  if (res->partition_history.size() != 1) {
    throw std::runtime_error("partition history size not 1: " +
                             std::to_string(res->partition_history.size()));
  }
  // LOG(INFO) << "result received ";
  // sub_query_result_time.insert_or_assign(res->query_id,
  // std::chrono::steady_clock::now());
  parent->sub_query_result_time.upsert(
      res->query_id,
      [&res](
          std::vector<std::pair<uint8_t, std::chrono::steady_clock::time_point>>
              &vec) {
        vec.push_back(
            {res->partition_history[0], std::chrono::steady_clock::now()});

        return false;
      },
      std::vector<std::pair<uint8_t, std::chrono::steady_clock::time_point>>{
          {res->partition_history[0], std::chrono::steady_clock::now()}});
  size_t num_res = 1;
  parent->sub_query_results.upsert(
      res->query_id,
      [&res, &num_res](
          std::vector<std::pair<uint8_t, std::shared_ptr<search_result_t>>>
              &vec) {
        vec.push_back({res->partition_history[0], res});
        num_res = vec.size();
        return false;
      },
      std::vector<std::pair<uint8_t, std::shared_ptr<search_result_t>>>{
          {res->partition_history[0], res}});
  if (num_res == parent->other_peer_ids.size()) {
    std::shared_ptr<search_result_t> combined_res =
        combine_results_scatter_gather(
            parent->sub_query_results.find(res->query_id));
    parent->results.insert_or_assign(combined_res->query_id, combined_res);

    parent->query_result_time.insert_or_assign(
        res->query_id, std::chrono::steady_clock::now());
    parent->num_results_received.fetch_add(1);
  }
}

template <typename T>
void StateSendClient<T>::ResultReceiveThread::
    process_state_send_client_gather_result(
        const std::shared_ptr<search_result_t> &res) {
  throw std::runtime_error("feature not fully implemented yet");
  bool all_results_arrived = false;
  parent->client_gather_results.upsert(
      res->query_id,
      [&res, &all_results_arrived](client_gather_results_t &results) {
        results.results.push_back(res);
        if (res->is_final_result) {
          results.final_result_idx = results.results.size() - 1;
        }
        all_results_arrived = results.all_results_arrived();
      });
  if (all_results_arrived) {
    std::shared_ptr<search_result_t> combined_res =
        combine_results_client_gather(
            parent->client_gather_results.find(res->query_id));
    parent->results.insert_or_assign(combined_res->query_id, combined_res);

    parent->query_result_time.insert_or_assign(
        res->query_id, std::chrono::steady_clock::now());
    parent->num_results_received.fetch_add(1);
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
      process_scatter_gather_result(res);
    } else if (parent->dist_search_mode ==
               DistributedSearchMode::STATE_SEND_CLIENT_GATHER) {
      this->process_state_send_client_gather_result(res);
    } else if (parent->dist_search_mode == DistributedSearchMode::STATE_SEND ||
               parent->dist_search_mode ==
                   DistributedSearchMode::SINGLE_SERVER ||
               parent->dist_search_mode ==
                   DistributedSearchMode::DISTRIBUTED_ANN) {
      this->process_singular_result(res);
    } else {
      throw std::invalid_argument("Weird DistributedSearchMode value");
    }
  }
}

template <typename T>
StateSendClient<T>::OrchestrationThread::OrchestrationThread(
    StateSendClient *parent)
    : parent(parent), ctok(parent->concurrent_query_queue) {}

template <typename T> void StateSendClient<T>::OrchestrationThread::start() {
  running = true;
  real_thread =
      std::thread(&StateSendClient<T>::OrchestrationThread::main_loop, this);
}

template <typename T>
void StateSendClient<T>::OrchestrationThread::signal_stop() {
  running = false;
}

template <typename T> void StateSendClient<T>::OrchestrationThread::join() {
  if (real_thread.joinable()) {
    real_thread.join();
  }
}

/*
  need to handle case of not sending a query embedding twice as well
 */
template <typename T>
size_t StateSendClient<T>::OrchestrationThread::send_scoring_queries(
    const distributedann::DistributedANNState<T> *state,
    const std::shared_ptr<QueryEmbedding<T>> &query, float threshold,
    std::vector<uint8_t> &partitions_with_emb) {
  std::vector<std::vector<uint32_t>> node_ids_per_partition;
  node_ids_per_partition.resize(parent->num_partitions);
  std::vector<uint8_t> used_partitions;
  for (const uint32_t &node_id : state->frontier) {
    uint8_t receiver_partition =
        parent->get_random_partition_assignment(node_id);
    node_ids_per_partition[receiver_partition].push_back(node_id);
    if (std::find(used_partitions.cbegin(), used_partitions.cend(),
                  receiver_partition) == used_partitions.cend()) {
      used_partitions.push_back(receiver_partition);
    }
  }
  for (const uint8_t &partition_id : used_partitions) {
    bool should_send_emb =
        std::find(partitions_with_emb.cbegin(), partitions_with_emb.cend(),
                  partition_id) == partitions_with_emb.cend();
    distributedann::scoring_query_t<T> scoring_query;
    if (should_send_emb) {
      scoring_query.query_emb = query.get();
    }
    scoring_query.query_id = state->query_id;
    scoring_query.record_stats = true;
    scoring_query.threshold = threshold;
    scoring_query.L = state->l_search;
    scoring_query.distributed_ann_state_ptr = (void *)(state);
    scoring_query.num_node_ids = node_ids_per_partition[partition_id].size();
    scoring_query.client_peer_id = parent->my_id;
    if (scoring_query.num_node_ids >
        distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN) {
      throw std::runtime_error("");
    }
    std::memcpy(scoring_query.node_ids,
                node_ids_per_partition[partition_id].data(),
                sizeof(uint32_t) * scoring_query.num_node_ids);
    MessageType msg_type = MessageType::SCORING_QUERIES;
    size_t region_size =
        sizeof(msg_type) +
        distributedann::scoring_query_t<T>::get_serialize_size_scoring_queries(
            {{&scoring_query, should_send_emb}});

    Region r;
    r.length = region_size;
    r.addr = new char[region_size];

    size_t offset = 0;
    std::memcpy(r.addr, &msg_type, sizeof(msg_type));
    offset += sizeof(msg_type);

    distributedann::scoring_query_t<T>::write_serialize_scoring_queries(
        r.addr + offset, {{&scoring_query, should_send_emb}});
    parent->communicator->send_to_peer(partition_id, r);

    if (should_send_emb) {
      partitions_with_emb.push_back(partition_id);
    }
  }
  return used_partitions.size();
}

// template <typename T> void print_pair_vec(T *start, uint32_t num) {
// for (uint32_t i = 0; i < num; i++) {
// LOG(INFO) << (start + i)->first << " " << (start + i)->second;
// }
// }

template <typename T>
std::shared_ptr<search_result_t>
StateSendClient<T>::OrchestrationThread::search_query(
    std::shared_ptr<QueryEmbedding<T>> query) {
  const auto timeout = std::chrono::milliseconds(1);
  std::array<distributedann::result_t<T> *,
             distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN>
      result_scratch;

  std::shared_ptr<QueryStats> stats;
  if (query->record_stats) {
    stats = std::make_shared<QueryStats>();
  }

  PreallocSingleManager<distributedann::DistributedANNState<T>> manager(
      parent->prealloc_states);
  distributedann::DistributedANNState<T> *state = manager.get_item();
  // LOG(INFO) << "Done with managerx";
  state->query_id = query->query_id;
  state->l_search = query->l_search;
  state->k_search = query->k_search;
  state->beam_width = query->beam_width;
  // used to ensure that we don't send the embedding twice
  std::vector<uint8_t> partitions_with_emb;

  // first we need to send the query to the server to get the head index results
  query->distributed_ann_state_ptr = (void *)state;
  std::vector<std::shared_ptr<QueryEmbedding<T>>> query_arr = {query};
  size_t region_size = sizeof(MessageType) +
                       QueryEmbedding<T>::get_serialize_size_queries(query_arr);
  Region head_index_query_region;
  head_index_query_region.length = region_size;
  head_index_query_region.addr = new char[region_size];
  size_t offset = 0;
  MessageType msg_type = MessageType::QUERIES;
  std::memcpy(head_index_query_region.addr, &msg_type, sizeof(msg_type));
  offset += sizeof(msg_type);
  QueryEmbedding<T>::write_serialize_queries(
      head_index_query_region.addr + offset, query_arr);

  uint32_t server_peer_id =
      parent->current_round_robin_peer_index.fetch_add(1) %
      parent->other_peer_ids.size();
  partitions_with_emb.push_back(static_cast<uint8_t>(server_peer_id));

  parent->communicator->send_to_peer(server_peer_id, head_index_query_region);
  // LOG(INFO) << "SENT the head index query";

  distributedann::result_t<T> *head_index_result;
  bool has_result = false;
  while (!has_result && running) {
    has_result =
        state->result_queue.wait_dequeue_timed(head_index_result, timeout);
  }
  if (!running) {
    if (has_result) {
      parent->prealloc_result.free(head_index_result);
    }
    return nullptr;
  }

  if (head_index_result->query_id != state->query_id) {
    throw std::runtime_error("result received doesn't match query id " +
                             std::to_string(state->query_id) + " " +
                             std::to_string(head_index_result->query_id));
  }
  // after we get the result for head index search, add it to result set
  state->cur_list_size = head_index_result->num_pq_nbrs;
  for (uint32_t i = 0; i < head_index_result->num_pq_nbrs; i++) {
    state->retset[i].id = head_index_result->sorted_pq_nbrs[i].first;
    state->retset[i].distance = head_index_result->sorted_pq_nbrs[i].second;
    state->retset[i].flag = true;
  }
  // print_neighbor_vec({state->retset, state->retset + state->cur_list_size});
  parent->prealloc_result.free(head_index_result);
  auto is_in_retset = [&](uint32_t node_id) {
    for (uint32_t i = 0; i < state->cur_list_size; i++) {
      if (state->retset[i].id == node_id) {
        return true;
      }
    }
    return false;
  };

  while (state->k < state->cur_list_size && running) {
    if (stats != nullptr) {
      stats->n_hops++;
    }
    auto nk = state->cur_list_size;
    float threshold = state->retset[state->cur_list_size - 1].distance;
    state->frontier.clear();
    // state->frontier_nhoods.clear();
    // state->frontier_read_reqs.clear();
    uint32_t marker = state->k;
    uint32_t num_seen = 0;

    // put stuff to send into frontier
    while (marker < state->cur_list_size &&
           state->frontier.size() < state->beam_width &&
           num_seen < state->beam_width) {
      if (state->retset[marker].flag) {
        num_seen++;
        state->frontier.push_back(state->retset[marker].id);
        state->retset[marker].flag = false;
      }
      marker++;
    }

    // this send queries based on what's in the frontier
    size_t num_scoring_queries = this->send_scoring_queries(
        state, query, threshold, partitions_with_emb);
    // if (unlikely(num_scoring_queries >
    // distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN)) { throw
    // std::runtime_error("number of scoring queries larger than max
    // beamwidth");

    // }

    // waits for results to comeback
    size_t num_scoring_results = 0;
    while (num_scoring_results < num_scoring_queries && running) {
      size_t left = num_scoring_queries - num_scoring_results;
      size_t num_results_received = state->result_queue.wait_dequeue_bulk_timed(
          result_scratch.data() + num_scoring_results, left, timeout);
      num_scoring_results += num_results_received;
    }
    if (!running) {
      for (uint64_t i = 0; i < num_scoring_results; i++) {
        parent->prealloc_result.free(result_scratch[i]);
      }
      return nullptr;
    }
    if (num_scoring_results > num_scoring_queries) {
      throw std::runtime_error(
          "Num results recived larger than num scoring queries : " +
          std::to_string(num_scoring_results) + " " +
          std::to_string(num_scoring_queries));
    }

    // need to merge results back into retset and full retset
    for (uint64_t i = 0; i < num_scoring_results; i++) {
      auto result = result_scratch[i];
      if (stats != nullptr) {
        if (result->stats == nullptr) {
          throw std::runtime_error("scoring result can't be nullptr");
        }
        stats->n_cmps += result->stats->n_cmps;
        stats->n_ios += result->stats->n_ios;
        stats->n_4k += result->stats->n_4k;
      }
      // LOG(INFO) << "NUM FULL NBRS " << result->num_full_nbrs;
      for (auto j = 0; j < result->num_full_nbrs; j++) {
        state->full_retset.push_back({result->sorted_full_nbrs[j].first,
                                      result->sorted_full_nbrs[j].second,
                                      false});
      }
      // batch insertion version
      if (result->sorted_pq_nbrs[0].second >=
          state->retset[state->cur_list_size - 1].distance) {
        parent->prealloc_result.free(result);
        continue;
      }

      auto [lowest_insertion_idx, new_cur_list_size] = BatchInsertIntoPool(
          state->retset, state->cur_list_size, state->l_search,
          result->sorted_pq_nbrs, result->num_pq_nbrs);
      state->cur_list_size = new_cur_list_size;
      if (lowest_insertion_idx < nk) {
        nk = lowest_insertion_idx;
      }
      parent->prealloc_result.free(result);
    }

    if (nk <= state->k) {
      state->k = nk; // k is the best position in retset updated in this round.
    } else {
      state->k++;
    }
  }
  state->partition_history = partitions_with_emb;
  std::shared_ptr<search_result_t> result =
      state->get_search_result(parent->dist_search_mode);

  result->stats = stats;
  return result;
}

template <typename T>
void StateSendClient<T>::OrchestrationThread::main_loop() {
  auto timeout = std::chrono::milliseconds(100);
  std::shared_ptr<QueryEmbedding<T>> query;
  while (running) {
    bool dequeued =
        parent->concurrent_query_queue.wait_dequeue_timed(query, timeout);
    // LOG(INFO) << "YO WASSUP";
    if (!dequeued) {
      continue;
    }

    std::shared_ptr<search_result_t> result = search_query(query);
    if (result == nullptr) {
      if (running) {
        throw std::runtime_error("result is nullptr but running = true");
      }
    } else {
      parent->result_queue.enqueue(result);
    }
  }
}

/**
   result_t from the servers
 */
template <typename T>
void StateSendClient<T>::distributed_ann_receive_handler(const char *buffer,
                                                         size_t size) {
  size_t offset = 0;
  MessageType msg_type;
  std::memcpy(&msg_type, buffer + offset, sizeof(msg_type));
  offset += sizeof(msg_type);

  if (msg_type == MessageType::DISTRIBUTED_ANN_RESULTS) {
    size_t num_results;
    std::memcpy(&num_results, buffer + offset, sizeof(num_results));
    offset += sizeof(num_results);

    if (num_results > MAX_ELEMENTS_HANDLER) {
      throw std::runtime_error("num_results (" + std::to_string(num_results) +
                               ") exceeds MAX_ELEMENTS_HANDLER (" +
                               std::to_string(MAX_ELEMENTS_HANDLER) + ")");
    }
    prealloc_result.dequeue_exact(num_results, handler_result_scratch.data());
    // the result will be deallocated within the search_query function
    distributedann::result_t<T>::deserialize_results(
        buffer + offset, num_results, handler_result_scratch.data());
    for (auto i = 0; i < num_results; i++) {
      distributedann::DistributedANNState<T> *state_ptr =
          reinterpret_cast<distributedann::DistributedANNState<T> *>(
              handler_result_scratch[i]->distributed_ann_state_ptr);
      state_ptr->result_queue.enqueue(handler_result_scratch[i]);
    }
  } else {
    throw std::runtime_error("Weird message type value");
  }
}

template class StateSendClient<uint8_t>;
template class StateSendClient<int8_t>;
template class StateSendClient<float>;
