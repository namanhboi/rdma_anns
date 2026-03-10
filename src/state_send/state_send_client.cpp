#include "state_send_client.h"
#include "disk_utils.h"
#include "distance.h"
#include "neighbor.h"
#include "ssd_partition_index.h"
#include "types.h"
#include "utils.h"
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

  // for scatter gather top n
  std::vector<std::vector<std::shared_ptr<QueryEmbedding<T>>>> query_batches(
      parent->other_peer_ids.size());

  MessageType msg_type = MessageType::QUERIES;
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

    if (parent->dist_search_mode == DistributedSearchMode::STATE_SEND ||
        parent->dist_search_mode == DistributedSearchMode::SINGLE_SERVER ||
        parent->dist_search_mode ==
            DistributedSearchMode::STATE_SEND_CLIENT_GATHER) {
      size_t region_size =
          sizeof(MessageType::QUERIES) +
          QueryEmbedding<T>::get_serialize_size_queries(batch_of_queries);

      Region *r;
      parent->prealloc_region_queue.dequeue_exact(1, &r);

      r->length = region_size;
      size_t offset = 0;
      std::memcpy(r->addr, &msg_type, sizeof(msg_type));
      offset += sizeof(msg_type);

      QueryEmbedding<T>::write_serialize_queries(r->addr + offset,
                                                 batch_of_queries);
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
      size_t region_size =
          sizeof(MessageType::QUERIES) +
          QueryEmbedding<T>::get_serialize_size_queries(batch_of_queries);

      std::vector<Region *> r_copy(parent->num_results_to_expect);
      parent->prealloc_region_queue.dequeue_exact(parent->num_results_to_expect,
                                                  r_copy.data());

      for (uint64_t i = 0; i < parent->num_results_to_expect; i++) {
        size_t offset = 0;
        r_copy[i]->length = region_size;
        write_data(r_copy[i]->addr + offset,
                   reinterpret_cast<const char *>(&msg_type), sizeof(msg_type),
                   offset);

        QueryEmbedding<T>::write_serialize_queries(r_copy[i]->addr + offset,
                                                   batch_of_queries);
        parent->communicator->send_to_peer(parent->other_peer_ids[i],
                                           r_copy[i]);
      }
    } else if (parent->dist_search_mode ==
               DistributedSearchMode::SCATTER_GATHER_TOP_N) {
      // need to loop through medoid and caclulate distance to all parittions
      // then send query to top n nearest ones
      // need to verfiy

      // loop through each query to see where we should send the queries

      for (size_t i = 0; i < num_dequeued; i++) {
        auto query = batch_of_queries[i];
        std::vector<std::pair<float, uint8_t>> distances;
        for (size_t pid = 0; pid < parent->other_peer_ids.size(); pid++) {
          distances.emplace_back(
              parent->distance_fn->compare(parent->partition_medoids +
                                               pid * parent->dim,
                                           query->query, parent->dim),
              (uint8_t)pid);
        }
        std::partial_sort(distances.begin(),
                          distances.begin() + parent->num_results_to_expect,
                          distances.end(),
                          [](const std::pair<float, uint8_t> &a,
                             const std::pair<float, uint8_t> &b) {
                            return a.first < b.first;
                          });
        for (size_t pid = 0; pid < parent->num_results_to_expect; pid++) {
          size_t region_size = sizeof(size_t) + sizeof(MessageType::QUERIES) +
                               query->get_serialize_size();
          MessageType msg_type = MessageType::QUERIES;

          Region *r;
          parent->prealloc_region_queue.dequeue_exact(1, &r);
          r->length = region_size;
          size_t num_queries = 1;
          size_t offset = 0;
          std::memcpy(r->addr + offset, &msg_type, sizeof(msg_type));
          offset += sizeof(msg_type);
          std::memcpy(r->addr + offset, &num_queries, sizeof(num_queries));
          offset += sizeof(num_queries);
          // QueryEmbedding<T>::write_serialize_queries(r.addr + offset,
          // batch_of_queries);
          query->write_serialize(r->addr + offset);
          parent->communicator->send_to_peer(distances[pid].second, r);
        }
      }
    } else if (parent->dist_search_mode ==
               DistributedSearchMode::DISTRIBUTED_ANN) {
      // send to last other_peer_id which is the orchestration server
      size_t region_size =
          sizeof(MessageType::QUERIES) +
          QueryEmbedding<T>::get_serialize_size_queries(batch_of_queries);

      Region *r;
      parent->prealloc_region_queue.dequeue_exact(1, &r);

      r->length = region_size;
      size_t offset = 0;
      std::memcpy(r->addr, &msg_type, sizeof(msg_type));
      offset += sizeof(msg_type);

      QueryEmbedding<T>::write_serialize_queries(r->addr + offset,
                                                 batch_of_queries);
      uint32_t server_peer_id =
        parent->other_peer_ids[parent->other_peer_ids.size() - 1];
      // LOG(INFO) << server_peer_id;
      parent->communicator->send_to_peer(parent->other_peer_ids[server_peer_id],
                                         r);
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
    uint64_t dim, uint32_t top_n, const std::string &medoid_file)
    : my_id(id), dim(dim), dist_search_mode(dist_search_mode),
      prealloc_region_queue(Region::MAX_PRE_ALLOC_ELEMENTS, Region::reset),
      prealloc_result_queue(search_result_t::MAX_PRE_ALLOC_ELEMENTS * 3,
                            search_result_t::reset) {
  communicator = std::make_unique<ZMQP2PCommunicator>(my_id, address_list);
  // std::cout << "Done with constructor for statesendclient" << std::endl;
  other_peer_ids = communicator->get_other_peer_ids();
  // if (dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
  // throw std::runtime_error("yet to be implemented");
  // } else {
  if (dist_search_mode == DistributedSearchMode::SCATTER_GATHER_TOP_N) {
    num_results_to_expect = top_n;
    if (!file_exists(medoid_file)) {
      throw std::invalid_argument("medoid file doesn't exist");
    }
    size_t np;
    pipeann::load_bin<T>(medoid_file, partition_medoids, np, dim);
    num_partitions = np;
    distance_fn = pipeann::get_distance_function<T>(pipeann::L2);
    LOG(INFO) << "Loading medoids done";
  } else if (dist_search_mode == DistributedSearchMode::SCATTER_GATHER) {
    num_results_to_expect = other_peer_ids.size();
  }

  num_client_threads = num_worker_threads;
  communicator->register_receive_handler(
      [this](const char *buffer, size_t size) {
        this->receive_result_handler(buffer, size);
      });
  for (uint64_t i = 0; i < num_client_threads; i++) {
    client_threads.emplace_back(std::make_unique<ClientThread>(i, this));
  }
  // }

  prealloc_region_queue.allocate_and_assign_additional_block(
      Region::MAX_BYTES_REGION, Region::assign_addr);

  result_thread = std::make_unique<ResultReceiveThread>(this);
}

template <typename T> void StateSendClient<T>::start() {
  communicator->start_recv_thread();
  for (auto &client_thread : client_threads) {
    client_thread->start();
  }
  result_thread->start();
}

template <typename T>
void StateSendClient<T>::wait_results(const uint64_t num_results) {
  while (num_results_received != num_results) {
    // LOG(INFO) << "waiting for results, num results received "
    // << num_results_received << ","
    // << "num prealloc results "
    // << prealloc_result_queue.get_approx_num_free();
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

template <typename T>
search_result_t *StateSendClient<T>::get_result(const uint64_t query_id) {
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

void combine_results_scatter_gather(
    const std::vector<std::pair<uint8_t, search_result_t *>> &vec_res,
    search_result_t *combined_res) {
  combined_res->query_id = vec_res[0].second->query_id;
  combined_res->k_search = vec_res[0].second->k_search;
  std::vector<std::pair<uint32_t, float>> node_id_dist;
  for (const auto &[cluster_id, res] : vec_res) {
    for (auto i = 0; i < res->num_res; i++) {
      node_id_dist.emplace_back(res->node_id[i], res->distance[i]);
      // LOG(INFO) << res ->node_id[i];
    }
  }
  std::partial_sort(
      node_id_dist.begin(), node_id_dist.begin() + combined_res->k_search,
      node_id_dist.end(),
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
  std::vector<search_result_t *> results(64);

  if (msg_type == MessageType::RESULT) {
    search_result_t *res;
    prealloc_result_queue.dequeue_exact(1, &res);
    search_result_t::deserialize(buffer + offset, res);
    this->result_queue.enqueue(res);
  } else if (msg_type == MessageType::RESULTS) {
    size_t num_search_results;
    std::memcpy(&num_search_results, buffer + offset,
                sizeof(num_search_results));
    offset += sizeof(num_search_results);
    prealloc_result_queue.dequeue_exact(num_search_results, results.data());

    search_result_t::deserialize_results(buffer + offset, results.data(),
                                         num_search_results);
    this->result_queue.enqueue_bulk(results.begin(), num_search_results);
  } else {
    throw std::runtime_error("Receive unexpected type" +
                             message_type_to_string(msg_type));

  }
}

// distributedann handling hasn't been thought of yet, need to think through
template <typename T>
void StateSendClient<T>::clear_results(
    const std::vector<search_result_t *> &results_used) {
  // for scatter gather and state send gather, we need to free every sub result
  // then clear the dict

  std::vector<uint64_t> query_ids;
  for (size_t i = 0; i < results_used.size(); i++) {
    query_ids.push_back(results_used[i]->query_id);
  }
  // LOG(INFO) << "num results used" << query_ids.size();

  query_send_time.clear();
  query_result_time.clear();
  num_results_received = 0;

  // for scatter gather
  if (dist_search_mode == DistributedSearchMode::SCATTER_GATHER ||
      dist_search_mode == DistributedSearchMode::SCATTER_GATHER_TOP_N) {
    sub_query_result_time.clear();
    for (auto query_id : query_ids) {
      sub_query_results.erase_fn(
          query_id,
          [this](std::vector<std::pair<uint8_t, search_result_t *>> &results) {
            for (auto &[partition_id, re] : results) {
              this->prealloc_result_queue.free(re);
            }
            return true;
          });
    }
    sub_query_results.clear();
  }
  if (dist_search_mode == DistributedSearchMode::STATE_SEND_CLIENT_GATHER) {
    client_gather_results.clear();
    // the final combined result is inserted into results so we just need to
    // free that at the end
  }

  for (auto query_id : query_ids) {
    bool deleted = results.erase_fn(query_id, [this](search_result_t *&res) {
      this->prealloc_result_queue.free(res);
      return true;
    });
    if (!deleted) {
      throw std::runtime_error("result was not deleted");
    }
  }
  results.clear();
  // LOG(INFO) << "num prealloc results "  <<
  // prealloc_result_queue.get_approx_num_free(); region queue already handles
  // the freeing of regions. We implemented it that way because of zmq memory
  // handling req
}

template <typename T> void StateSendClient<T>::shutdown() {
  // if (dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
  // throw std::runtime_error("yet to be implemented");
  // } else {
  for (auto &client_thread : client_threads) {
    client_thread->signal_stop();
  }
  for (auto &client_thread : client_threads) {
    client_thread->join();
  }
  // }
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
void StateSendClient<T>::send_acks(const search_result_t *result) {
  std::unordered_set<uint8_t> server_peer_ids(
      result->partition_history.cbegin(), result->partition_history.cend());
  for (const auto &server_peer_id : server_peer_ids) {
    Region *r;
    prealloc_region_queue.dequeue_exact(1, &r);
    MessageType msg_type = MessageType::RESULT_ACK;
    ack a;
    a.query_id = result->query_id;
    r->length = sizeof(msg_type) + a.get_serialize_size();
    size_t offset = 0;
    std::memcpy(r->addr + offset, &msg_type, sizeof(msg_type));
    offset += sizeof(msg_type);
    a.write_serialize(r->addr + offset);
    communicator->send_to_peer(server_peer_id, r);
  }
}

template <typename T>
void StateSendClient<T>::ResultReceiveThread::process_singular_result(
    search_result_t *res) {
  parent->results.insert_or_assign(res->query_id, res);
  parent->query_result_time.insert_or_assign(res->query_id,
                                             std::chrono::steady_clock::now());
  parent->num_results_received.fetch_add(1);
  parent->send_acks(res);
}

template <typename T>
void StateSendClient<T>::ResultReceiveThread::process_scatter_gather_result(
    search_result_t *res) {
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
      [&res,
       &num_res](std::vector<std::pair<uint8_t, search_result_t *>> &vec) {
        vec.push_back({res->partition_history[0], res});
        num_res = vec.size();
        return false;
      },
      std::vector<std::pair<uint8_t, search_result_t *>>{
          {res->partition_history[0], res}});
  if (num_res == parent->num_results_to_expect) {
    search_result_t *combined_res;
    parent->prealloc_result_queue.dequeue_exact(1, &combined_res);
    combine_results_scatter_gather(
        parent->sub_query_results.find(res->query_id), combined_res);
    parent->results.insert_or_assign(combined_res->query_id, combined_res);

    parent->query_result_time.insert_or_assign(
        res->query_id, std::chrono::steady_clock::now());
    parent->num_results_received.fetch_add(1);
  }
}

void print_result(const std::shared_ptr<search_result_t> &res) {
  std::cout << "result for query " << res->query_id;
  for (size_t i = 0; i < res->num_res; i++) {
    std::cout << "(" << res->node_id[i] << "," << res->distance[i] << "),";
  }
  std::cout << std::endl;
}

template <typename T>
void StateSendClient<T>::ResultReceiveThread::
    process_state_send_client_gather_result(search_result_t *res) {
  // throw std::runtime_error("feature not fully implemented yet");
  bool all_results_arrived = false, no_entry = true;

  search_result_t *tmp_result;
  parent->prealloc_result_queue.dequeue_exact(1, &tmp_result);
  parent->client_gather_results.upsert(
      res->query_id,
      [res, &all_results_arrived, &no_entry](client_gather_result_t &results) {
        results.combine_result(res);
        all_results_arrived = results.all_results_arrived();
        no_entry = false;
        return false;
      },
      client_gather_result_t(tmp_result, res, all_results_arrived));
  if (!no_entry) {
    parent->prealloc_result_queue.free(tmp_result);
  }
  if (all_results_arrived) {
    parent->results.insert_or_assign(
        res->query_id,
        parent->client_gather_results.find(res->query_id).combined_result);
    parent->query_result_time.insert_or_assign(
        res->query_id, std::chrono::steady_clock::now());
    parent->num_results_received.fetch_add(1);
    parent->send_acks(res);
  }
}

template <typename T>
void StateSendClient<T>::ResultReceiveThread::main_loop() {
  while (running) {
    search_result_t *res;
    parent->result_queue.wait_dequeue(res);

    // for scatter gather and client gather, when is res freed?
    // scatter gather stores intermediate results into sub_query* so its freed
    // in clear_results
    // client_gather doesn't so it frees then after calling process
    // state_send_client_gather_result

    // LOG(INFO) << "hello";
    if (res == nullptr) {
      assert(!running);
      LOG(INFO) << "exit";
      break;
    }
    if (parent->dist_search_mode == DistributedSearchMode::SCATTER_GATHER ||
        parent->dist_search_mode ==
            DistributedSearchMode::SCATTER_GATHER_TOP_N) {
      process_scatter_gather_result(res);
    } else if (parent->dist_search_mode ==
               DistributedSearchMode::STATE_SEND_CLIENT_GATHER) {
      // if (res->is_final_result) LOG(INFO) << " final res received";
      this->process_state_send_client_gather_result(res);
      parent->prealloc_result_queue.free(res);
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

template class StateSendClient<uint8_t>;
template class StateSendClient<int8_t>;
template class StateSendClient<float>;
