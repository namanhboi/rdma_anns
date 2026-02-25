/** will mostly contain serialization functions for types.h */

#include "types.h"
#include "singleton_logger.h"
#include <chrono>
#include <cstdint>
#include <limits>

// void print_neighbor_vec(const std::vector<pipeann::Neighbor> &nbrs) {
// for (const auto &nbr : nbrs) {
// LOG(INFO) << nbr.id << " " << nbr.distance;
// }
// }
// template <typename T>  {

// }

size_t write_data(char *buffer, const char *data, size_t size,
                         size_t &offset) {
  std::memcpy(buffer + offset, data, size);
  offset += size;
  return size;
}

size_t QueryStats::write_serialize(char *buffer) const {
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&total_us),
             sizeof(total_us), offset);
  write_data(buffer, reinterpret_cast<const char *>(&n_4k), sizeof(n_4k),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(&n_ios), sizeof(n_ios),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(&io_us), sizeof(io_us),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(&head_us), sizeof(head_us),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(&cpu_us), sizeof(cpu_us),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(&n_cmps), sizeof(n_cmps),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(&n_hops), sizeof(n_hops),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(&n_inter_partition_hops),
             sizeof(n_inter_partition_hops), offset);
  return offset;
}

size_t QueryStats::get_serialize_size() const {
  return sizeof(total_us) + sizeof(n_4k) + sizeof(n_ios) + sizeof(io_us) +
         sizeof(head_us) + sizeof(cpu_us) + sizeof(n_cmps) + sizeof(n_hops) +
         sizeof(n_inter_partition_hops);
}

std::shared_ptr<QueryStats> QueryStats::deserialize(const char *buffer) {
  size_t offset = 0;
  std::shared_ptr<QueryStats> stats = std::make_shared<QueryStats>();

  std::memcpy(&stats->total_us, buffer + offset, sizeof(stats->total_us));
  offset += sizeof(stats->total_us);

  std::memcpy(&stats->n_4k, buffer + offset, sizeof(stats->n_4k));
  offset += sizeof(stats->n_4k);

  std::memcpy(&stats->n_ios, buffer + offset, sizeof(stats->n_ios));
  offset += sizeof(stats->n_ios);

  std::memcpy(&stats->io_us, buffer + offset, sizeof(stats->io_us));
  offset += sizeof(stats->io_us);

  std::memcpy(&stats->head_us, buffer + offset, sizeof(stats->head_us));
  offset += sizeof(stats->head_us);

  std::memcpy(&stats->cpu_us, buffer + offset, sizeof(stats->cpu_us));
  offset += sizeof(stats->cpu_us);

  std::memcpy(&stats->n_cmps, buffer + offset, sizeof(stats->n_cmps));
  offset += sizeof(stats->n_cmps);

  std::memcpy(&stats->n_hops, buffer + offset, sizeof(stats->n_hops));
  offset += sizeof(stats->n_hops);

  std::memcpy(&stats->n_inter_partition_hops, buffer + offset,
              sizeof(stats->n_inter_partition_hops));
  offset += sizeof(stats->n_inter_partition_hops);
  return stats;
}

/**
   write the serialized form of this state into the buffer.
   Data to be serialized:
   - full_retset
   - retset
   - visited nodes
   - frontier
   - cur_list_size
   - k
   - k_search
   - l_search
   - beamwidth
   - cmps
 */
template <typename T, typename TagT>
size_t SearchState<T, TagT>::write_serialize(char *buffer) const {
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&query_id),
             sizeof(query_id), offset);
  size_t num_partitions = partition_history.size();
  write_data(buffer, reinterpret_cast<const char *>(&num_partitions),
             sizeof(num_partitions), offset);
  for (const auto partition_id : partition_history) {
    write_data(buffer, reinterpret_cast<const char *>(&partition_id),
               sizeof(partition_id), offset);
  }
  size_t num_partition_history_hop_idx = partition_history_hop_idx.size();
  write_data(buffer, (char *)(&num_partition_history_hop_idx),
             sizeof(num_partition_history_hop_idx), offset);
  for (const auto node_id : partition_history_hop_idx) {
    write_data(buffer, reinterpret_cast<const char *>(&node_id),
               sizeof(node_id), offset);
  }

  size_t size_full_retset = full_retset.size();
  write_data(buffer, reinterpret_cast<const char *>(&size_full_retset),
             sizeof(size_full_retset), offset);

  for (const auto &res : full_retset) {
    write_data(buffer, reinterpret_cast<const char *>(&(res.id)),
               sizeof(res.id), offset);
    write_data(buffer, reinterpret_cast<const char *>(&(res.distance)),
               sizeof(res.distance), offset);
    write_data(buffer, reinterpret_cast<const char *>(&(res.flag)),
               sizeof(res.flag), offset);
  }
  write_data(buffer, reinterpret_cast<const char *>(&cur_list_size),
             sizeof(cur_list_size), offset);
  for (auto i = 0; i < cur_list_size; i++) {
    auto res = retset[i];
    write_data(buffer, reinterpret_cast<const char *>(&(res.id)),
               sizeof(res.id), offset);
    write_data(buffer, reinterpret_cast<const char *>(&(res.distance)),
               sizeof(res.distance), offset);
    write_data(buffer, reinterpret_cast<const char *>(&(res.flag)),
               sizeof(res.flag), offset);
  }
  // don't write the visited set
  // size_t size_visited = visited.size();
  // write_data(buffer, reinterpret_cast<const char *>(&size_visited),
  // sizeof(size_visited), offset);
  // for (const auto &node_id : visited) {
  // write_data(buffer, reinterpret_cast<const char *>(&node_id),
  // sizeof(node_id), offset);
  // }
  // size_t size_frontier = frontier.size();

  // write_data(buffer, reinterpret_cast<const char *>(&size_frontier),
  // sizeof(size_frontier), offset);
  // for (const auto &frontier_ele : frontier) {
  // write_data(buffer, reinterpret_cast<const char *>(&frontier_ele),
  // sizeof(frontier_ele), offset);
  // }

  write_data(buffer, reinterpret_cast<const char *>(&cmps), sizeof(cmps),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(&k), sizeof(k), offset);
  write_data(buffer, reinterpret_cast<const char *>(&mem_l), sizeof(mem_l),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(&l_search),
             sizeof(l_search), offset);
  write_data(buffer, reinterpret_cast<const char *>(&k_search),
             sizeof(k_search), offset);
  write_data(buffer, reinterpret_cast<const char *>(&beam_width),
             sizeof(beam_width), offset);

  bool record_stats = (stats != nullptr);

  write_data(buffer, reinterpret_cast<const char *>(&record_stats),
             sizeof(record_stats), offset);
  if (stats != nullptr) {
    offset += stats->write_serialize(buffer + offset);
  }

  write_data(buffer, reinterpret_cast<const char *>(&client_type),
             sizeof(client_type), offset);

  write_data(buffer, reinterpret_cast<const char *>(&client_peer_id),
             sizeof(client_peer_id), offset);

  return offset;
}

template <typename T, typename TagT>
size_t SearchState<T, TagT>::get_serialize_size() const {
  size_t num_bytes = 0;
  num_bytes += sizeof(full_retset.size());
  for (const auto &res : full_retset) {
    num_bytes += sizeof(res.id);
    num_bytes += sizeof(res.distance);
    num_bytes += sizeof(res.flag);
  }
  num_bytes += sizeof(cur_list_size);
  for (uint32_t i = 0; i < cur_list_size; i++) {
    num_bytes += sizeof(retset[i].id);
    num_bytes += sizeof(retset[i].distance);
    num_bytes += sizeof(retset[i].flag);
  }

  // num_bytes += sizeof(frontier.size());
  // for (const auto &frontier_ele : frontier) {
  // num_bytes += sizeof(frontier_ele);
  // }

  num_bytes += sizeof(cmps);
  num_bytes += sizeof(k);
  num_bytes += sizeof(mem_l);
  num_bytes += sizeof(l_search);
  num_bytes += sizeof(k_search);
  num_bytes += sizeof(beam_width);

  num_bytes += sizeof(query_id);

  num_bytes += sizeof(size_t);
  num_bytes += sizeof(uint8_t) * partition_history.size();
  num_bytes += sizeof(size_t);
  num_bytes += sizeof(uint32_t) * partition_history_hop_idx.size();

  num_bytes += sizeof(bool);
  if (stats != nullptr) {
    num_bytes += stats->get_serialize_size();
  }
  num_bytes += sizeof(client_type);
  num_bytes += sizeof(client_peer_id);
  return num_bytes;
}

template <typename T, typename TagT>
void SearchState<T, TagT>::deserialize(const char *buffer, SearchState *state) {
  uint64_t query_id;
  size_t offset = 0;
  std::memcpy(&query_id, buffer + offset, sizeof(query_id));
  offset += sizeof(query_id);

  // --- partition history ---
  size_t size_partition_history;
  std::memcpy(&size_partition_history, buffer + offset,
              sizeof(size_partition_history));
  offset += sizeof(size_partition_history);
  uint64_t log_msg_id =
      query_id << 32 | static_cast<uint32_t>(size_partition_history);
  // SingletonLogger::get_logger().info("[{}] [{}]
  // [{}]:BEGIN_DESERIALIZE_STATE", SingletonLogger::get_timestamp_ns(),
  // log_msg_id, "STATE");
  // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_ALLOCATE_STATE",
  // SingletonLogger::get_timestamp_ns(),
  // log_msg_id, "STATE");
  // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_ALLOCATE_STATE",
  // SingletonLogger::get_timestamp_ns(),
  // log_msg_id, "STATE");
  state->query_id = query_id;
  state->full_retset.reserve(1024);

  const uint8_t *start_partition_history =
      reinterpret_cast<const uint8_t *>(buffer + offset);
  state->partition_history =
      std::vector<uint8_t>(start_partition_history,
                           start_partition_history + size_partition_history);

  offset += size_partition_history * sizeof(uint8_t);

  size_t num_partition_history_hop_idx;
  std::memcpy(&num_partition_history_hop_idx, buffer + offset,
              sizeof(num_partition_history_hop_idx));
  offset += sizeof(num_partition_history_hop_idx);
  state->partition_history_hop_idx.resize(num_partition_history_hop_idx);
  std::memcpy(state->partition_history_hop_idx.data(), buffer + offset,
              sizeof(uint32_t) * num_partition_history_hop_idx);
  offset += sizeof(uint32_t) * num_partition_history_hop_idx;

  state->query_emb = nullptr;
  // SingletonLogger::get_logger().info(
  // "[{}] [{}] [{}]:BEGIN_DESERIALIZE_FULL_RETSET",
  // SingletonLogger::get_timestamp_ns(), log_msg_id, "STATE");
  // --- full_retset ---
  size_t size_full_retset;
  std::memcpy(&size_full_retset, buffer + offset, sizeof(size_full_retset));
  offset += sizeof(size_full_retset);
  state->full_retset.resize(size_full_retset);

  for (size_t i = 0; i < size_full_retset; i++) {
    std::memcpy(&state->full_retset[i].id, buffer + offset,
                sizeof(state->full_retset[i].distance));
    offset += sizeof(state->full_retset[i].id);

    std::memcpy(&state->full_retset[i].distance, buffer + offset,
                sizeof(state->full_retset[i].distance));
    offset += sizeof(state->full_retset[i].distance);

    std::memcpy(&state->full_retset[i].flag, buffer + offset,
                sizeof(state->full_retset[i].flag));
    offset += sizeof(state->full_retset[i].flag);
  }
  // SingletonLogger::get_logger().info(
  // "[{}] [{}] [{}]:END_DESERIALIZE_FULL_RETSET",
  // SingletonLogger::get_timestamp_ns(), log_msg_id, "STATE");
  // SingletonLogger::get_logger().info("[{}] [{}]
  // [{}]:BEGIN_DESERIALIZE_RETSET", SingletonLogger::get_timestamp_ns(),
  // log_msg_id, "STATE");
  // --- retset ---
  std::memcpy(&state->cur_list_size, buffer + offset,
              sizeof(state->cur_list_size));
  offset += sizeof(state->cur_list_size);

  for (size_t i = 0; i < state->cur_list_size; i++) {
    std::memcpy(&state->retset[i].id, buffer + offset,
                sizeof(state->retset[i].id));
    offset += sizeof(state->retset[i].id);

    std::memcpy(&state->retset[i].distance, buffer + offset,
                sizeof(state->retset[i].distance));
    offset += sizeof(state->retset[i].distance);

    std::memcpy(&state->retset[i].flag, buffer + offset,
                sizeof(state->retset[i].flag));
    offset += sizeof(state->retset[i].flag);
    // state->retset[i] = {id, distance, f};
  }
  // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE_RETSET",
  // SingletonLogger::get_timestamp_ns(),
  // log_msg_id, "STATE");
  // --- misc fields ---
  std::memcpy(&state->cmps, buffer + offset, sizeof(state->cmps));
  offset += sizeof(state->cmps);

  std::memcpy(&state->k, buffer + offset, sizeof(state->k));
  offset += sizeof(state->k);

  std::memcpy(&state->mem_l, buffer + offset, sizeof(state->mem_l));
  offset += sizeof(state->mem_l);

  std::memcpy(&state->l_search, buffer + offset, sizeof(state->l_search));
  offset += sizeof(state->l_search);

  std::memcpy(&state->k_search, buffer + offset, sizeof(state->k_search));
  offset += sizeof(state->k_search);

  std::memcpy(&state->beam_width, buffer + offset, sizeof(state->beam_width));
  offset += sizeof(state->beam_width);

  bool record_stats;
  std::memcpy(&record_stats, buffer + offset, sizeof(record_stats));
  offset += sizeof(record_stats);
  if (record_stats) {
    state->stats = QueryStats::deserialize(buffer + offset);
    offset += state->stats->get_serialize_size();
  }

  // --- client type ---
  uint32_t client_type_raw;
  std::memcpy(&client_type_raw, buffer + offset, sizeof(client_type_raw));
  offset += sizeof(client_type_raw);
  state->client_type = static_cast<ClientType>(client_type_raw);

  // --- client peer id ---
  std::memcpy(&state->client_peer_id, buffer + offset,
              sizeof(state->client_peer_id));
  offset += sizeof(state->client_peer_id);
  // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE_STATE",
  // SingletonLogger::get_timestamp_ns(),
  // log_msg_id, "STATE");
}

template <typename T, typename TagT>
size_t SearchState<T, TagT>::write_serialize_states(
						    char *buffer, SearchState **states, size_t num_states) {
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&num_states),
             sizeof(num_states), offset);

  size_t num_queries = 0;
  for (size_t i = 0; i < num_states; i++) {
    if (states[i]->should_send_emb) {
      num_queries++;
    }
  }
  write_data(buffer, reinterpret_cast<const char *>(&num_queries),
             sizeof(num_queries), offset);

  for (size_t i = 0; i < num_states; i++) {
    offset += states[i]->write_serialize(buffer + offset);
  }
  for (size_t i = 0; i < num_states; i++) {
    if (states[i]->should_send_emb) {
      offset += states[i]->query_emb->write_serialize(buffer + offset);
    }
  }
  return offset;
}


template <typename T, typename TagT>
size_t SearchState<T, TagT>::get_serialize_size_states(
						       SearchState ** states, size_t num_states) {
  size_t num_bytes = sizeof(size_t);
  num_bytes += sizeof(size_t);
  for (size_t i =0; i < num_states; i++) {
    num_bytes += states[i]->get_serialize_size();
  }
  for (size_t i =0; i < num_states; i++) {
    if (states[i]->should_send_emb) {
      num_bytes += states[i]->query_emb->get_serialize_size();
    }
  }
  return num_bytes;
}

template <typename T, typename TagT>
void SearchState<T, TagT>::deserialize_states(const char *buffer,
                                              uint64_t num_states,
                                              uint64_t num_queries,
                                              SearchState **states,
                                              QueryEmbedding<T> **queries) {
  size_t offset = 0;
  for (size_t i = 0; i < num_states; i++) {
    SearchState::deserialize(buffer + offset, states[i]);
    offset += states[i]->get_serialize_size();
  }
  for (size_t i = 0; i < num_queries; i++) {
    QueryEmbedding<T>::deserialize(buffer + offset, queries[i]);
    offset += queries[i]->get_serialize_size();
  }
}


template <typename T, typename TagT>
size_t SearchState<T, TagT>::write_search_result(
					       DistributedSearchMode dist_search_mode, char *buffer) const {
  static uint32_t node_id[MAX_L_SEARCH * 2];
  static float distance[MAX_L_SEARCH * 2];
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&this->query_id),
             sizeof(this->query_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&this->client_peer_id),
             sizeof(this->client_peer_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&this->k_search),
             sizeof(k_search), offset);


  uint64_t num_res = 0;

  for (uint64_t i = 0; i < full_retset.size() && (dist_search_mode == DistributedSearchMode::STATE_SEND_CLIENT_GATHER? true : num_res < this->k_search);
       i++) {
    if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
      continue; // deduplicate.
    }
    // write_data(char *buffer, const char *data, size_t size, size_t &offset)
    node_id[num_res] = full_retset[i].id; // use ID to replace tags
    distance[num_res] = full_retset[i].distance;
    num_res++;
  }
  if (num_res > search_result_t::get_max_num_res()) {
    throw std::runtime_error("num res larger than max allowable");
    
  }  
  write_data(buffer, reinterpret_cast<const char *>(&num_res),
             sizeof(num_res), offset);
  write_data(buffer, reinterpret_cast<const char *>(node_id),
             sizeof(uint32_t) * num_res, offset);
  write_data(buffer, reinterpret_cast<const char *>(distance),
             sizeof(float) * num_res, offset);
  size_t num_partitions = this->partition_history.size();
  write_data(buffer, reinterpret_cast<const char *>(&num_partitions),
             sizeof(num_partitions), offset);
  write_data(buffer, reinterpret_cast<const char *>(partition_history.data()),
             sizeof(uint8_t) * num_partitions, offset);

  size_t num_partition_history_idx = this->partition_history_hop_idx.size();
  write_data(buffer, reinterpret_cast<const char *>(&num_partition_history_idx),
             sizeof(num_partition_history_idx), offset);
  write_data(buffer,
             reinterpret_cast<const char *>(partition_history_hop_idx.data()),
             sizeof(uint32_t) * num_partition_history_idx, offset);
  bool record_stats = (stats != nullptr);
  write_data(buffer, reinterpret_cast<const char *>(&record_stats),
             sizeof(record_stats), offset);
  if (record_stats) {
    offset += stats->write_serialize(buffer + offset);
  }
  bool is_final_result = k >= cur_list_size;
  
  write_data(buffer, reinterpret_cast<const char *>(&is_final_result),
             sizeof(is_final_result), offset);
  return offset;  
}




template <typename T, typename TagT>
std::shared_ptr<search_result_t> SearchState<T, TagT>::get_search_result(
    DistributedSearchMode dist_search_mode) const {
  std::shared_ptr<search_result_t> result = std::make_shared<search_result_t>();
  result->client_peer_id = this->client_peer_id;
  result->partition_history = this->partition_history;
  result->partition_history_hop_idx = this->partition_history_hop_idx;

  auto &full_retset = this->full_retset;
  result->query_id = query_id;
  // write_data(buffer, reinterpret_cast<const char *>(&this->query_id),
  // sizeof(this->query_id), offset);

  // uint64_t num_res =
  //     (dist_search_mode == DistributedSearchMode::STATE_SEND_CLIENT_GATHER)
  //         ? full_retset.size()
  //         : std::min(full_retset.size(), this->k_search);
  // if (num_res > search_result_t::get_max_num_res()) {
  //   throw std::runtime_error(
  //       "Num results trying to write is larger than max allowable");
  // }
  uint64_t num_res = 0;

  for (uint64_t i = 0; i < full_retset.size() && (dist_search_mode == DistributedSearchMode::STATE_SEND_CLIENT_GATHER? true : num_res < this->k_search);
       i++) {
    if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
      continue; // deduplicate.
    }
    // write_data(char *buffer, const char *data, size_t size, size_t &offset)
    result->node_id[num_res] = full_retset[i].id; // use ID to replace tags
    result->distance[num_res] = full_retset[i].distance;
    num_res++;
  }
  if (num_res > search_result_t::get_max_num_res()) {
    throw std::runtime_error("num res larger than max allowable");
    
  }
  result->num_res = num_res;
  result->stats = stats;
  result->k_search = this->k_search;
  return result;
}

template <typename T, typename TagT>
void SearchState<T, TagT>::reset(SearchState *state) {
  state->frontier.clear();
  state->sector_idx = 0;
  // state->visited.clear(); // does not deallocate memory.
  state->full_retset.clear();
  state->cur_list_size = state->cmps = state->k = 0;
  state->query_emb = nullptr;
  state->frontier.clear();
  state->frontier_nhoods.clear();
  state->frontier_read_reqs.clear();
  state->partition_history.clear();
  state->partition_history_hop_idx.clear();
  state->query_id = std::numeric_limits<uint64_t>::max();
  state->stats = nullptr;
  state->client_peer_id = std::numeric_limits<uint64_t>::max();
  state->io_timer.reset();
  state->query_timer.reset();
  state->cpu_timer.reset();
  // state->need_to_send_result_when_send_state = false;
  // state->sent_state = false;
  state->should_send_emb = false;
}

std::shared_ptr<search_result_t>
search_result_t::deserialize(const char *buffer) {
  std::shared_ptr<search_result_t> res = std::make_shared<search_result_t>();
  size_t offset = 0;
  std::memcpy(&res->query_id, buffer + offset, sizeof(res->query_id));
  offset += sizeof(res->query_id);

  std::memcpy(&res->client_peer_id, buffer + offset,
              sizeof(res->client_peer_id));
  offset += sizeof(res->client_peer_id);

  std::memcpy(&res->k_search, buffer + offset, sizeof(k_search));
  offset += sizeof(res->k_search);

  std::memcpy(&res->num_res, buffer + offset, sizeof(res->num_res));
  offset += sizeof(res->num_res);

  std::memcpy(res->node_id, buffer + offset, sizeof(uint32_t) * res->num_res);
  offset += sizeof(uint32_t) * res->num_res;

  std::memcpy(res->distance, buffer + offset, sizeof(float) * res->num_res);
  offset += sizeof(float) * res->num_res;

  size_t num_partitions;
  std::memcpy(&num_partitions, buffer + offset, sizeof(size_t));
  offset += sizeof(num_partitions);

  res->partition_history.resize(num_partitions);

  std::memcpy(res->partition_history.data(), buffer + offset,
              sizeof(uint8_t) * num_partitions);
  offset += sizeof(uint8_t) * num_partitions;

  size_t num_partition_history_hop_idx;
  std::memcpy(&num_partition_history_hop_idx, buffer + offset,
              sizeof(num_partition_history_hop_idx));
  offset += sizeof(num_partition_history_hop_idx);
  res->partition_history_hop_idx.resize(num_partition_history_hop_idx);
  std::memcpy(res->partition_history_hop_idx.data(), buffer + offset,
              sizeof(uint32_t) * num_partition_history_hop_idx);
  offset += sizeof(uint32_t) * num_partition_history_hop_idx;

  bool record_stats;
  std::memcpy(&record_stats, buffer + offset, sizeof(record_stats));
  offset += sizeof(record_stats);

  if (record_stats) {
    res->stats = QueryStats::deserialize(buffer + offset);
    offset += res->stats->get_serialize_size();
  }
  std::memcpy(&res->is_final_result, buffer + offset,
              sizeof(res->is_final_result));
  offset += sizeof(res->is_final_result);
  return res;
}

size_t search_result_t::write_serialize(char *buffer) const {
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&this->query_id),
             sizeof(this->query_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&this->client_peer_id),
             sizeof(this->client_peer_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&this->k_search),
             sizeof(k_search), offset);
  write_data(buffer, reinterpret_cast<const char *>(&this->num_res),
             sizeof(this->num_res), offset);
  write_data(buffer, reinterpret_cast<const char *>(this->node_id),
             sizeof(uint32_t) * num_res, offset);
  write_data(buffer, reinterpret_cast<const char *>(this->distance),
             sizeof(float) * num_res, offset);

  size_t num_partitions = this->partition_history.size();
  write_data(buffer, reinterpret_cast<const char *>(&num_partitions),
             sizeof(num_partitions), offset);
  write_data(buffer, reinterpret_cast<const char *>(partition_history.data()),
             sizeof(uint8_t) * num_partitions, offset);

  size_t num_partition_history_idx = this->partition_history_hop_idx.size();
  write_data(buffer, reinterpret_cast<const char *>(&num_partition_history_idx),
             sizeof(num_partition_history_idx), offset);
  write_data(buffer,
             reinterpret_cast<const char *>(partition_history_hop_idx.data()),
             sizeof(uint32_t) * num_partition_history_idx, offset);

  bool record_stats = (stats != nullptr);
  write_data(buffer, reinterpret_cast<const char *>(&record_stats),
             sizeof(record_stats), offset);
  if (record_stats) {
    offset += stats->write_serialize(buffer + offset);
  }
  write_data(buffer, reinterpret_cast<const char *>(&is_final_result),
             sizeof(is_final_result), offset);
  return offset;
}

size_t search_result_t::get_serialize_size() const {
  size_t num_bytes = 0;
  num_bytes += sizeof(query_id) + sizeof(client_peer_id) + sizeof(k_search) +
               sizeof(num_res) + sizeof(uint32_t) * num_res +
               sizeof(float) * num_res + sizeof(size_t) +
               sizeof(uint8_t) * partition_history.size() + sizeof(size_t) +
               sizeof(uint32_t) * partition_history_hop_idx.size() +
               sizeof(bool) + sizeof(is_final_result);
  if (stats != nullptr) {
    num_bytes += stats->get_serialize_size();
  }
  return num_bytes;
}

size_t search_result_t::get_serialize_results_size(
    const std::vector<std::shared_ptr<search_result_t>> &results) {
  size_t num_bytes = 0;
  num_bytes += sizeof(size_t);
  for (const auto &res : results) {
    num_bytes += res->get_serialize_size();
  }
  return num_bytes;
}

size_t search_result_t::write_serialize_results(
    char *buffer,
    const std::vector<std::shared_ptr<search_result_t>> &results) {
  size_t offset = 0;
  size_t num_res = results.size();
  std::memcpy(buffer + offset, &num_res, sizeof(num_res));
  offset += sizeof(num_res);
  for (const auto &res : results) {
    offset += res->write_serialize(buffer + offset);
  }
  return offset;
}

std::vector<std::shared_ptr<search_result_t>>
search_result_t::deserialize_results(const char *buffer) {
  std::vector<std::shared_ptr<search_result_t>> results;
  size_t offset = 0;
  size_t num_search_results;
  std::memcpy(&num_search_results, buffer + offset, sizeof(num_search_results));
  offset += sizeof(num_search_results);

  results.reserve(num_search_results);

  for (auto i = 0; i < num_search_results; i++) {
    results.emplace_back(search_result_t::deserialize(buffer + offset));
    offset += results[i]->get_serialize_size();
  }
  return results;
}

void search_result_t::reset(search_result_t * res) {
  res->query_id = std::numeric_limits<uint64_t>::max();
  res->client_peer_id = std::numeric_limits<uint64_t>::max();
  res->k_search = std::numeric_limits<uint64_t>::max();
  res->num_res = 0;
  res->partition_history.clear();
  res->partition_history_hop_idx.clear();
  res->stats = nullptr;
  res->is_final_result = false;
}



template <typename T>
void QueryEmbedding<T>::deserialize(const char *buffer, QueryEmbedding *query) {
  size_t offset = 0;
  uint64_t query_id;
  std::memcpy(&query_id, buffer + offset, sizeof(query_id));
  offset += sizeof(query_id);

  // SingletonLogger::get_logger().info("[{}] [{}]
  // [{}]:BEGIN_DESERIALIZE_QUERY", SingletonLogger::get_timestamp_ns(),
  // query_id, "QUERY");
  query->query_id = query_id;
  std::memcpy(&query->client_peer_id, buffer + offset,
              sizeof(query->client_peer_id));

  offset += sizeof(query->client_peer_id);
  std::memcpy(&query->mem_l, buffer + offset, sizeof(query->mem_l));
  offset += sizeof(query->mem_l);
  std::memcpy(&query->l_search, buffer + offset, sizeof(query->l_search));
  offset += sizeof(query->l_search);
  std::memcpy(&query->k_search, buffer + offset, sizeof(query->k_search));
  offset += sizeof(query->k_search);
  std::memcpy(&query->beam_width, buffer + offset, sizeof(query->beam_width));
  offset += sizeof(query->beam_width);
  std::memcpy(&query->dim, buffer + offset, sizeof(query->dim));
  offset += sizeof(query->dim);
  std::memcpy(&query->num_chunks, buffer + offset, sizeof(query->num_chunks));
  offset += sizeof(query->num_chunks);
  std::memcpy(&query->query_norm, buffer + offset, sizeof(query->query_norm));
  offset += sizeof(query->query_norm);
  std::memcpy(&query->record_stats, buffer + offset,
              sizeof(query->record_stats));
  offset += sizeof(query->record_stats);
  std::memcpy(&query->normalized, buffer + offset, sizeof(query->normalized));
  offset += sizeof(query->normalized);

  bool has_distributed_ann_state_ptr;
  std::memcpy(&has_distributed_ann_state_ptr, buffer + offset,
              sizeof(has_distributed_ann_state_ptr));
  offset += sizeof(has_distributed_ann_state_ptr);
  if (has_distributed_ann_state_ptr) {
    std::memcpy(&query->distributed_ann_state_ptr, buffer + offset,
                sizeof(query->distributed_ann_state_ptr));
    offset += sizeof(query->distributed_ann_state_ptr);
  }
  // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_COPY_QUERY_EMB",
  // SingletonLogger::get_timestamp_ns(),
  // query->query_id, "QUERY");
  std::memcpy(&query->query, buffer + offset, sizeof(T) * query->dim);
  offset += sizeof(T) * query->dim;
  // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_COPY_QUERY_EMB",
  // SingletonLogger::get_timestamp_ns(),
  // query->query_id, "QUERY");
  // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE_QUERY",
  // SingletonLogger::get_timestamp_ns(),
  // query_id, "QUERY");
}

template <typename T>
void QueryEmbedding<T>::deserialize_queries(const char *buffer,
                                            uint64_t num_queries,
                                            QueryEmbedding **queries) {
  size_t offset = 0;
  for (size_t i = 0; i < num_queries; i++) {
    QueryEmbedding<T>::deserialize(buffer + offset, queries[i]);
    offset += queries[i]->get_serialize_size();
  }
}

template <typename T>
size_t QueryEmbedding<T>::write_serialize(char *buffer) const {
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&query_id),
             sizeof(query_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&client_peer_id),
             sizeof(client_peer_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&mem_l), sizeof(mem_l),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(&l_search),
             sizeof(l_search), offset);
  write_data(buffer, reinterpret_cast<const char *>(&k_search),
             sizeof(k_search), offset);
  write_data(buffer, reinterpret_cast<const char *>(&beam_width),
             sizeof(beam_width), offset);
  write_data(buffer, reinterpret_cast<const char *>(&dim), sizeof(dim), offset);
  write_data(buffer, reinterpret_cast<const char *>(&num_chunks),
             sizeof(num_chunks), offset);
  write_data(buffer, reinterpret_cast<const char *>(&query_norm),
             sizeof(query_norm), offset);
  write_data(buffer, reinterpret_cast<const char *>(&record_stats),
             sizeof(record_stats), offset);
  write_data(buffer, reinterpret_cast<const char *>(&normalized),
             sizeof(normalized), offset);

  bool has_distributed_ann_state_ptr = (distributed_ann_state_ptr != nullptr);
  write_data(buffer, (char *)&has_distributed_ann_state_ptr,
             sizeof(has_distributed_ann_state_ptr), offset);
  if (has_distributed_ann_state_ptr) {
    write_data(buffer, (char *)&distributed_ann_state_ptr,
               sizeof(distributed_ann_state_ptr), offset);
  }

  write_data(buffer, reinterpret_cast<const char *>(query), sizeof(T) * dim,
             offset);
  return offset;
}

template <typename T> size_t QueryEmbedding<T>::get_serialize_size() const {
  size_t num_bytes = sizeof(query_id) + sizeof(client_peer_id) + sizeof(mem_l) +
                     sizeof(l_search) + sizeof(k_search) + sizeof(beam_width) +
                     sizeof(dim) + sizeof(num_chunks) + sizeof(query_norm) +
                     sizeof(record_stats) + sizeof(normalized) +
                     sizeof(T) * dim;

  num_bytes += sizeof(bool);
  if (distributed_ann_state_ptr != nullptr) {
    num_bytes += sizeof(distributed_ann_state_ptr);
  }
  return num_bytes;
}

template <typename T>
size_t QueryEmbedding<T>::write_serialize_queries(
    char *buffer, const std::vector<std::shared_ptr<QueryEmbedding>> &queries) {
  size_t offset = 0;
  size_t num_queries = queries.size();
  write_data(buffer, reinterpret_cast<const char *>(&num_queries),
             sizeof(num_queries), offset);
  for (const auto &query : queries) {
    offset += query->write_serialize(buffer + offset);
  }
  return offset;
}

template <typename T>
size_t QueryEmbedding<T>::get_serialize_size_queries(
    const std::vector<std::shared_ptr<QueryEmbedding>> &queries) {
  size_t num_bytes = 0;
  num_bytes += sizeof(size_t);
  for (const auto &query : queries) {
    num_bytes += query->get_serialize_size();
  }
  return num_bytes;
}

template <typename T> void QueryEmbedding<T>::reset(QueryEmbedding<T> *query) {
  query->dim = 0;
  query->record_stats = 0;
  query->populated_pq_dists = false;
  query->query_id = std::numeric_limits<uint64_t>::max();
  query->distributed_ann_state_ptr = nullptr;
  query->query_norm = 0.0;
  query->normalized = false;
}

size_t ack::write_serialize(char *buffer) const {
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&query_id),
             sizeof(query_id), offset);
  return offset;
}

size_t ack::get_serialize_size() const { return sizeof(query_id); }

ack ack::deserialize(const char *buffer) {
  ack a;
  std::memcpy(&a.query_id, buffer, sizeof(a.query_id));
  return a;
}

namespace distributedann {

template <typename T>
size_t scoring_query_t<T>::write_serialize(char *buffer) const {
  size_t offset = 0;
  write_data(buffer, (char *)&query_id, sizeof(query_id), offset);
  write_data(buffer, (char *)&num_node_ids, sizeof(num_node_ids), offset);
  write_data(buffer, (char *)node_ids, sizeof(uint32_t) * num_node_ids, offset);
  write_data(buffer, (char *)&threshold, sizeof(threshold), offset);
  write_data(buffer, (char *)&L, sizeof(L), offset);
  write_data(buffer, (char *)&record_stats, sizeof(record_stats), offset);
  write_data(buffer, (char *)&distributed_ann_state_ptr,
             sizeof(distributed_ann_state_ptr), offset);
  write_data(buffer, (char *)&client_peer_id, sizeof(client_peer_id), offset);
  return offset;
}

template <typename T> size_t scoring_query_t<T>::get_serialize_size() const {
  return sizeof(query_id) + sizeof(num_node_ids) +
         sizeof(uint32_t) * num_node_ids + sizeof(threshold) + sizeof(L) +
         sizeof(record_stats) + sizeof(distributed_ann_state_ptr) +
         sizeof(client_peer_id);
}

template <typename T>
size_t scoring_query_t<T>::write_serialize_scoring_queries(
    char *buffer,
    const std::vector<std::pair<scoring_query_t<T> *, bool>> &queries) {
  size_t offset = 0;
  size_t num_queries = queries.size();
  write_data(buffer, (char *)&num_queries, sizeof(num_queries), offset);
  size_t num_query_emb = 0;
  for (const auto &[query, should_send_emb] : queries) {
    if (should_send_emb)
      num_query_emb++;
  }
  write_data(buffer, (char *)&num_query_emb, sizeof(num_query_emb), offset);
  for (const auto &[query, with_embedding] : queries) {
    offset += query->write_serialize(buffer + offset);
  }
  for (const auto &[query, with_embedding] : queries) {
    if (with_embedding) {
      offset += query->query_emb->write_serialize(buffer + offset);
    }
  }

  return offset;
}

template <typename T>
size_t scoring_query_t<T>::get_serialize_size_scoring_queries(
    const std::vector<std::pair<scoring_query_t<T> *, bool>> &queries) {
  size_t num_bytes = sizeof(size_t) * 2;
  for (const auto &[query, with_embedding] : queries) {
    num_bytes += query->get_serialize_size();
    if (with_embedding) {
      num_bytes += query->query_emb->get_serialize_size();
    }
  }
  return num_bytes;
}

template <typename T>
void scoring_query_t<T>::deserialize(const char *buffer,
                                     scoring_query_t<T> *query) {
  size_t offset = 0;
  std::memcpy(&query->query_id, buffer + offset, sizeof(query->query_id));
  offset += sizeof(query->query_id);
  std::memcpy(&query->num_node_ids, buffer + offset,
              sizeof(query->num_node_ids));
  offset += sizeof(query->num_node_ids);

  std::memcpy(query->node_ids, buffer + offset,
              sizeof(uint32_t) * query->num_node_ids);
  offset += sizeof(uint32_t) * query->num_node_ids;
  std::memcpy(&query->threshold, buffer + offset, sizeof(query->threshold));
  offset += sizeof(query->threshold);
  std::memcpy(&query->L, buffer + offset, sizeof(query->L));
  offset += sizeof(query->L);
  std::memcpy(&query->record_stats, buffer + offset,
              sizeof(query->record_stats));
  offset += sizeof(query->record_stats);
  std::memcpy(&query->distributed_ann_state_ptr, buffer + offset,
              sizeof(query->distributed_ann_state_ptr));
  offset += sizeof(query->distributed_ann_state_ptr);
  std::memcpy(&query->client_peer_id, buffer + offset,
              sizeof(query->client_peer_id));
  offset += sizeof(query->client_peer_id);
}

template <typename T>
void scoring_query_t<T>::deserialize_scoring_queries(
    const char *buffer, uint64_t num_scoring_queries, uint64_t num_query_embs,
    scoring_query_t **scoring_queries, QueryEmbedding<T> **query_embs) {
  size_t offset = 0;
  for (size_t i = 0; i < num_scoring_queries; i++) {
    scoring_query_t<T>::deserialize(buffer + offset, scoring_queries[i]);
    offset += scoring_queries[i]->get_serialize_size();
  }
  for (size_t i = 0; i < num_query_embs; i++) {
    QueryEmbedding<T>::deserialize(buffer + offset, query_embs[i]);
    offset += query_embs[i]->get_serialize_size();
  }
}

template <typename T>
void scoring_query_t<T>::reset(scoring_query_t<T> *query) {
  query->num_node_ids = 0;
  query->query_id = std::numeric_limits<uint64_t>::max();
  query->threshold = 0;
  query->L = 0;
  query->query_emb = nullptr;
  query->record_stats = false;
  query->distributed_ann_state_ptr = nullptr;
  query->client_peer_id = std::numeric_limits<uint64_t>::max();
}

template <typename T> size_t result_t<T>::write_serialize(char *buffer) const {
  size_t offset = 0;
  write_data(buffer, (char *)&query_id, sizeof(query_id), offset);
  write_data(buffer, (char *)&num_full_nbrs, sizeof(num_full_nbrs), offset);
  write_data(buffer, (char *)sorted_full_nbrs,
             (sizeof(float) + sizeof(uint32_t)) * num_full_nbrs, offset);
  write_data(buffer, (char *)&num_pq_nbrs, sizeof(num_pq_nbrs), offset);
  write_data(buffer, (char *)sorted_pq_nbrs,
             num_pq_nbrs * (sizeof(uint32_t) + sizeof(float)), offset);

  bool write_stats = (stats != nullptr);
  write_data(buffer, (char *)&write_stats, sizeof(write_stats), offset);
  if (write_stats) {
    offset += stats->write_serialize(buffer + offset);
  }
  write_data(buffer, (char *)&distributed_ann_state_ptr,
             sizeof(distributed_ann_state_ptr), offset);
  write_data(buffer, (char *)&client_peer_id, sizeof(client_peer_id), offset);
  return offset;
}

template <typename T> size_t result_t<T>::get_serialize_size() const {
  size_t num_bytes = sizeof(query_id) + sizeof(num_full_nbrs) +
                     (num_full_nbrs * (sizeof(float) + sizeof(uint32_t))) +
                     sizeof(num_pq_nbrs) +
                     (num_pq_nbrs * (sizeof(uint32_t) + sizeof(float)));

  bool has_stats = (stats != nullptr);
  num_bytes += sizeof(has_stats);
  if (has_stats) {
    num_bytes += stats->get_serialize_size();
  }
  num_bytes += sizeof(distributed_ann_state_ptr);
  num_bytes += sizeof(client_peer_id);
  return num_bytes;
}

template <typename T>
void result_t<T>::deserialize(const char *buffer, result_t<T> *result) {
  static_assert(sizeof(std::pair<uint32_t, float>) == 8,
                "std::pair<uint32_t, float> must be 8 bytes for serialization");
  size_t offset = 0;
  std::memcpy(&result->query_id, buffer + offset, sizeof(result->query_id));
  offset += sizeof(result->query_id);

  std::memcpy(&result->num_full_nbrs, buffer + offset,
              sizeof(result->num_full_nbrs));
  offset += sizeof(num_full_nbrs);
  std::memcpy(result->sorted_full_nbrs, buffer + offset,
              result->num_full_nbrs * (sizeof(float) + sizeof(uint32_t)));
  offset += result->num_full_nbrs * (sizeof(float) + sizeof(uint32_t));

  std::memcpy(&result->num_pq_nbrs, buffer + offset, sizeof(num_pq_nbrs));
  offset += sizeof(num_pq_nbrs);
  std::memcpy(result->sorted_pq_nbrs, buffer + offset,
              result->num_pq_nbrs * (sizeof(float) + sizeof(uint32_t)));
  offset += result->num_pq_nbrs * (sizeof(float) + sizeof(uint32_t));

  bool has_stats;
  std::memcpy(&has_stats, buffer + offset, sizeof(has_stats));
  offset += sizeof(has_stats);
  if (has_stats) {
    result->stats = QueryStats::deserialize(buffer + offset);
    offset += result->stats->get_serialize_size();
  }
  std::memcpy(&result->distributed_ann_state_ptr, buffer + offset,
              sizeof(result->distributed_ann_state_ptr));
  offset += sizeof(result->distributed_ann_state_ptr);
  std::memcpy(&result->client_peer_id, buffer + offset,
              sizeof(result->client_peer_id));
  offset += sizeof(result->client_peer_id);
}

template <typename T>
void result_t<T>::deserialize_results(const char *buffer, uint64_t num_results,
                                      result_t **results) {
  size_t offset = 0;
  for (uint64_t i = 0; i < num_results; i++) {
    result_t<T>::deserialize(buffer + offset, results[i]);
    offset += results[i]->get_serialize_size();
  }
}

template <typename T>
size_t
result_t<T>::write_serialize_results(char *buffer,
                                     const std::vector<result_t *> &results) {
  size_t offset = 0;
  size_t num_results = results.size();
  write_data(buffer, (char *)&num_results, sizeof(num_results), offset);
  for (const auto &res : results) {
    offset += res->write_serialize(buffer + offset);
  }
  return offset;
}

template <typename T>
size_t result_t<T>::get_serialize_size_results(
    const std::vector<result_t *> &results) {
  size_t num_bytes = sizeof(size_t);
  for (const auto &res : results) {
    num_bytes += res->get_serialize_size();
  }
  return num_bytes;
}

template <typename T> void result_t<T>::reset(result_t<T> *result) {
  result->query_id = std::numeric_limits<uint64_t>::max();
  result->num_full_nbrs = 0;
  result->num_pq_nbrs = 0;
  result->stats = nullptr;
  result->distributed_ann_state_ptr = nullptr;
  result->client_peer_id = std::numeric_limits<uint64_t>::max();
}

template <typename T, typename TagT>
void DistributedANNState<T, TagT>::reset(DistributedANNState<T, TagT> *state) {
  SearchState<T, TagT>::reset(state);
  // state->result_queue = moodycamel::BlockingConcurrentQueue<result_t<T>>();
}

}; // namespace distributedann

template struct QueryEmbedding<float>;
template struct QueryEmbedding<uint8_t>;
template struct QueryEmbedding<int8_t>;

template struct SearchState<float>;
template struct SearchState<uint8_t>;
template struct SearchState<int8_t>;

template class PreallocatedQueue<QueryEmbedding<float>>;
template class PreallocatedQueue<QueryEmbedding<uint8_t>>;
template class PreallocatedQueue<QueryEmbedding<int8_t>>;

template class PreallocatedQueue<SearchState<float>>;
template class PreallocatedQueue<SearchState<uint8_t>>;
template class PreallocatedQueue<SearchState<int8_t>>;

namespace distributedann {
template struct scoring_query_t<float>;
template struct scoring_query_t<uint8_t>;
template struct scoring_query_t<int8_t>;

template struct result_t<float>;
template struct result_t<uint8_t>;
template struct result_t<int8_t>;

template class DistributedANNState<float>;
template class DistributedANNState<uint8_t>;
template class DistributedANNState<int8_t>;

}; // namespace distributedann
template class PreallocSingleManager<
    distributedann::DistributedANNState<float>>;
template class PreallocSingleManager<
    distributedann::DistributedANNState<uint8_t>>;
template class PreallocSingleManager<
    distributedann::DistributedANNState<int8_t>>;
