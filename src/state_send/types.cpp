/** will mostly contain serialization functions for types.h */

#include "types.h"
#include <chrono>
#include "singleton_logger.h"



inline size_t write_data(char *buffer, const char *data, size_t size,
                         size_t &offset) {
  std::memcpy(buffer + offset, data, size);
  offset += size;
  return size;
}

size_t QueryStats::write_serialize(char *buffer) const {
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&total_us),
             sizeof(total_us), offset);
  write_data(buffer, reinterpret_cast<const char *>(&n_4k),
             sizeof(n_4k), offset);
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
  return offset;
}

size_t QueryStats::get_serialize_size() const {
  return sizeof(total_us) + sizeof(n_4k) + sizeof(n_ios) + sizeof(io_us) +
         sizeof(head_us) + sizeof(cpu_us) + sizeof(n_cmps) + sizeof(n_hops);
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
size_t SearchState<T, TagT>::write_serialize(char *buffer, bool with_embedding) const {
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

  write_data(buffer, reinterpret_cast<const char *>(&with_embedding),
             sizeof(with_embedding), offset);
  if (with_embedding) {
    offset += this->query_emb->write_serialize(buffer + offset);
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
  size_t size_frontier = frontier.size();

  write_data(buffer, reinterpret_cast<const char *>(&size_frontier),
             sizeof(size_frontier), offset);
  for (const auto &frontier_ele : frontier) {
    write_data(buffer, reinterpret_cast<const char *>(&frontier_ele),
               sizeof(frontier_ele), offset);
  }

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
size_t SearchState<T, TagT>::get_serialize_size(bool with_embedding) const {
  size_t num_bytes = 0;
  num_bytes += sizeof(with_embedding);
  if (with_embedding) {
    num_bytes += query_emb->get_serialize_size();
  }
  
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
  // num_bytes += sizeof(visited.size());
  // for (const auto &node_id : visited) {
    // num_bytes += sizeof(node_id);
  // }
  num_bytes += sizeof(frontier.size());
  for (const auto &frontier_ele : frontier) {
    num_bytes += sizeof(frontier_ele);
  }

  num_bytes += sizeof(cmps);
  num_bytes += sizeof(k);
  num_bytes += sizeof(mem_l);  
  num_bytes += sizeof(l_search);
  num_bytes += sizeof(k_search);
  num_bytes += sizeof(beam_width);

  num_bytes += sizeof(query_id);

  num_bytes += sizeof(partition_history.size());
  num_bytes += sizeof(uint8_t) * partition_history.size();

  num_bytes += sizeof(bool);
  if (stats != nullptr) {
    num_bytes += stats->get_serialize_size();
  }
  num_bytes += sizeof(client_type);
  num_bytes += sizeof(client_peer_id);
  return num_bytes;
}

template <typename T, typename TagT>
SearchState<T, TagT> *SearchState<T, TagT>::deserialize(const char *buffer) {
  uint64_t query_id;
  size_t offset = 0;
  std::memcpy(&query_id, buffer + offset, sizeof(query_id));
  offset += sizeof(query_id);
  

  // --- partition history ---
  size_t size_partition_history;
  std::memcpy(&size_partition_history, buffer + offset,
              sizeof(size_partition_history));
  offset += sizeof(size_partition_history);
  uint64_t log_msg_id = query_id << 32 | static_cast<uint32_t>(size_partition_history);
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_DESERIALIZE_STATE",
                                     SingletonLogger::get_timestamp_ns(),
                                     log_msg_id, "STATE");
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_ALLOCATE_STATE",
                                     SingletonLogger::get_timestamp_ns(),
                                     log_msg_id, "STATE");
  SearchState *state = new SearchState;
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_ALLOCATE_STATE",
                                     SingletonLogger::get_timestamp_ns(),
                                     log_msg_id, "STATE");  
  state->query_id = query_id;
  state->full_retset.reserve(1024);  


  const uint8_t *start_partition_history =
      reinterpret_cast<const uint8_t *>(buffer + offset);
  state->partition_history =
      std::vector<uint8_t>(start_partition_history,
                           start_partition_history + size_partition_history);
  offset += size_partition_history * sizeof(uint8_t);
  

  bool has_embedding;
  std::memcpy(&has_embedding, buffer + offset, sizeof(has_embedding));
  offset += sizeof(has_embedding);
  if (has_embedding) {
    SingletonLogger::get_logger().info(
        "[{}] [{}] [{}]:BEGIN_DESERIALIZE_QUERY_EMB",
				       SingletonLogger::get_timestamp_ns(), log_msg_id, "STATE");
    state->query_emb = QueryEmbedding<T>::deserialize(buffer + offset);
    offset += state->query_emb->get_serialize_size();
    SingletonLogger::get_logger().info(
        "[{}] [{}] [{}]:END_DESERIALIZE_QUERY_EMB",
				       SingletonLogger::get_timestamp_ns(), log_msg_id, "STATE");
  } else {
    state->query_emb = nullptr;
  }

  SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_DESERIALIZE_FULL_RETSET",
                                     SingletonLogger::get_timestamp_ns(),
                                     log_msg_id, "STATE");
  // --- full_retset ---
  size_t size_full_retset;
  std::memcpy(&size_full_retset, buffer + offset, sizeof(size_full_retset));
  offset += sizeof(size_full_retset);
  state->full_retset.resize(size_full_retset);

  for (size_t i = 0; i < size_full_retset; i++) {
    std::memcpy(&state->full_retset[i].id, buffer + offset,
                sizeof(state->full_retset[i].distance));
    offset += sizeof(state->full_retset[i].id);

    std::memcpy(&state->full_retset[i].distance, buffer + offset, sizeof(state->full_retset[i].distance));
    offset += sizeof(state->full_retset[i].distance);

    std::memcpy(&state->full_retset[i].flag, buffer + offset,
                sizeof(state->full_retset[i].flag));
    offset += sizeof(state->full_retset[i].flag);
  }
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE_FULL_RETSET",
                                     SingletonLogger::get_timestamp_ns(),
                                     log_msg_id, "STATE");  
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_DESERIALIZE_RETSET",
                                     SingletonLogger::get_timestamp_ns(),
                                     log_msg_id, "STATE");  
  // --- retset ---
  std::memcpy(&state->cur_list_size, buffer + offset,
              sizeof(state->cur_list_size));
  offset += sizeof(state->cur_list_size);

  for (size_t i = 0; i < state->cur_list_size; i++) {
    std::memcpy(&state->retset[i].id, buffer + offset, sizeof(state->retset[i].id));
    offset += sizeof(state->retset[i].id);

    std::memcpy(&state->retset[i].distance, buffer + offset, sizeof(state->retset[i].distance));
    offset += sizeof(state->retset[i].distance);

    std::memcpy(&state->retset[i].flag, buffer + offset, sizeof(state->retset[i].flag));
    offset += sizeof(state->retset[i].flag);
    // state->retset[i] = {id, distance, f};
  }
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE_RETSET",
                                     SingletonLogger::get_timestamp_ns(),
                                     log_msg_id, "STATE");
  // SingletonLogger::get_logger().info(
      // "[{}] [{}] [{}]:BEGIN_DESERIALIZE_VISITED_STATE",
				     // SingletonLogger::get_timestamp_ns(), log_msg_id, "STATE");

  // --- visited ---
  // size_t size_visited;
  // std::memcpy(&size_visited, buffer + offset, sizeof(size_visited));
  // offset += sizeof(size_visited);
  // SingletonLogger::get_logger().info("[{}] [{}] [{}]:VISITED_STATE {}",
                                     // SingletonLogger::get_timestamp_ns(),
                                     // log_msg_id, "STATE", size_visited);
  // Read elements safely (no reinterpret_cast)
  // state->visited.clear();
  // state->visited.reserve(1024);
  // state->visited.insert(reinterpret_cast<const uint32_t *>(buffer + offset),
                        // reinterpret_cast<const uint32_t *>(
							   // buffer + offset + sizeof(uint32_t) * size_visited));
  // for (size_t i = 0; i < size_visited; ++i) {
    // uint32_t val;
    // std::memcpy(&val, buffer + offset, sizeof(val));
    // offset += sizeof(val);
    // state->visited.insert(val);
  // }
  // offset += sizeof(uint32_t) * size_visited;

  // SingletonLogger::get_logger().info(
      // "[{}] [{}] [{}]:END_DESERIALIZE_VISITED_STATE",
				     // SingletonLogger::get_timestamp_ns(), log_msg_id, "STATE");
  // --- frontier ---
  size_t size_frontier;
  std::memcpy(&size_frontier, buffer + offset, sizeof(size_frontier));
  offset += sizeof(size_frontier);

  // Read elements safely
  state->frontier.resize(size_frontier);
  for (size_t i = 0; i < size_frontier; ++i) {
    unsigned val;
    std::memcpy(&val, buffer + offset, sizeof(val));
    offset += sizeof(val);
    state->frontier[i] = val;
  }

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
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE_STATE",
                                     SingletonLogger::get_timestamp_ns(),
                                     log_msg_id, "STATE");
  return state;
}

template <typename T, typename TagT>
size_t SearchState<T, TagT>::write_serialize_states(
						    char *buffer, const std::vector<std::pair<SearchState *, bool>> &states) {
  size_t offset = 0;
  size_t num_states = states.size();
  write_data(buffer, reinterpret_cast<const char *>(&num_states),
             sizeof(num_states), offset);
  for (const auto &[state, with_embedding] : states) {
    offset += state->write_serialize(buffer + offset, with_embedding);
  }
  return offset;
}

template <typename T, typename TagT>
std::shared_ptr<search_result_t> SearchState<T, TagT>::get_search_result() {
  std::shared_ptr<search_result_t> result = std::make_shared<search_result_t>();
  result->client_peer_id = this->client_peer_id;
  result->partition_history = this->partition_history;

  auto &full_retset = this->full_retset;
  std::sort(full_retset.begin(), full_retset.end(),
            [](const pipeann::Neighbor &left, const pipeann::Neighbor &right) {
              return left < right;
            });
  size_t offset = 0;
  result->query_id = query_id;
  // write_data(buffer, reinterpret_cast<const char *>(&this->query_id),
             // sizeof(this->query_id), offset);

  uint64_t num_res = 0;
  // static uint32_t res_id[256];
  // static uint32_t res_dist[256];

  for (uint64_t i = 0; i < full_retset.size() && num_res < this->k_search; i++) {
    if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
      continue;  // deduplicate.
    }
    // write_data(char *buffer, const char *data, size_t size, size_t &offset)
    result->node_id[num_res] = full_retset[i].id;  // use ID to replace tags
    result->distance[num_res] = full_retset[i].distance;
    num_res++;
  }
  result->num_res = num_res;
  result->stats = stats;
  return result;
}


template <typename T, typename TagT>
size_t SearchState<T, TagT>::get_serialize_size_states(
    const std::vector<std::pair<SearchState *, bool>> &states) {
  size_t num_bytes = sizeof(states.size());
  for (const auto &[state, with_embedding] : states) {
    num_bytes += state->get_serialize_size(with_embedding);
  }
  return num_bytes;
}

template <typename T, typename TagT>
std::vector<SearchState<T, TagT> *>
SearchState<T, TagT>::deserialize_states(const char *buffer, size_t size) {
  size_t offset = 0;
  std::vector<SearchState *> states;

  size_t num_states;
  std::memcpy(&num_states, buffer + offset, sizeof(num_states));
  offset += sizeof(num_states);

  states.reserve(num_states);
  for (size_t i = 0; i < num_states; i++) {
    auto *state = SearchState::deserialize(buffer + offset);
    offset += state->get_serialize_size(state->query_emb != nullptr);
    states.push_back(state);
  }
  return states;
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
  
  std::memcpy(res->partition_history.data(), buffer + offset, sizeof(uint8_t) * num_partitions);
  offset += sizeof(uint8_t) * num_partitions;

  bool record_stats;
  std::memcpy(&record_stats, buffer + offset, sizeof(record_stats));
  offset += sizeof(record_stats);

  if (record_stats) {
    res->stats = QueryStats::deserialize(buffer + offset);
    offset += res->stats->get_serialize_size();
  }
  return res;
}

size_t search_result_t::write_serialize(char *buffer) const {
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&this->query_id),
             sizeof(this->query_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&this->client_peer_id),
             sizeof(this->client_peer_id), offset);
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
  
  bool record_stats = (stats != nullptr);
  write_data(buffer, reinterpret_cast<const char *>(&record_stats),
             sizeof(record_stats), offset);
  if (record_stats) {
    offset += stats->write_serialize(buffer + offset);
  }
  return offset;
}

size_t search_result_t::get_serialize_size() const {
  size_t num_bytes = 0;
  num_bytes += sizeof(query_id) + sizeof(client_peer_id) + sizeof(num_res) +
               sizeof(uint32_t) * num_res + sizeof(float) * num_res +
               sizeof(size_t) + sizeof(uint8_t) * partition_history.size() +
               sizeof(bool);
  if (stats != nullptr) {
    num_bytes+= stats->get_serialize_size();
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
						char *buffer, const std::vector<std::shared_ptr<search_result_t>> &results) {
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


template <typename T>
std::shared_ptr<QueryEmbedding<T>>
QueryEmbedding<T>::deserialize(const char *buffer) {
  size_t offset = 0;
  uint64_t query_id;
  std::memcpy(&query_id, buffer + offset, sizeof(query_id));
  offset += sizeof(query_id);
  
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_DESERIALIZE_QUERY",
                                     SingletonLogger::get_timestamp_ns(),
                                     query_id, "QUERY");  
  
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_ALLOCATE_QUERY",
                                     SingletonLogger::get_timestamp_ns(),
                                     query_id, "QUERY");  
  std::shared_ptr<QueryEmbedding<T>> query =
    std::make_shared<QueryEmbedding<T>>();
  query->query_id = query_id;
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_ALLOCATE_QUERY",
                                     SingletonLogger::get_timestamp_ns(),
                                     query_id, "QUERY");  


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
  std::memcpy(&query->record_stats, buffer + offset, sizeof(query->record_stats));
  offset += sizeof(query->record_stats);
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_COPY_QUERY_EMB",
                                     SingletonLogger::get_timestamp_ns(),
                                     query->query_id, "QUERY");
  std::memcpy(&query->query, buffer + offset, sizeof(T) * query->dim);
  offset += sizeof(T) * query->dim;
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_COPY_QUERY_EMB",
                                     SingletonLogger::get_timestamp_ns(),
                                     query->query_id, "QUERY");
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE_QUERY",
                                     SingletonLogger::get_timestamp_ns(),
                                     query_id, "QUERY");    
  return query;
}

template <typename T>
std::vector<std::shared_ptr<QueryEmbedding<T>>>
QueryEmbedding<T>::deserialize_queries(const char *buffer, size_t size) {
  std::vector<std::shared_ptr<QueryEmbedding<T>>> queries;
  size_t offset = 0;

  size_t num_queries;
  std::memcpy(&num_queries, buffer + offset, sizeof(num_queries));
  offset += sizeof(num_queries);

  queries.reserve(num_queries);
  for (size_t i = 0; i < num_queries; i++) {
    auto query = QueryEmbedding<T>::deserialize(buffer + offset);
    offset += query->get_serialize_size();
    queries.push_back(query);
  }
  return queries;
}


template <typename T>
size_t QueryEmbedding<T>::write_serialize(char *buffer) const {
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&query_id),
             sizeof(query_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&client_peer_id),
             sizeof(client_peer_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&mem_l),
             sizeof(mem_l), offset);
  write_data(buffer, reinterpret_cast<const char *>(&l_search),
             sizeof(l_search), offset);
  write_data(buffer, reinterpret_cast<const char *>(&k_search),
             sizeof(k_search), offset);
  write_data(buffer, reinterpret_cast<const char *>(&beam_width),
             sizeof(beam_width), offset);
  write_data(buffer, reinterpret_cast<const char *>(&dim), sizeof(dim), offset);
  write_data(buffer, reinterpret_cast<const char *>(&num_chunks),
             sizeof(num_chunks), offset);
  write_data(buffer, reinterpret_cast<const char *>(&record_stats),
             sizeof(record_stats), offset);  
  write_data(buffer, reinterpret_cast<const char *>(query), sizeof(T) * dim,
             offset);
  return offset;
}

template <typename T> size_t QueryEmbedding<T>::get_serialize_size() const {
  return sizeof(query_id) + sizeof(client_peer_id) + sizeof(mem_l) +
         sizeof(l_search) + sizeof(k_search) + sizeof(beam_width) +
         sizeof(dim) + sizeof(num_chunks) + sizeof(record_stats) +
         sizeof(T) * dim;
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

size_t ack::write_serialize(char *buffer) const {
  size_t offset =0;
  write_data(buffer, reinterpret_cast<const char *>(&query_id),
             sizeof(query_id), offset);
  return offset;
}

size_t ack::get_serialize_size() const {
  return sizeof(query_id);
}

ack ack::deserialize(const char *buffer) {
  ack a;
  std::memcpy(&a.query_id, buffer, sizeof(a.query_id));
  return a;
}


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
