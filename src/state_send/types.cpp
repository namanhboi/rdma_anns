/** will mostly contain serialization functions for types.h */

#include "types.h"

inline size_t write_data(char *buffer, const char *data, size_t size, size_t &offset) {
  std::memcpy(buffer + offset, data, size);
  offset += size;
  return size;
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
  size_t num_bytes = 0;
  size_t size_full_retset = full_retset.size();
  num_bytes +=
      write_data(buffer, reinterpret_cast<const char *>(&size_full_retset),
                 sizeof(size_full_retset), offset);
  for (const auto &res : full_retset) {
    num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&(res.id)),
               sizeof(res.id), offset);
    num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&(res.distance)),
               sizeof(res.distance), offset);
    num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&(res.flag)),
               sizeof(res.flag), offset);
  }
  size_t size_retset = retset.size();
  num_bytes += write_data(buffer, reinterpret_cast<const char *>(&size_retset),
                          sizeof(size_retset), offset);
  for (const auto &res : retset) {
    num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&(res.id)),
               sizeof(res.id), offset);
    num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&(res.distance)),
               sizeof(res.distance), offset);
    num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&(res.flag)),
               sizeof(res.flag), offset);
  }
  size_t size_visited = visited.size();
  num_bytes += write_data(buffer, reinterpret_cast<const char *>(&size_visited),
                          sizeof(size_visited), offset);  
  for (const auto &node_id : visited) {
    num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&node_id),
               sizeof(node_id), offset);
  }
  size_t size_frontier = frontier.size();
  num_bytes +=
      write_data(buffer, reinterpret_cast<const char *>(&size_frontier),
                 sizeof(size_frontier), offset);
  for (const auto &frontier_ele : frontier) {
    num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&frontier_ele),
               sizeof(frontier_ele), offset);
  }
  num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&cur_list_size),
             sizeof(cur_list_size), offset);
  num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&cmps), sizeof(cmps),
             offset);
  num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&k), sizeof(k), offset);
  num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&l_search), sizeof(l_search),
             offset);
  num_bytes+=write_data(buffer, reinterpret_cast<const char *>(&k_search), sizeof(k_search),
             offset);
  num_bytes += write_data(buffer, reinterpret_cast<const char *>(&beam_width),
                          sizeof(beam_width), offset);

  num_bytes += write_data(buffer, reinterpret_cast<const char *>(&query_id),
                          sizeof(query_id), offset);
  num_bytes += write_data(buffer, reinterpret_cast<const char *>(&client_type),
                          sizeof(client_type), offset);

  
  num_bytes +=
      write_data(buffer, reinterpret_cast<const char *>(&client_peer_id),
                 sizeof(client_peer_id), offset);
  return num_bytes;
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
  num_bytes += sizeof(retset.size());
  for (const auto &res : retset) {
    num_bytes += sizeof(res.id);
    num_bytes += sizeof(res.distance);
    num_bytes += sizeof(res.flag);
  }
  num_bytes += sizeof(visited.size());
  for (const auto &node_id : visited) {
    num_bytes += sizeof(node_id);
  }
  num_bytes += sizeof(frontier.size());
  for (const auto &frontier_ele : frontier) {
    num_bytes += sizeof(frontier_ele);
  }
  num_bytes += sizeof(cur_list_size);
  num_bytes += sizeof(cmps);
  num_bytes += sizeof(k);
  num_bytes += sizeof(l_search);
  num_bytes += sizeof(k_search);
  num_bytes += sizeof(beam_width);

  num_bytes += sizeof(query_id);

  num_bytes += sizeof(client_type);
  num_bytes += sizeof(client_peer_id);
  return num_bytes;
}

template <typename T, typename TagT>
SearchState<T, TagT> *SearchState<T, TagT>::deserialize(const char *buffer) {
  SearchState *state = new SearchState;
  size_t offset = 0;
  size_t size_full_retset = *reinterpret_cast<const size_t *>(buffer + offset);
  offset += sizeof(size_full_retset);
  state->full_retset.reserve(size_full_retset);
  for (size_t i = 0; i < size_full_retset; i++) {
    const unsigned id = *reinterpret_cast<const unsigned *>(buffer + offset);
    offset += sizeof(id);
    const float distance = *reinterpret_cast<const float *>(buffer + offset);
    offset += sizeof(distance);
    const bool f = *reinterpret_cast<const bool *>(buffer + offset);
    offset += sizeof(f);
    state->full_retset.emplace_back(id, distance, f);
  }

  size_t size_retset = *reinterpret_cast<const size_t *>(buffer + offset);
  offset += sizeof(size_retset);
  state->retset.reserve(size_retset);
  for (size_t i = 0; i < size_retset; i++) {
    const unsigned id = *reinterpret_cast<const unsigned *>(buffer + offset);
    offset += sizeof(id);
    const float distance = *reinterpret_cast<const float *>(buffer + offset);
    offset += sizeof(distance);
    const bool f = *reinterpret_cast<const bool *>(buffer + offset);
    offset += sizeof(f);
    state->retset.emplace_back(id, distance, f);
  }

  size_t size_visited = *reinterpret_cast<const size_t *>(buffer + offset);
  offset += sizeof(size_visited);
  const uint64_t * start_visited = reinterpret_cast<const uint64_t *>(buffer + offset);
  state->visited =
    tsl::robin_set<uint64_t>(start_visited, start_visited + size_visited);
  offset += size_visited * sizeof(uint64_t);


  size_t size_frontier = *reinterpret_cast<const size_t *>(buffer + offset);
  offset += sizeof(size_frontier);
  const unsigned * start_frontier = reinterpret_cast<const unsigned *>(buffer + offset);
  state->frontier =
    std::vector<unsigned>(start_frontier, start_frontier + size_frontier);
  offset += size_frontier * sizeof(unsigned);

  state->cur_list_size = *reinterpret_cast<const unsigned *>(buffer + offset);
  offset += sizeof(state->cur_list_size);
  state->cmps = *reinterpret_cast<const unsigned *>(buffer + offset);
  offset += sizeof(state->cmps);
  state->k = *reinterpret_cast<const unsigned *>(buffer + offset);
  offset += sizeof(state->k);

  state->l_search = *reinterpret_cast<const uint64_t *>(buffer + offset);
  offset += sizeof(state->l_search);
  state->k_search = *reinterpret_cast<const uint64_t *>(buffer + offset);
  offset += sizeof(state->k_search);
  state->beam_width = *reinterpret_cast<const uint64_t *>(buffer + offset);
  offset += sizeof(state->beam_width);

  state->query_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
  offset += sizeof(state->query_id);
  
  state->client_type = static_cast<ClientType>(
					       *reinterpret_cast<const uint32_t *>(buffer + offset));
  offset += sizeof(state->client_type);

  state->client_peer_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
  offset += sizeof(state->client_peer_id);
  return state;
}

template <typename T, typename TagT>
size_t SearchState<T, TagT>::write_serialize_states(
						    char *buffer, const std::vector<SearchState *> &states) {
  size_t offset = 0;
  size_t num_states = states.size();
  write_data(buffer, reinterpret_cast<const char *>(&num_states),
             sizeof(num_states), offset);
  for (const auto &state : states) {
    offset += state->write_serialize(buffer + offset);
  }
  return offset;
}


template <typename T, typename TagT>
size_t SearchState<T, TagT>::get_serialize_size_states(
						       const std::vector<SearchState *> &states) {
  size_t num_bytes = sizeof(states.size());
  for (const auto &state : states) {
    num_bytes += state->get_serialize_size();
  }
  return num_bytes;
}


template <typename T, typename TagT>
std::vector<SearchState<T, TagT> *>
SearchState<T, TagT>::deserialize_states(const char *buffer,
                                                            size_t size) {
  size_t offset = 0;
  std::vector<SearchState *> states;

  size_t num_states = *reinterpret_cast<const size_t *>(buffer + offset);
  states.reserve(num_states);
  offset += sizeof(num_states);
  for (size_t i = 0; i < num_states; i++) {
    auto *state = SearchState::deserialize(buffer + offset);
    offset += state->get_serialize_size();
    states.push_back(state);
  }
  return states;
}




template struct SearchState<float>;
template struct SearchState<uint8_t>;
template struct SearchState<int8_t>;
