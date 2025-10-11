#include "ssd_partition_index.h"
#include <cstdint>

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchState::compute_dists(
    const unsigned *ids, const uint64_t n_ids, float *dists_out) {
  ::aggregate_coords(ids, n_ids, parent->data.data(), parent->n_chunks,
                     pq_coord_scratch);
  ::pq_dist_lookup(pq_coord_scratch, n_ids, parent->n_chunks, pq_dists,
                   dists_out);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchState::print() {
  LOG(INFO) << "Full retset size " << full_retset.size()
  << " retset size: " << retset.size()
  << " visited size: " << visited.size()
  << " frontier size: " << frontier.size()
  << " frontier nhood size: " << frontier_nhoods.size()
  << " frontier read reqs size: " << frontier_read_reqs.size();
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchState::reset() {
  data_buf_idx = 0;
  sector_idx = 0;
  visited.clear(); // does not deallocate memory.
  retset.resize(4096);
  retset.clear();
  full_retset.clear();
  cur_list_size = cmps = k = 0;
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchState::compute_and_add_to_retset(
									const unsigned *node_ids, const uint64_t n_ids) {
  compute_dists(node_ids, n_ids, dist_scratch);
  for (uint64_t i = 0; i < n_ids; ++i) {
    auto &item = retset[cur_list_size];
    item.id = node_ids[i];
    item.distance = dist_scratch[i];
    item.flag = true;
    cur_list_size++;
    visited.insert(node_ids[i]);
  }
  update_frontier_based_on_retset();  
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchState::issue_next_io_batch(void *ctx) {
  // read nhoods of frontier ids
  if (!frontier.empty()) {
    for (uint64_t i = 0; i < frontier.size(); i++) {
      uint32_t loc = parent->id2loc(frontier[i]);
      uint64_t offset = parent->loc_sector_no(loc) * SECTOR_LEN;
      auto sector_buf = sectors + sector_idx * parent->size_per_io;
      fnhood_t fnhood = std::make_tuple(loc, loc, sector_buf);
      sector_idx++;
      frontier_nhoods.push_back(fnhood);
      frontier_read_reqs.emplace_back(IORequest(
						offset, parent->size_per_io, sector_buf, 0, 0, nullptr, this));
    }
    parent->reader->send_io(frontier_read_reqs, ctx, false);
  }
  // LOG(INFO) << "k, size of batch " << k << " " << frontier_read_reqs.size();
  // LOG(INFO) << "k, size of frontier " << k << " " << frontier.size();
  // if (frontier.empty()) {
  //   LOG(INFO) << "k, frontier[0] " << "null";
  // }
  // else {
  //   LOG(INFO) << "k, frontier[0] " << frontier[0];
  // }
}

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::SearchExecutionState SSDPartitionIndex<T, TagT>::SearchState::explore_frontier() {
  auto nk = cur_list_size;

  for (auto &frontier_nhood : frontier_nhoods) {
    auto [id, loc, sector_buf] = frontier_nhood;
    char *node_disk_buf = parent->offset_to_loc(sector_buf, loc);
    unsigned *node_buf = parent->offset_to_node_nhood(node_disk_buf);
    uint64_t nnbrs = (uint64_t)(*node_buf);
    T *node_fp_coords = parent->offset_to_node_coords(node_disk_buf);

    T *node_fp_coords_copy =
      data_buf + (data_buf_idx * parent->aligned_dim);
    data_buf_idx++;
    memcpy(node_fp_coords_copy, node_fp_coords,
           parent->data_dim * sizeof(T));
    float cur_expanded_dist = parent->dist_cmp->compare(
							query, node_fp_coords_copy, (unsigned)parent->aligned_dim);

    pipeann::Neighbor n(id, cur_expanded_dist, true);
    full_retset.push_back(n);

    unsigned *node_nbrs = (node_buf + 1);
    // compute node_nbrs <-> query dist in PQ space
    compute_dists(node_nbrs, nnbrs, dist_scratch);

    // process prefetch-ed nhood
    for (uint64_t m = 0; m < nnbrs; ++m) {
      unsigned id = node_nbrs[m];
      if (visited.find(id) != visited.end()) {
        continue;
      } else {
        visited.insert(id);
        cmps++;
        float dist = dist_scratch[m];
        if (dist >= retset[cur_list_size - 1].distance &&
            (cur_list_size == l_search))
          continue;
        pipeann::Neighbor nn(id, dist, true);
        // variable search_L for deleted nodes.
        // Return position in sorted list where nn inserted.

        auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

        if (cur_list_size < l_search) {
          ++cur_list_size;
        }

        if (r < nk)
          nk = r;
      }
    }
  }

  if (nk <= k) {
    k = nk; // k is the best position in retset updated in this round.
  }else {
    ++k;
  }
  if (search_ends()) {
    return SearchExecutionState::FINISHED;
  }
  // updates frontier
  update_frontier_based_on_retset();

  if (frontier.empty()) {
    return SearchExecutionState::FINISHED;
  }
  if (parent->num_partitions > 1) {
    if (parent->get_cluster_assignment(frontier[0]) != parent->my_partition_id) {
      return SearchExecutionState::TOP_CAND_NODE_OFF_SERVER;
    }
  }

  return SearchExecutionState::TOP_CAND_NODE_ON_SERVER;
}


template <typename T, typename TagT>
bool SSDPartitionIndex<T, TagT>::SearchState::search_ends() {
  // this->print();
  return k >= cur_list_size;
}


template <typename T, typename TagT>
void SSDPartitionIndex<T,
                       TagT>::SearchState::update_frontier_based_on_retset() {
  // updates frontier
  frontier.clear();
  frontier_nhoods.clear();
  frontier_read_reqs.clear();
  sector_idx = 0;

  uint32_t marker = k;
  uint32_t num_seen = 0;
  while (marker < cur_list_size && frontier.size() < beam_width &&
         num_seen < beam_width) {
    if (retset[marker].flag) {
      num_seen++;
      frontier.push_back(retset[marker].id);
      retset[marker].flag = false;
    }
    marker++;
  }    
}



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
size_t SSDPartitionIndex<T, TagT>::SearchState::write_serialize(char *buffer) const {
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

  num_bytes += write_data(buffer, reinterpret_cast<const char *>(&client_type),
                          sizeof(client_type), offset);
  num_bytes +=
      write_data(buffer, reinterpret_cast<const char *>(&client_peer_id),
                 sizeof(client_peer_id), offset);
  return num_bytes;
}

template <typename T, typename TagT>
size_t SSDPartitionIndex<T, TagT>::SearchState::get_serialize_size() const {
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

  num_bytes += sizeof(client_type);
  num_bytes += sizeof(client_peer_id);
  return num_bytes;
}

template <typename T, typename TagT>
typename SSDPartitionIndex<T, TagT>::SearchState *
SSDPartitionIndex<T, TagT>::SearchState::deserialize(const char *buffer) {
  SSDPartitionIndex<T, TagT>::SearchState *state =
    new SSDPartitionIndex<T, TagT>::SearchState;
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

  state->client_type = static_cast<ClientType>(
					       *reinterpret_cast<const uint32_t *>(buffer + offset));
  offset += sizeof(state->client_type);

  state->client_peer_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
  offset += sizeof(state->client_peer_id);
  return state;
}

template <typename T, typename TagT>
size_t SSDPartitionIndex<T, TagT>::SearchState::write_serialize_states(
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
size_t SSDPartitionIndex<T, TagT>::SearchState::get_serialize_size_states(
									  const std::vector<SearchState *> &states) {
  size_t num_bytes = sizeof(states.size());
  for (const auto &state : states) {
    num_bytes += state->get_serialize_size();
  }
  return num_bytes;
}


template <typename T, typename TagT>
std::vector<typename SSDPartitionIndex<T, TagT>::SearchState *>
SSDPartitionIndex<T, TagT>::SearchState::deserialize_states(const char *buffer,
                                                            size_t size) {
  size_t offset = 0;
  std::vector<SSDPartitionIndex<T, TagT>::SearchState *> states;

  size_t num_states = *reinterpret_cast<const size_t *>(buffer + offset);
  states.reserve(num_states);
  offset += sizeof(num_states);
  for (size_t i = 0; i < num_states; i++) {
    auto *state =
      SSDPartitionIndex<T, TagT>::SearchState::deserialize(buffer + offset);
    offset += state->get_serialize_size();
    states.push_back(state);
  }
  return states;
}




template struct SSDPartitionIndex<float>::SearchState;
template struct SSDPartitionIndex<uint8_t>::SearchState;
template struct SSDPartitionIndex<int8_t>::SearchState;


