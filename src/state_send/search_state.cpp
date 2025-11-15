#include "ssd_partition_index.h"
#include "types.h"
#include <cstdint>
#include <stdexcept>

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::state_compute_dists(
    SearchState<T, TagT> *state, const unsigned *ids, const uint64_t n_ids,
    float *dists_out) {
  ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks,
                     state->pq_coord_scratch);
  ::pq_dist_lookup(state->pq_coord_scratch, n_ids, this->n_chunks,
                   state->query_emb->pq_dists, dists_out);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::state_print(SearchState<T, TagT> *state) {
  LOG(INFO) << " query id " << state->query_id << " k search "
            << state->k_search << " l search " << state->l_search
            << " beam width " << state->beam_width << " Full retset size "
            << state->full_retset.size()
            << " retset size: " << state->cur_list_size
            << " visited size: " << state->visited.size()
            << " frontier size: " << state->frontier.size()
            << " frontier nhood size: " << state->frontier_nhoods.size()
            << " frontier read reqs size: " << state->frontier_read_reqs.size();
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::state_compute_and_add_to_retset(
    SearchState<T, TagT> *state, const unsigned *node_ids,
								 const uint64_t n_ids) {

  // will overflow the dist_scratch buffer
  if (unlikely(n_ids > MAX_NUM_NEIGHBORS)) {
    throw std::invalid_argument("n_ids larger than max capacity for dist_scratch");
  }
  state_compute_dists(state, node_ids, n_ids, state->dist_scratch);
  for (uint64_t i = 0; i < n_ids; ++i) {
    auto &item = state->retset[state->cur_list_size];
    item.id = node_ids[i];
    item.distance = state->dist_scratch[i];
    item.flag = true;
    state->cur_list_size++;
    // state->visited.insert(node_ids[i]);
  }
  state_update_frontier(state);
}

template <typename T, typename TagT>
bool SSDPartitionIndex<T, TagT>::state_issue_next_io_batch(
    SearchState<T, TagT> *state, void *ctx) {
  // read nhoods of frontier ids
  if (!state->frontier.empty()) {
    state->sector_idx = 0;
    for (uint64_t i = 0; i < state->frontier.size(); i++) {
      uint32_t loc = id2loc(state->frontier[i]);
      uint64_t offset = this->loc_sector_no(loc) * SECTOR_LEN;
      auto sector_buf = state->sectors + state->sector_idx * this->size_per_io;
      fnhood_t fnhood = std::make_tuple(state->frontier[i], loc, sector_buf);
      state->sector_idx++;
      state->frontier_nhoods.push_back(fnhood);
      state->frontier_read_reqs.emplace_back(IORequest(
          offset, this->size_per_io, sector_buf, 0, 0, nullptr, state));
      // LOG(INFO) << "read offset is  " << offset;
    }

    this->reader->send_io(state->frontier_read_reqs, ctx, false);
    return true;
  }
  // LOG(INFO) << "k, size of batch " << k << " " << frontier_read_reqs.size();
  // LOG(INFO) << "k, size of frontier " << k << " " << frontier.size();
  // if (frontier.empty()) {
  //   LOG(INFO) << "k, frontier[0] " << "null";
  // }
  // else {
  //   LOG(INFO) << "k, frontier[0] " << frontier[0];
  // }
  return false;
}

template <typename T, typename TagT>
SearchExecutionState SSDPartitionIndex<T, TagT>::state_explore_frontier(
    SearchState<T, TagT> *state) {
  auto nk = state->cur_list_size;
  state->stats->n_hops++;
  for (auto &frontier_nhood : state->frontier_nhoods) {
    auto [id, loc, sector_buf] = frontier_nhood;
    char *node_disk_buf = this->offset_to_loc(sector_buf, loc);
    unsigned *node_buf = this->offset_to_node_nhood(node_disk_buf);
    uint64_t nnbrs = (uint64_t)(*node_buf);
    T *node_fp_coords = this->offset_to_node_coords(node_disk_buf);

    T *node_fp_coords_copy = state->data_buf;
    memcpy(node_fp_coords_copy, node_fp_coords, this->data_dim * sizeof(T));
    float cur_expanded_dist =
        this->dist_cmp->compare(state->query_emb->query, node_fp_coords_copy,
                                (unsigned)this->aligned_dim);

    pipeann::Neighbor n(id, cur_expanded_dist, true);
    state->full_retset.push_back(n);

    unsigned *node_nbrs = (node_buf + 1);
    state->cpu_timer.reset();
    // compute node_nbrs <-> query dist in PQ space
    state_compute_dists(state, node_nbrs, nnbrs, state->dist_scratch);
    if (state->stats != nullptr) {
      state->stats->n_cmps += (double)nnbrs;
      state->stats->cpu_us += (double)state->cpu_timer.elapsed();
    }
    state->cpu_timer.reset();
    // process prefetch-ed nhood
    for (uint64_t m = 0; m < nnbrs; ++m) {
      unsigned id = node_nbrs[m];
      // if (state->visited.find(id) != state->visited.end()) {
      // continue;
      // } else {
      // state->visited.insert(id);
      bool cont_flag = false;
      for (uint32_t i = 0; i < state->cur_list_size; i++) {
        if (state->retset[i].id == id) {
          cont_flag = true;
          break;
        }
      }
      if (cont_flag) {
        continue;
      }

      state->cmps++;
      float dist = state->dist_scratch[m];
      if (state->stats != nullptr) {
        state->stats->n_cmps++;
      }
      if (dist >= state->retset[state->cur_list_size - 1].distance &&
          (state->cur_list_size == state->l_search))
        continue;
      pipeann::Neighbor nn(id, dist, true);
      // variable search_L for deleted nodes.
      // Return position in sorted list where nn inserted.

      auto r = InsertIntoPool(state->retset, state->cur_list_size, nn);

      if (state->cur_list_size < state->l_search) {
        state->cur_list_size++;
      }

      if (r < nk) {
        nk = r;
      }
      // }
    }
    if (state->stats != nullptr) {
      state->stats->cpu_us += (double)state->cpu_timer.elapsed();
    }
  }

  if (nk <= state->k) {
    state->k = nk; // k is the best position in retset updated in this round.
  } else {
    state->k++;
  }

  // updates frontier
  state_update_frontier(state);

  if (state_search_ends(state)) {
    return SearchExecutionState::FINISHED;
  }
  if (state->frontier.empty()) {
    return SearchExecutionState::FINISHED;
  }
  if (this->dist_search_mode == DistributedSearchMode::STATE_SEND) {
    if (state_is_top_cand_off_server(state)) {
      return SearchExecutionState::TOP_CAND_NODE_OFF_SERVER;
    }
  }
  return SearchExecutionState::TOP_CAND_NODE_ON_SERVER;
}

template <typename T, typename TagT>
bool SSDPartitionIndex<T, TagT>::state_search_ends(
    SearchState<T, TagT> *state) {
  return state->k >= state->cur_list_size;
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::state_update_frontier(
    SearchState<T, TagT> *state) {
  // updates frontier
  state->frontier.clear();
  state->frontier_nhoods.clear();
  state->frontier_read_reqs.clear();
  state->sector_idx = 0;

  uint32_t marker = state->k;
  uint32_t num_seen = 0;
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
}

template <typename T, typename TagT>
uint8_t SSDPartitionIndex<T, TagT>::state_top_cand_random_partition(
    SearchState<T, TagT> *state) {
  if (state->frontier.size() == 0) {
    throw std::invalid_argument("State has frontier size 0");
  }
  return this->get_random_partition_assignment(state->frontier[0]);
}

template <typename T, typename TagT>
bool SSDPartitionIndex<T, TagT>::state_is_top_cand_off_server(
							      SearchState<T, TagT> *state) {
  if (std::find(partition_assignment[state->frontier[0]].cbegin(),
                partition_assignment[state->frontier[0]].cend(),
                this->my_partition_id) !=
      partition_assignment[state->frontier[0]].cend()) {
    return false;
  }
  return true;
}

std::string neighbors_to_string(pipeann::Neighbor *neighbors,
                                uint32_t num_neighbors) {
  std::stringstream str;
  for (auto i = 0; i < num_neighbors; i++) {
    str << "[" << neighbors[i].id << "," << neighbors[i].distance << "]";
    if (i != num_neighbors - 1)
      str << ",";
  }
  return str.str();
}






template <typename T, typename TagT>
std::string state_visited_to_string(SearchState<T, TagT> *state) {
  std::vector<uint32_t> node_ids;
  for (const auto &node_id : state->visited) {
    node_ids.push_back(node_id);
    // write_data(buffer, reinterpret_cast<const char *>(&node_id),
    // sizeof(node_id), offset);
  }
  std::sort(node_ids.begin(), node_ids.end());
  std::stringstream str;
  for (const auto &node_id : node_ids) {
    str << node_id << ",";
  }
  return str.str();
}

template <typename T, typename TagT>
std::string state_partition_history_to_string(SearchState<T, TagT> *state) {
  std::stringstream str;
  for (auto p : state->partition_history) {
    str << static_cast<int>(p) << ",";
  }
  return str.str();
}

// template <typename T, typename TagT>
// std::string state_frontier_nhoods_to_string(SearchState<T, TagT> *state) {
//   std::vector<uint32_t> node_ids;
//   for (const auto &node_id : state->visited) {
//     node_ids.push_back(node_id);
//     // write_data(buffer, reinterpret_cast<const char *>(&node_id),
//     // sizeof(node_id), offset);
//   }
//   std::sort(node_ids.begin(), node_ids.end());
//   std::stringstream str;
//   for (const auto &node_id : node_ids) {
//     str << node_id << ",";
//   }
//   return str.str();
// }

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::state_print_detailed(
    SearchState<T, TagT> *state) {
  LOG(INFO) << "State for query " << state->query_id << ", hop "
            << state->stats->n_hops;
  LOG(INFO) << "frontier size " << state->frontier.size();
  if (state->frontier.size() == 1) {
    LOG(INFO) << "frontier: " << state->frontier[0];
  }
  LOG(INFO) << "full_retset size: " << state->full_retset.size();
  LOG(INFO) << "full_retset: "
            << neighbors_to_string(state->full_retset.data(),
                                   state->full_retset.size());
  LOG(INFO) << "cur_list_size: " << state->cur_list_size;
  LOG(INFO) << "retset: "
            << neighbors_to_string(state->retset, state->cur_list_size);
  LOG(INFO) << "visited size: " << state->visited.size();
  // LOG(INFO) << "visited: " << state_visited_to_string(state);
  LOG(INFO) << "cmps: " << state->cmps;
  LOG(INFO) << "k: " << state->k;
  LOG(INFO) << "mem_l: " << state->mem_l;
  // LOG(INFO) << "l_search: " << state->l_search;
  // LOG(INFO) << "k_search: " << state->k_search;
  // LOG(INFO) << "beam_width: " << state->beam_width;
  LOG(INFO) << "partition_history size: " << state->partition_history.size();
  LOG(INFO) << "partition_history: "
            << state_partition_history_to_string(state);
  // LOG(INFO) << "client_peer_id " <<state->client_peer_id;
}

// Explicit instantiations for the SSDPartitionIndex specializations used by the
// program. Put these in this .cpp because the template definitions are here.
template class SSDPartitionIndex<float, uint32_t>;
template class SSDPartitionIndex<unsigned char, uint32_t>;
template class SSDPartitionIndex<signed char, uint32_t>;
