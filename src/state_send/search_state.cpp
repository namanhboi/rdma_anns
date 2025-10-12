#include "ssd_partition_index.h"
#include <cstdint>

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::state_compute_dists(SearchState<T, TagT> *state,
                                                     const unsigned *ids,
                                                     const uint64_t n_ids,
                                                     float *dists_out) {
  ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks,
                     state->pq_coord_scratch);
  ::pq_dist_lookup(state->pq_coord_scratch, n_ids, this->n_chunks,
                   state->pq_dists, dists_out);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::state_print(SearchState<T, TagT> *state) {
  LOG(INFO) << "Full retset size " << state->full_retset.size()
  << " retset size: " << state->retset.size()
  << " visited size: " << state->visited.size()
  << " frontier size: " << state->frontier.size()
  << " frontier nhood size: " << state->frontier_nhoods.size()
  << " frontier read reqs size: " << state->frontier_read_reqs.size();
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::state_reset(SearchState<T, TagT> *state) {
  state->data_buf_idx = 0;
  state->sector_idx = 0;
  state->visited.clear(); // does not deallocate memory.
  state->retset.resize(4096);
  state->retset.clear();
  state->full_retset.clear();
  state->cur_list_size = state->cmps = state->k = 0;
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::state_compute_and_add_to_retset(
    SearchState<T, TagT> *state, const unsigned *node_ids,
								 const uint64_t n_ids) {
  state_compute_dists(state, node_ids, n_ids, state->dist_scratch);
  for (uint64_t i = 0; i < n_ids; ++i) {
    auto &item = state->retset[state->cur_list_size];
    item.id = node_ids[i];
    item.distance = state->dist_scratch[i];
    item.flag = true;
    state->cur_list_size++;
    state->visited.insert(node_ids[i]);
  }
  state_update_frontier(state);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::state_issue_next_io_batch(SearchState<T, TagT> *state, void *ctx) {
  // read nhoods of frontier ids
  if (!state->frontier.empty()) {
    for (uint64_t i = 0; i < state->frontier.size(); i++) {
      uint32_t loc = id2loc(state->frontier[i]);
      uint64_t offset = this->loc_sector_no(loc) * SECTOR_LEN;
      auto sector_buf = state->sectors + state->sector_idx * this->size_per_io;
      fnhood_t fnhood = std::make_tuple(loc, loc, sector_buf);
      state->sector_idx++;
      state->frontier_nhoods.push_back(fnhood);
      state->frontier_read_reqs.emplace_back(IORequest(
						       offset, this->size_per_io, sector_buf, 0, 0, nullptr, state));
    }
    this->reader->send_io(state->frontier_read_reqs, ctx, false);
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
SearchExecutionState SSDPartitionIndex<T, TagT>::state_explore_frontier(SearchState<T, TagT> *state) {
  auto nk = state->cur_list_size;

  for (auto &frontier_nhood : state->frontier_nhoods) {
    auto [id, loc, sector_buf] = frontier_nhood;
    char *node_disk_buf = this->offset_to_loc(sector_buf, loc);
    unsigned *node_buf = this->offset_to_node_nhood(node_disk_buf);
    uint64_t nnbrs = (uint64_t)(*node_buf);
    T *node_fp_coords = this->offset_to_node_coords(node_disk_buf);

    T *node_fp_coords_copy =
      state->data_buf + (state->data_buf_idx * this->aligned_dim);
    state->data_buf_idx++;
    memcpy(node_fp_coords_copy, node_fp_coords,
           this->data_dim * sizeof(T));
    float cur_expanded_dist = this->dist_cmp->compare(
							state->query, node_fp_coords_copy, (unsigned)this->aligned_dim);

    pipeann::Neighbor n(id, cur_expanded_dist, true);
    state->full_retset.push_back(n);

    unsigned *node_nbrs = (node_buf + 1);
    // compute node_nbrs <-> query dist in PQ space
    state_compute_dists(state, node_nbrs, nnbrs, state->dist_scratch);

    // process prefetch-ed nhood
    for (uint64_t m = 0; m < nnbrs; ++m) {
      unsigned id = node_nbrs[m];
      if (state->visited.find(id) != state->visited.end()) {
        continue;
      } else {
        state->visited.insert(id);
        state->cmps++;
        float dist = state->dist_scratch[m];
        if (dist >= state->retset[state->cur_list_size - 1].distance &&
            (state->cur_list_size == state->l_search))
          continue;
        pipeann::Neighbor nn(id, dist, true);
        // variable search_L for deleted nodes.
        // Return position in sorted list where nn inserted.

        auto r = InsertIntoPool(state->retset.data(), state->cur_list_size, nn);

        if (state->cur_list_size < state->l_search) {
          state->cur_list_size++;
        }

        if (r < nk)
          nk = r;
      }
    }
  }

  if (nk <= state->k) {
    state->k = nk; // k is the best position in retset updated in this round.
  }else {
    state->k++;
  }
  if (state_search_ends(state)) {
    return SearchExecutionState::FINISHED;
  }
  // updates frontier
  state_update_frontier(state);

  if (state->frontier.empty()) {
    return SearchExecutionState::FINISHED;
  }
  if (this->num_partitions > 1) {
    if (this->get_cluster_assignment(state->frontier[0]) != this->my_partition_id) {
      return SearchExecutionState::TOP_CAND_NODE_OFF_SERVER;
    }
  }

  return SearchExecutionState::TOP_CAND_NODE_ON_SERVER;
}


template <typename T, typename TagT>
bool SSDPartitionIndex<T, TagT>::state_search_ends(SearchState<T, TagT> *state) {
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
  while (marker < state->cur_list_size && state->frontier.size() < state->beam_width &&
         num_seen < state->beam_width) {
    if (state->retset[marker].flag) {
      num_seen++;
      state->frontier.push_back(state->retset[marker].id);
      state->retset[marker].flag = false;
    }
    marker++;
  }    
}


// Explicit instantiations for the SSDPartitionIndex specializations used by the program.
// Put these in this .cpp because the template definitions are here.
template class SSDPartitionIndex<float, uint32_t>;
template class SSDPartitionIndex<unsigned char, uint32_t>;
template class SSDPartitionIndex<signed char, uint32_t>;
