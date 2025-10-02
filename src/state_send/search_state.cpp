#include "ssd_partition_index.h"

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
    if (parent->get_cluster_assignment(frontier[0]) != parent->my_cluster_id) {
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

template <typename T, typename TagT> uint64_t get_serialize_size() {
  

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


template struct SSDPartitionIndex<float>::SearchState;
template struct SSDPartitionIndex<uint8_t>::SearchState;
template struct SSDPartitionIndex<int8_t>::SearchState;


