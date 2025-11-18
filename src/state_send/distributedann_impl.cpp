#include "neighbor.h"
#include "query_buf.h"
#include "ssd_partition_index.h"
#include "types.h"
#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include "singleton_logger.h"


template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::DistributedANNWorkerThread::
    DistributedANNWorkerThread(SSDPartitionIndex<T, TagT> *parent)
: parent(parent), thread_ctok(parent->distributed_ann_task_queue) {}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNWorkerThread::start() {
  running = true;
  real_thread = std::thread(
			    &SSDPartitionIndex<T, TagT>::DistributedANNWorkerThread::main_loop, this);
  
}



template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNWorkerThread::signal_stop() {
  running = false;
  //timout in wait will cause the thread to exit
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNWorkerThread::join() {
  if (real_thread.joinable()) {
    real_thread.join();
  }
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNWorkerThread::
    compute_head_index_query(QueryEmbedding<T> *query,
                             distributedann::result_t<T> *result) {
  static thread_local std::array<unsigned,MAX_L_SEARCH> mem_tags;
  static thread_local std::array<float, MAX_L_SEARCH> mem_fp_dists;
  static thread_local std::array<float, MAX_L_SEARCH> mem_pq_dists;
  // this should be sufficient because MAX_L_SEARCH is also max results returned
  // by searching mem index
  static thread_local uint8_t
  pq_coord_scratch[MAX_L_SEARCH * MAX_NUM_PQ_CHUNKS];
  
  if (unlikely(query->mem_l == 0)) {
    throw std::invalid_argument("must use in mem index, mem_l can't be 0");
  }
  if (unlikely(query->mem_l > MAX_L_SEARCH)) {
    throw std::invalid_argument(
        "mem_l too big: " + std::to_string(query->mem_l) + " " +
        std::to_string(MAX_L_SEARCH));
  }
  // std::vector<unsigned> mem_tags(query->mem_l);
  // std::vector<float> mem_dists(query->mem_l);


  if (query == nullptr) {
    throw std::invalid_argument("query can't be nullptr");
  }
  if (!query->populated_pq_dists) {
    parent->pq_table.populate_chunk_distances(query->query, query->pq_dists);
    query->populated_pq_dists = true;
  }

  parent->mem_index_->search_with_tags(query->query, query->mem_l, query->mem_l,
                                       mem_tags.data(), mem_fp_dists.data());

  parent->compute_pq_dists(query->pq_dists, pq_coord_scratch, mem_tags.data(),
                           query->mem_l, mem_pq_dists.data());
  result->query_id = query->query_id;
  result->num_pq_nbrs = query->mem_l;
  for (uint64_t i = 0; i < result->num_pq_nbrs; i++) {
    result->sorted_pq_nbrs[i] = {mem_tags[i], mem_pq_dists[i]};
  }
  result->distributed_ann_state_ptr = query->distributed_ann_state_ptr;
  result->client_peer_id = query->client_peer_id;
  // need to do pq distance now fuckkkkkkkk
  // need to make the handler and query_emb_map work fuckkkking christ
  
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNWorkerThread::
    compute_scoring_query(distributedann::scoring_query_t<T> *scoring_query,
                          distributedann::result_t<T> *result) {
  static thread_local std::vector<std::pair<uint32_t, float>> retset;
  retset.clear();
  retset.reserve(distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN * MAX_NUM_NEIGHBORS);
  
  // static thread_local std::vector<char> sectors;
  // sectors.resize(SECTOR_LEN * distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN);
  static thread_local char* sectors = nullptr;
  static thread_local size_t sectors_size = 0;

  size_t needed_size = SECTOR_LEN * distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN;
  if (!sectors || sectors_size < needed_size) {
    if (sectors) free(sectors);
    sectors = static_cast<char*>(aligned_alloc(4096, needed_size));
    sectors_size = needed_size;
  }

  
  static thread_local std::vector<uint8_t> pq_coord_scratch;
  pq_coord_scratch.resize(distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN * 
                          MAX_NUM_NEIGHBORS * MAX_NUM_PQ_CHUNKS);
  
  static thread_local std::vector<float> pq_dist_scratch;
  pq_dist_scratch.resize(distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN * 
                         MAX_NUM_NEIGHBORS * MAX_NUM_PQ_CHUNKS);
  
  static thread_local std::vector<T> fp_data_buf;
  fp_data_buf.resize(kMaxVectorDim * distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN);


  if (scoring_query->query_emb == nullptr) {
    scoring_query->query_emb =
      parent->query_emb_map.find(scoring_query->query_id);
  }
  if (!scoring_query->query_emb->populated_pq_dists) {
    parent->pq_table.populate_chunk_distances(
					      scoring_query->query_emb->query, scoring_query->query_emb->pq_dists);
    scoring_query->query_emb->populated_pq_dists = true;
  }

  // need to compute the pq dist here instead of at the receiver end
  // 
  
  uint32_t sector_scratch_idx = 0;
 
  std::vector<fnhood_t> frontier_nhoods;
  std::vector<IORequest> frontier_read_reqs;
  if (scoring_query->record_stats) {
    result->stats = std::make_shared<QueryStats>();
  }
  

  for (uint32_t i = 0; i < scoring_query->num_node_ids; i++) {
    uint32_t node_id = scoring_query->node_ids[i];
    uint32_t loc = parent->id2loc(node_id);
    uint64_t offset = parent->loc_sector_no(loc) * SECTOR_LEN;
    if (sector_scratch_idx >= distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN) {
      throw std::runtime_error("sector_scratch idx too large");
    }
    auto sector_buf = sectors + sector_scratch_idx * parent->size_per_io;
    fnhood_t fnhood = std::make_tuple(node_id, loc, sector_buf);
    sector_scratch_idx++;
    frontier_nhoods.push_back(fnhood);
    frontier_read_reqs.emplace_back(
        IORequest(offset, parent->size_per_io, sector_buf,
                  parent->u_loc_offset(loc), parent->max_node_len));

    if (result->stats != nullptr) {
      result->stats->n_4k++;
      result->stats->n_ios++; 
    }
  }
  parent->reader->read(frontier_read_reqs, this->ctx);


  for (auto &frontier_nhood : frontier_nhoods) {
    auto [id, loc, sector_buf] = frontier_nhood;      
    char *node_disk_buf = parent->offset_to_loc(sector_buf, loc);
    unsigned *node_buf = parent->offset_to_node_nhood(node_disk_buf);
    uint64_t nnbrs = (uint64_t)(*node_buf);
    T *node_fp_coords = parent->offset_to_node_coords(node_disk_buf);
    T *node_fp_coords_copy = fp_data_buf.data();
    memcpy(node_fp_coords_copy, node_fp_coords, parent->data_dim * sizeof(T));
    float cur_expanded_dist = parent->dist_cmp->compare(
        scoring_query->query_emb->query, node_fp_coords_copy,
							(unsigned)parent->aligned_dim);
    // LOG(INFO) << id << " " << cur_expanded_dist;

    // full_retset.emplace_back(id, cur_expanded_dist);
    result->sorted_full_nbrs[result->num_full_nbrs++] = {id, cur_expanded_dist};

    unsigned *node_nbrs = (node_buf + 1);
    // LOG(INFO) << id << " " << nnbrs << ": " << list_to_string<unsigned>(node_nbrs, nnbrs);

    parent->compute_pq_dists(scoring_query->query_emb->pq_dists, pq_coord_scratch.data(), node_nbrs,
                             nnbrs, pq_dist_scratch.data());
    for (uint64_t m = 0; m < nnbrs; m++) {
      unsigned id = node_nbrs[m];

      float dist = pq_dist_scratch[m];
      if (result->stats != nullptr) {
        result->stats->n_cmps++;
      }
      // worry about stats here
      if (dist <= scoring_query->threshold) {
        // LOG(INFO) << scoring_query->threshold;
        retset.emplace_back(id, dist);
      }
    }
    // worry about stats here
  }
  if (result->num_full_nbrs != scoring_query->num_node_ids) {
    throw std::runtime_error(
			     "Number of results doesn't match the number of node ids");
  }
  std::sort(result->sorted_full_nbrs, result->sorted_full_nbrs + result->num_full_nbrs,
            [](auto a, auto b) { return a.second < b.second; });
  std::sort(retset.begin(), retset.end(),
            [](auto a, auto b) { return a.second < b.second; });
  result->num_pq_nbrs =
    std::min(static_cast<uint32_t>(retset.size()), scoring_query->L);
  for (auto i = 0; i < result->num_pq_nbrs; i++) {
    result->sorted_pq_nbrs[i] = retset[i];
  }
  result->distributed_ann_state_ptr = scoring_query->distributed_ann_state_ptr;
  result->client_peer_id = scoring_query->client_peer_id;
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNWorkerThread::main_loop() {
  this->parent->reader->register_thread();
  ctx = this->parent->reader->get_ctx();  
  auto timeout = std::chrono::milliseconds(100);
  distributedann::DistributedANNTask<T> task;
  while (running) {
    bool dequeued = parent->distributed_ann_task_queue.wait_dequeue_timed(
									  thread_ctok, task, timeout);
    // LOG(INFO) << "HOLA";
    if (!dequeued)
      continue;
    distributedann::result_t<T> *result = nullptr;
    parent->prealloc_distributedann_result.dequeue_exact(1, &result);
    if (task.task_type == distributedann::DistributedANNTaskType::HEAD_INDEX) {
      // LOG(INFO) << "task type is distributedann::DistributedANNTaskType::HEAD_INDEX";
      QueryEmbedding<T> *query = std::get<QueryEmbedding<T> *>(task.task);
      compute_head_index_query(query, result);
      
      // query will be deallocated once the ack arrives
    } else if (task.task_type ==
               distributedann::DistributedANNTaskType::SCORING_QUERY) {
      // LOG(INFO) << "task type is distributedann::DistributedANNTaskType::SCORING_QUERY";      
      distributedann::scoring_query_t<T> *query =
        std::get<distributedann::scoring_query_t<T> *>(task.task);
      compute_scoring_query(query, result);
      parent->prealloc_distributedann_scoring_query.free(query);
    } else {
      throw std::invalid_argument("weird task type value");
    }
    parent->distributedann_batching_thread->push_result_to_batch(result);
    // result will be deallocated in batching thread once it is sent
    // need to send result to batch
  }
}




template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::DistributedANNBatchingThread::
    DistributedANNBatchingThread(SSDPartitionIndex<T, TagT> *parent)
: parent(parent) {}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNBatchingThread::start() {
  running = true;
  real_thread = std::thread(
      &SSDPartitionIndex<T, TagT>::DistributedANNBatchingThread::main_loop,
			    this);
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNBatchingThread::signal_stop() {
  running = false;
}



template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNBatchingThread::join() {
  if (real_thread.joinable()) {
    real_thread.join();
  }
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNBatchingThread::
    push_result_to_batch(distributedann::result_t<T> *result) {
  std::unique_lock lock(result_queue_mutex);
  if (!result_queue.contains(result->client_peer_id)) {
    result_queue[result->client_peer_id] =
      std::make_unique<std::vector<distributedann::result_t<T> *>>();
    result_queue[result->client_peer_id]->reserve(parent->max_batch_size);
  }
  result_queue[result->client_peer_id]->emplace_back(result);
  result_queue_cv.notify_all();
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::DistributedANNBatchingThread::main_loop() {
  std::unique_lock<std::mutex> lock(result_queue_mutex, std::defer_lock);

  auto timeout = std::chrono::milliseconds(100);
  auto result_queue_empty = [this]() {
    for (const auto &[peer_id, results] : this->result_queue) {
      if (!results->empty())
        return false;
    }
    return true;
  };
  while (running) {
    lock.lock();
    while (result_queue_empty() && running) {
      result_queue_cv.wait_for(lock, timeout);
    }
    // after this, either result queue is not empty (there is work to do) or
    // running = false, inwhich case we need to check
    if (!running) {
      break;
    }

    std::unordered_map<
        uint64_t, std::unique_ptr<std::vector<distributedann::result_t<T> *>>>
    results_to_send;

    for (auto &[client_peer_id, results] : result_queue) {
      results_to_send[client_peer_id] = std::move(result_queue[client_peer_id]);
      result_queue[client_peer_id] =
        std::make_unique<std::vector<distributedann::result_t<T> *>>();
      result_queue[client_peer_id]->reserve(parent->max_batch_size);
    }
    lock.unlock();
    for (auto &[client_peer_id, results] : results_to_send) {
      uint64_t num_sent = 0;
      uint64_t total = results->size();
      while (num_sent < total) {
        uint64_t left = total - num_sent;
        uint64_t batch_size = std::min(parent->max_batch_size, left);
        Region r;
        std::vector<distributedann::result_t<T> *> result_batch;
        result_batch.reserve(parent->max_batch_size);
        for (uint64_t i = num_sent; i < num_sent + batch_size; i++) {
          result_batch.emplace_back(results->at(i));
        }
        r.length = sizeof(MessageType) +
                   distributedann::result_t<T>::get_serialize_size_results(
									   result_batch);
        r.addr = new char[r.length];
        size_t offset = 0;
        MessageType msg_type = MessageType::DISTRIBUTED_ANN_RESULTS;
        std::memcpy(r.addr + offset, &msg_type, sizeof(msg_type));
        offset += sizeof(msg_type);
        distributedann::result_t<T>::write_serialize_results(r.addr + offset,
                                                             result_batch);
        parent->communicator->send_to_peer(client_peer_id, r);
        num_sent += batch_size;
      }
      for (auto &result : *results) {
        parent->prealloc_distributedann_result.free(result);
      }
    }
  }
}



template class SSDPartitionIndex<float>::DistributedANNWorkerThread;
template class SSDPartitionIndex<uint8_t>::DistributedANNWorkerThread;
template class SSDPartitionIndex<int8_t>::DistributedANNWorkerThread;

template class SSDPartitionIndex<float>::DistributedANNBatchingThread;
template class SSDPartitionIndex<uint8_t>::DistributedANNBatchingThread;
template class SSDPartitionIndex<int8_t>::DistributedANNBatchingThread;

