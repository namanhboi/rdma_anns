#include "query_buf.h"
#include "ssd_partition_index.h"
#include "utils.h"
#include <stdexcept>

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::ScoringThread::ScoringThread(
    SSDPartitionIndex<T, TagT> *parent, uint64_t thread_id)
    : parent(parent), thread_id(thread_id),
    search_thread_consumer_token(parent->global_state_queue) {}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::ScoringThread::start() {
  running = true;
  real_thread = std::thread(
			    &SSDPartitionIndex<T, TagT>::ScoringThread::main_loop_batch, this);
  
}
template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::ScoringThread::signal_stop() {
  if (this->ctx == nullptr) {
    throw std::runtime_error(
        "tried stopping search threads but ctx is nullptr");
  }
  running = false;
  IORequest *noop_req = new IORequest;
  this->parent->reader->send_noop(noop_req, this->ctx);
}
template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::ScoringThread::join() {
  if (real_thread.joinable()) {
    real_thread.join();
  }
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::ScoringThread::add_states_to_batch(
    SearchState<T, TagT> **states, size_t num_states) {
  for (size_t i = 0; i < num_states; i++) {
    if (states[i] == nullptr) {
      assert(this->running == false);
      // poison pill from queue
      break;
    }
    if (!states[i]->is_distributed_ann_scoring_state) {
      throw std::runtime_error("not a distributedaann scoring state");
    }
    // LOG(INFO) << "size of frontier is " << states[i]->frontier.size();
    // parent->state_print_detailed(states[i]);
    if (states[i]->query_emb == nullptr) {
      states[i]->query_emb =
          this->parent->query_emb_map.find(states[i]->query_id);
    }
    if (!states[i]->query_emb->normalized) {
      if (this->parent->metric == pipeann::Metric::COSINE ||
          this->parent->metric == pipeann::Metric::INNER_PRODUCT) {
        // LOG(INFO) << "normalizing metric " <<
        // pipeann::get_metric_str(parent->metric);

        // inherent_dim is the dim of the actuall query
        uint64_t inherent_dim = this->parent->metric == pipeann::Metric::INNER_PRODUCT
                                    ? this->parent->data_dim - 1
                                    : this->parent->data_dim;
        if (unlikely(inherent_dim != states[i]->query_emb->dim)) {
          throw std::runtime_error("inherint dim diff from query dim");
        }
        float query_norm = 0;
        for (size_t j = 0; j < inherent_dim; j++) {
          query_norm += states[i]->query_emb->query[j] *
                        states[i]->query_emb->query[j];
        }
        if (this->parent->metric == pipeann::Metric::INNER_PRODUCT) {
          states[i]->query_emb->query[this->parent->data_dim - 1] = 0;
          // zero the extra dim because of mips conversion to l2 having 1
          // extra dim
        }
        query_norm = std::sqrt(query_norm);
        // query_norm = 1;
        for (size_t j = 0; j < inherent_dim; j++) {
          states[i]->query_emb->query[j] =
              (T)(states[i]->query_emb->query[j] / query_norm);
        }

        states[i]->query_emb->query_norm = query_norm;
      }
      states[i]->query_emb->normalized = true;
    }
    // need to initialize pq data on scoring server
    if (!states[i]->query_emb->populated_pq_dists) {
      this->parent->pq_table.populate_chunk_distances_l2(
          states[i]->query_emb->query, states[i]->query_emb->pq_dists);
      states[i]->query_emb->populated_pq_dists = true;
    }

    // now do mem index or centroid calculation
    if (states[i]->cur_list_size != 0) {
      throw std::runtime_error("cur_list_size should be 0, meaning retset is "
                               "empty since scoring server will fill it");
    }
    this->number_concurrent_queries++;
    // this->number_own_states++;
    this->parent->num_new_states_global_queue--;
    assert(states[i]->partition_history.size() == 1);

    // states[i]->query_timer.reset();
    // allocated_states[i]->io_timer.reset();
    if (states[i]->frontier.empty()) {
      if (states[i]->mem_l > 0) {
        assert(parent->mem_index_ != nullptr);
        std::vector<unsigned> mem_tags(states[i]->mem_l);
        std::vector<float> mem_dists(states[i]->mem_l);
        this->parent->mem_index_->search_with_tags(
            states[i]->query_emb->query, states[i]->mem_k, states[i]->mem_l,
						   mem_tags.data(), mem_dists.data());
        this->parent->state_compute_and_add_to_retset(
            states[i], mem_tags.data(),
            std::min((unsigned)states[i]->mem_l,
                     (unsigned)states[i]->l_search));
	
        assert(states[i]->cur_list_size > 0);
      } else {
        uint32_t best_medoid = this->parent->medoids[0];
        this->parent->state_compute_and_add_to_retset(states[i], &best_medoid,
                                                      1);
        assert(states[i]->cur_list_size > 0);
      }
      // send back to server mate
      this->parent->state_finalize_distance(states[i]);
      // LOG(INFO) << "Done with mem search";
      // parent->state_print_detailed(states[i]);
      number_concurrent_queries--;
      // number_own_states--;
      this->parent->notify_client(states[i]);
    } else {
      // LOG(INFO) << " frontier not empty, issueing io"; 
      // this->number_concurrent_queries++;
      // this->number_foreign_states++;
      this->parent->num_foreign_states_global_queue--;
      this->parent->state_issue_next_io_batch(states[i], this->ctx);
    }
  }
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::ScoringThread::main_loop_batch() {
  LOG(INFO) << "executing main_loop_batch";
  this->parent->reader->register_thread();
  this->ctx = this->parent->reader->get_ctx();
  if (this->ctx == nullptr) {
    throw std::runtime_error("ctx given by get_ctx is nullptr");
  }

  std::array<SearchState<T, TagT> *, max_queries_balance> allocated_states;
  while (this->running) {
    uint64_t num_states_to_dequeue =
      this->parent->num_queries_balance - this->number_concurrent_queries;

    if (num_states_to_dequeue > 0) {
      // LOG(INFO) << "num_states_to_dequeue " << num_states_to_dequeue;
      size_t num_dequeued = this->parent->global_state_queue.try_dequeue_bulk(
          this->search_thread_consumer_token, allocated_states.begin(),
									      num_states_to_dequeue);
      
      if (num_dequeued != 0) {
        // LOG(INFO) << "num dequeued" << num_dequeued;
        add_states_to_batch(allocated_states.data(), num_dequeued);
      }
    }
    IORequest *req = this->parent->reader->poll(this->ctx);
    if (req == nullptr)
      continue;

    if (req->search_state == nullptr) {
      std::cerr << "poison pill detected" << std::endl;
      // this is a poison pill to shutdown the thread
      break;
    }

    SearchState<T, TagT> *state =
      reinterpret_cast<SearchState<T, TagT> *>(req->search_state);
    // LOG(INFO) << "arrived";

    if (!this->parent->state_io_finished(state)) {
      continue;
    }
    this->parent->state_explore_frontier_scoring_simple(state);
    // send result to batch
    if (state->stats != nullptr) {
      state->stats->total_us += (double)state->query_timer.elapsed();
    }
    state->query_timer.reset();
    this->number_concurrent_queries--;
    if (state->partition_history.size() == 1) {
      this->number_own_states--;
    } else {
      this->number_foreign_states--;
    }
    this->parent->state_finalize_distance(state);
    // LOG(INFO) << "notifying client ";
    // parent->state_print_detailed(state);
    this->parent->notify_client(state);
  }
}
template class SSDPartitionIndex<float>::ScoringThread;
template class SSDPartitionIndex<uint8_t>::ScoringThread;
template class SSDPartitionIndex<int8_t>::ScoringThread;

