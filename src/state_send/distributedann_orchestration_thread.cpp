#include "query_buf.h"
#include "ssd_partition_index.h"
#include "types.h"

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::OrchestrationThread::OrchestrationThread(
    SSDPartitionIndex<T, TagT> *parent, uint64_t thread_id)
    : parent(parent), thread_id(thread_id),
      state_consumer_token(parent->global_state_queue),
      result_consumer_token(computation_result_queue) {}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::OrchestrationThread::start() {
  running = true;
  real_thread = std::thread(
			    &SSDPartitionIndex<T, TagT>::OrchestrationThread::main_loop_batch, this);
  
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::OrchestrationThread::signal_stop() {
  running = false;
  enqueue_computation_result(nullptr);
  // need to do something with the queue to stop, like batching thread

}
template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::OrchestrationThread::join() {
  // similar to search thread
  if (real_thread.joinable()) {
    real_thread.join();
  }

}





template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::OrchestrationThread::
    enqueue_computation_result(search_result_t *result) {
  while (!computation_result_queue.enqueue(result)) {
  };
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::OrchestrationThread::add_states_to_batch(
    SearchState<T, TagT> **states, size_t num_states) {
  for (size_t i = 0; i < num_states; i++) {
    if (states[i] == nullptr) {
      assert(this->running == false);
      // poison pill from queue
      break;
    }
    states[i]->orchestration_thread_id = this->thread_id;
    if (states[i]->query_emb == nullptr) {
      states[i]->query_emb =
          this->parent->query_emb_map.find(states[i]->query_id);
    }
    if (states[i]->query_emb->normalized) {
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
    // we don't need to populate pq dists because orchestration thread should
    // know nothing about pq or any data related to index except partition
    // assignment

    // push an empty state to push_state
    // scoring server scoring threads will handle empty state depending on mem_l
    // value if mem_l > 0. Destination server will be random
    this->parent->state_send_scoring_queries_distributedann(states[i]);
  }
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::OrchestrationThread::process_state(
								    SearchExecutionState s, SearchState<T, TagT> *state) {

  if (s == SearchExecutionState::FINISHED) {
    //
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
    this->parent->notify_client(state);
  } else if (s == SearchExecutionState::FRONTIER_OFF_SERVER) {
    // this means the state can issue rpc to scoring threads

    // we send states as queries, need distance cutoff, which is the distance of
    // last node in retset

    this->parent->state_send_scoring_queries_distributedann(state);
  } else {
    throw std::runtime_error(
        "For distributedann Don't expect any value except FINISHED, "
        "FRONTIER_OFF_SERVER (meaning an rpc can be issued) ");
  }
}
 

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::OrchestrationThread::main_loop_batch() {
  std::array<SearchState<T, TagT> *, max_queries_balance> allocated_states;
  
  while (this->running) {
    uint64_t num_states_to_dequeue =
        this->parent->num_queries_balance - this->number_concurrent_queries;
    if (num_states_to_dequeue > 0) {
      size_t num_dequeued = this->parent->global_state_queue.try_dequeue_bulk(
          this->state_consumer_token, allocated_states.begin(),
									      num_states_to_dequeue);
      add_states_to_batch(allocated_states.data(), num_dequeued);
    }
    search_result_t *computation_result;
    this->parent->preallocated_result_queue.dequeue_exact(1,
                                                          &computation_result);
    bool dequeued = computation_result_queue.try_dequeue(result_consumer_token,
                                                         computation_result);
    if (!dequeued) {
      this->parent->preallocated_result_queue.free(computation_result);
      continue;
    }
    if (computation_result == nullptr) {
      if (running) {
	throw std::runtime_error("computation result is null when still running ");
      }
      break;
    }
    if (!running) break;
    
    // need to call function to update state with results
    IORequest *io_req= static_cast<IORequest *>(computation_result->hint);
    io_req->finished = true;
    SearchState<T, TagT> *state =
      static_cast<SearchState<T, TagT> *>(io_req->search_state);
    state->frontier_distributedann_result.push_back(computation_result);
    if (!this->parent->state_io_finished(state)) {
      continue;
    }
    
    // each io request to a frontier in state is an rpc to scoring service,
    // which computes for a set of nodes its pq distance to neighbors + full
    // distance to the node itself.

    // TODO Look into the frontier stuff and use
    // state_explore_frontier_distributedann need to run until the state is
    // ended

    SearchExecutionState s = SearchExecutionState::FRONTIER_EMPTY;
    while (s == SearchExecutionState::FRONTIER_EMPTY) {
      s = this->parent->state_explore_frontier_orchestration(state);
    }
    this->process_state(s, state);
    // results need to be dequeued once frontier explore is done.
  }

}



template class SSDPartitionIndex<float>::OrchestrationThread;
template class SSDPartitionIndex<uint8_t>::OrchestrationThread;
template class SSDPartitionIndex<int8_t>::OrchestrationThread;
