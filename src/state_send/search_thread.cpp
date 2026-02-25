#include "distance.h"
#include "query_buf.h"
#include "ssd_partition_index.h"
#include "types.h"
#include <array>
#include <chrono>
#include <iomanip>
#include <stdexcept>

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::SearchThread::SearchThread(
    SSDPartitionIndex *parent, uint64_t thread_id)
    : parent(parent), thread_id(thread_id),
    search_thread_consumer_token(parent->global_state_queue) {}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::start() {
  running = true;
  real_thread = std::thread(
			    &SSDPartitionIndex<T, TagT>::SearchThread::main_loop_batch, this);
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::signal_stop() {
  // right now, we issue every io operation through the
  // issue a read but with search_state in io request = nullptr
  // when main_loop in search thread gets the cqe, check this, and continue, in
  // which case the loop guard is checked and thread finishes

  // need to free this after
  if (this->ctx == nullptr) {
    throw std::runtime_error(
			     "tried stopping search threads but ctx is nullptr");
  }
  running = false;
  IORequest *noop_req = new IORequest;
  this->parent->reader->send_noop(noop_req, this->ctx);
}



template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::main_loop_batch() {
  LOG(INFO) << "executing main_loop_batch";
  this->parent->reader->register_thread();
  ctx = this->parent->reader->get_ctx();
  if (ctx == nullptr) {
    throw std::runtime_error("ctx given by get_ctx is nullptr");
  }

  std::array<SearchState<T, TagT> *, max_queries_balance> allocated_states;
  // std::cout << "pq data dim is " << parent->data_dim;

  while (running) {
    // LOG(INFO) <<"Concurrent queries " <<number_concurrent_queries;
    assert(parent->num_queries_balance >= number_concurrent_queries);
    uint64_t num_states_to_dequeue = parent->num_queries_balance - number_concurrent_queries;
    // LOG(INFO) << "number of concurrent queries " << number_concurrent_queries;
    if (num_states_to_dequeue > 0) {
      // size_t num_dequeued = thread_state_queue.try_dequeue_bulk(
							 // allocated_states.begin(), num_states_to_dequeue);
      size_t num_dequeued = parent->global_state_queue.try_dequeue_bulk(
          search_thread_consumer_token, allocated_states.begin(),
									num_states_to_dequeue);
      for (size_t i = 0; i < num_dequeued; i++) {
        if (allocated_states[i] == nullptr) {
          assert(running == false);
          //poison pill from queue
          break;
        }
        if (allocated_states[i]->query_emb == nullptr) {
          allocated_states[i]->query_emb =
            parent->query_emb_map.find(allocated_states[i]->query_id);
        }

        // for (size_t j = 0; j < parent->data_dim; j++) {
        //   std::cout << j << "," << std::fixed << std::setprecision(9)
        //   << allocated_states[i]->query_emb->query[j] << " ";
        // }
        // std::cout << std::endl;

        if (!allocated_states[i]->query_emb->normalized) {
          // LOG(INFO) << "NOT NORMALIZED";
          if (parent->metric == pipeann::Metric::COSINE ||
              parent->metric == pipeann::Metric::INNER_PRODUCT) {
            // LOG(INFO) << "normalizing metric " <<
            // pipeann::get_metric_str(parent->metric);

            // inherent_dim is the dim of the actuall query
            uint64_t inherent_dim =
                parent->metric == pipeann::Metric::INNER_PRODUCT
                    ? parent->data_dim - 1
                : parent->data_dim;
            if (unlikely(inherent_dim != allocated_states[i]->query_emb->dim)) {
              throw std::runtime_error("inherint dim diff from query dim");
            }
            float query_norm = 0;
            for (size_t j = 0; j < inherent_dim; j++) {
              query_norm += allocated_states[i]->query_emb->query[j] *
                            allocated_states[i]->query_emb->query[j];
            }
            if (parent->metric == pipeann::Metric::INNER_PRODUCT) {
              allocated_states[i]->query_emb->query[parent->data_dim - 1] = 0;
              // zero the extra dim because of mips conversion to l2 having 1
              // extra dim
            }
            query_norm = std::sqrt(query_norm);
            // query_norm = 1;
            for (size_t j = 0; j < inherent_dim; j++) {
              allocated_states[i]->query_emb->query[j] =
                (T)(allocated_states[i]->query_emb->query[j] / query_norm);
            }
            
            allocated_states[i]->query_emb->query_norm = query_norm;
          }
          allocated_states[i]->query_emb->normalized = true;
        }
        if (!allocated_states[i]->query_emb->populated_pq_dists) {
          // LOG(INFO) << "NOT INITIALIZED YET";
          parent->pq_table.populate_chunk_distances_l2(
              allocated_states[i]->query_emb->query,
						    allocated_states[i]->query_emb->pq_dists);
          allocated_states[i]->query_emb->populated_pq_dists = true;
        }
	// std::cout << "query_norm is " << allocated_states[i]->query_emb->query_norm << std::endl;
        // for (size_t j = 0; j < parent->data_dim; j++) {
	//   std::cout << std::fixed << std::setprecision(8)<< allocated_states[i]->query_emb->query[j] << " ";
        // }
	// std::cout << std::endl;
        // if (allocated_states[i]->query_emb->query_norm == 0.0 && ) {

        // }
        // initialize the result set: either with in mem index or by just using
        // the medoid
        // brand new state, must be sent from client
        if (allocated_states[i]->cur_list_size == 0) {
          number_concurrent_queries++;
          number_own_states++;
          parent->num_new_states_global_queue--;
          assert(allocated_states[i]->partition_history.size() == 1);

          allocated_states[i]->query_timer.reset();
          // allocated_states[i]->io_timer.reset();          

          if (allocated_states[i]->mem_l > 0) {
            assert(parent->mem_index_ != nullptr);
            std::vector<unsigned> mem_tags(allocated_states[i]->mem_l);
            std::vector<float> mem_dists(allocated_states[i]->mem_l);
            parent->mem_index_->search_with_tags(
                allocated_states[i]->query_emb->query,
                allocated_states[i]->mem_l, allocated_states[i]->mem_l,
						 mem_tags.data(), mem_dists.data());
            parent->state_compute_and_add_to_retset(
                allocated_states[i], mem_tags.data(),
                std::min((unsigned)allocated_states[i]->mem_l,
                         (unsigned)allocated_states[i]->l_search));
            // parent->state_print_detailed(allocated_states[i]);
            assert(allocated_states[i]->cur_list_size > 0);
          } else {
	    uint32_t best_medoid = parent->medoids[0];
            parent->state_compute_and_add_to_retset(allocated_states[i],
                                                    &best_medoid, 1);
            assert(allocated_states[i]->cur_list_size > 0);
          }

          // allocated_states[i]->query_timer.reset();
          UpdateFrontierValue ret_val =
            parent->state_update_frontier(allocated_states[i]);
          if (ret_val == UpdateFrontierValue::FRONTIER_EMPTY_ONLY_OFF_SERVER) {
            if (allocated_states[i]->stats) {
              allocated_states[i]->stats->total_us +=
		allocated_states[i]->query_timer.elapsed();
            }
            allocated_states[i]->query_timer.reset();
            parent->send_state(allocated_states[i]);
            // send state will delete the state later
            number_concurrent_queries--;
            number_own_states--;
          } else if (ret_val == UpdateFrontierValue::FRONTIER_HAS_ON_SERVER) {
            allocated_states[i]->io_timer.reset();
            parent->state_issue_next_io_batch(allocated_states[i], ctx);
          } else if (ret_val == UpdateFrontierValue::FRONTIER_EMPTY_NO_OFF_SERVER){
            throw std::runtime_error(
                "frontier can't be actually empty because we just added either "
                "return value from mem index or medoid/node 0 to it");
          } else {
	    throw std::runtime_error("werid return value from update frontier");
          }
        } else {
          // assert(parent->dist_search_mode == DistributedSearchMode::STATE_SEND);
          number_concurrent_queries++;
          number_foreign_states++;
          parent->num_foreign_states_global_queue--;
          // state that was sent, need to check that the top node in frontier is
          // on this server
          allocated_states[i]->io_timer.reset();
          allocated_states[i]->query_timer.reset();
          parent->state_update_frontier(allocated_states[i]);
          parent->state_issue_next_io_batch(allocated_states[i], ctx);
        }
      }
    }

    IORequest *req = this->parent->reader->poll(ctx);
    if (req == nullptr)
      continue;

    if (req->search_state == nullptr) {
      std::cerr << "poison pill detected" << std::endl;
      // this is a poison pill to shutdown the thread
      break;
    }

    SearchState<T, TagT> *state =
      reinterpret_cast<SearchState<T, TagT> *>(req->search_state);

    if (!parent->state_io_finished(state)) {
      continue;
    }
    if (state->stats) {
      state->stats->io_us += state->io_timer.elapsed();
    }
    // parent->state_print_detailed(state);

    SearchExecutionState s = SearchExecutionState::FRONTIER_EMPTY;
    // this loop will advance the state until there is something to read or
    // the state ends
    while (s == SearchExecutionState::FRONTIER_EMPTY) {
      s = parent->state_explore_frontier(state);
      // LOG(INFO) << "bruh";
    }
    if (s == SearchExecutionState::FINISHED) {
      if (state->stats != nullptr) {
        state->stats->total_us += (double)state->query_timer.elapsed();
      }
      state->query_timer.reset();
      number_concurrent_queries--;
      if (state->partition_history.size() == 1) {
        number_own_states--;
      } else {
	number_foreign_states--;
      }
      // LOG(INFO) << "DONE WITH QUERY " << state->query_emb->query_id;
      // std::cout << "results " <<std::endl;
      // for (size_t i = 0; i < state->k; i++) {
	// std::cout << state->full_retset[i].id << " " << state->full_retset[i].distance << ",";
	// }
      
      this->parent->state_finalize_distance(state);
      // for (size_t i = 0; i < state->k_search; i++) {
      //   std::cout << "(" << state->full_retset[i].id << " "
      //   << state->full_retset[i].distance << "),";
      // }
      // std::cout << std::endl;
      this->parent->notify_client(state);
    } else if (s == SearchExecutionState::FRONTIER_ON_SERVER) {
      // LOG(INFO) << "Issuing io";
      state->io_timer.reset();
      parent->state_issue_next_io_batch(state, ctx);
    } else if (s == SearchExecutionState::FRONTIER_OFF_SERVER) {
      if (state->partition_history.size() == 1) {
        number_own_states--;
      } else {
	number_foreign_states--;
      }
      number_concurrent_queries--;
      if (state->stats != nullptr) {
        state->stats->total_us += state->query_timer.elapsed();
      }
      state->query_timer.reset();
      // LOG(INFO) << "sending state";
      parent->send_state(state);
    }    
  }
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::join() {
  if (real_thread.joinable()) {
    real_thread.join();
  }
}


template class SSDPartitionIndex<float>::SearchThread;
template class SSDPartitionIndex<uint8_t>::SearchThread;
template class SSDPartitionIndex<int8_t>::SearchThread;
