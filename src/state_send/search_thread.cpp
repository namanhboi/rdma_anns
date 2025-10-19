#include "query_buf.h"
#include "ssd_partition_index.h"
#include "types.h"
#include <array>
#include <chrono>
#include <stdexcept>

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::SearchThread::SearchThread(
    SSDPartitionIndex *parent, uint64_t thread_id)
    : parent(parent), thread_id(thread_id),
    search_thread_consumer_token(parent->global_state_queue) {}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::start() {
  running = true;
#ifdef BALANCE_ALL
  real_thread = std::thread(
			    &SSDPartitionIndex<T, TagT>::SearchThread::main_loop_balance_all, this);
#else
  real_thread = std::thread(
			    &SSDPartitionIndex<T, TagT>::SearchThread::main_loop_batch, this);
#endif
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::main_loop_balance_all() {
  this->parent->reader->register_thread();
  ctx = this->parent->reader->get_ctx();
  if (ctx == nullptr) {
    throw std::runtime_error("ctx given by get_ctx is nullptr");
  }
  while (running) {
    IORequest *req = this->parent->reader->poll_wait(ctx);
    if (req->search_state == nullptr) {
      std::cerr << "poison poll detected" << std::endl;
      // this is a poison pill to shutdown the thread
      break;
    }
    // unsigned int ready = io_uring_cq_ready(reinterpret_cast<io_uring *>(ctx));
    // LOG(INFO) << "number of completions: " << ready;

    SearchState<T, TagT> *state =
      reinterpret_cast<SearchState<T, TagT> *>(req->search_state);
    // assert(state->io_finished(ctx));
    // std::cout << "l and k are  " << state->l_search << " " << state->k_search << std::endl;
    // std::cout << "k and cur_list_size " << state->k << " " << state->cur_list_size << std::endl;
    SearchExecutionState s = parent->state_explore_frontier(state);
    if (s == SearchExecutionState::FINISHED) {
      // TODO:send results to client, delete the state
      if (state->client_type == ClientType::LOCAL) {
        this->parent->notify_client_local(state);
      }
    } else if (s == SearchExecutionState::TOP_CAND_NODE_ON_SERVER) {
      // LOG(INFO) << "Issuing io";
      bool read_issued = parent->state_issue_next_io_batch(state, ctx);
      // assert(read_issued);
      if (state->frontier.empty()) {
        // weird case, when frontier empty, search is technically complete, no read just iterates k to l_search
        this->parent->notify_client(state);
      }
    } else {
      throw std::runtime_error("multiple partitions not yet implemented");
    }
  }
  this->parent->reader->deregister_thread();
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
#ifndef BALANCE_ALL
  thread_state_queue.enqueue(nullptr);
#endif  
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::push_state(
							  SearchState<T, TagT> *new_state) {
  bool ret = thread_state_queue.enqueue(new_state);
  assert(ret);
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::main_loop_batch() {
  LOG(INFO) << "executing main_loop_batch";
  this->parent->reader->register_thread();
  ctx = this->parent->reader->get_ctx();
  if (ctx == nullptr) {
    throw std::runtime_error("ctx given by get_ctx is nullptr");
  }
  uint64_t number_concurrent_queries = 0;
  std::array<SearchState<T, TagT> *, max_queries_balance> allocated_states;

  while (running) {
    // LOG(INFO) <<"Concurrent queries " <<number_concurrent_queries;
    assert(parent->num_queries_balance >= number_concurrent_queries);
    uint64_t num_states_to_dequeue = parent->num_queries_balance - number_concurrent_queries;
    // LOG(INFO) << "number of concurrent queries " << number_concurrent_queries;
    if (num_states_to_dequeue > 0) {
#ifdef PER_THREAD_QUEUE      
      size_t num_dequeued = thread_state_queue.try_dequeue_bulk(
							 allocated_states.begin(), num_states_to_dequeue);
#else
      size_t num_dequeued = parent->global_state_queue.try_dequeue_bulk(
          search_thread_consumer_token, allocated_states.begin(),
									num_states_to_dequeue);
#endif
      // LOG(INFO) << "Dequeued " << num_dequeued;
      for (size_t i = 0; i < num_dequeued; i++) {
        if (allocated_states[i] == nullptr) {
          assert(running == false);
          //poison pill from queue
          break;
        }

        if (allocated_states[i]->stats != nullptr) {
          allocated_states[i]->query_timer.reset();
          allocated_states[i]->io_timer.reset();
          allocated_states[i]->stats->n_4k++;
          allocated_states[i]->stats->n_ios++;
        }

        // initialize the result set: either with in mem index or by just using
        // the medoid
        // brand new state, must be sent from client
        if (allocated_states[i]->cur_list_size == 0) {
          assert(allocated_states[i]->partition_history.size() == 1);
          if (allocated_states[i]->mem_l > 0) {
            // LOG(INFO) << "SEARCHING WITH MEM INDEX";
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
            assert(allocated_states[i]->cur_list_size > 0);
            // parent->state_print_detailed(allocated_states[i]);
          } else {
	    uint32_t best_medoid = parent->medoids[0];
            parent->state_compute_and_add_to_retset(allocated_states[i],
                                                    &best_medoid, 1);
            assert(allocated_states[i]->cur_list_size > 0);
            // parent->state_print_detailed(allocated_states[i]);
          }
          if (parent->dist_search_mode !=
              DistributedSearchMode::STATE_SEND) {
            bool read_issued =
              parent->state_issue_next_io_batch(allocated_states[i], ctx);
            assert(read_issued);
            number_concurrent_queries++;
          } else {

            // uint8_t partition_assignment_top_cand =
              // parent->cluster_assignment[allocated_states[i]->frontier[0]];
            uint8_t partition_assignment_top_cand =
                parent->get_cluster_assignment(
					       allocated_states[i]->frontier[0]);
            if (partition_assignment_top_cand != parent->my_partition_id) {
              parent->send_state(allocated_states[i]);
              delete allocated_states[i];
            } else {
              bool read_issued =
                parent->state_issue_next_io_batch(allocated_states[i], ctx);
              assert(read_issued);
              number_concurrent_queries++;
            }
          }
        } else {
          assert(parent->dist_search_mode == DistributedSearchMode::STATE_SEND);
          // state that was sent, need to check that the top node in frontier is
          // on this server
          uint8_t partition_assignment_top_cand =
            parent->state_top_cand_partition(allocated_states[i]);
          if (partition_assignment_top_cand != parent->my_partition_id) {
            throw std::runtime_error(
                "Partition assigmnent of sent state is not the same as "
                "server " +
                std::to_string(partition_assignment_top_cand) + " " +
                std::to_string(parent->my_partition_id));
          }
          assert(allocated_states[i]->frontier.size()> 0);
          bool read_issued =
            parent->state_issue_next_io_batch(allocated_states[i], ctx);
          assert(read_issued);
          number_concurrent_queries++;
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
    // unsigned int ready = io_uring_cq_ready(reinterpret_cast<io_uring *>(ctx));
    // LOG(INFO) << "Read received: " << req->offset;

    SearchState<T, TagT> *state =
      reinterpret_cast<SearchState<T, TagT> *>(req->search_state);
    if (state->stats != nullptr) {
      state->stats->io_us += (double) state->io_timer.elapsed();
    }
    // if (state->query_emb == nullptr) {
    //   if (parent->query_emb_map.contains(state->query_id)) {
    //     state->query_emb = parent->query_emb_map.find(state->query_id);
    //   } else {
    //     throw std::runtime_error("the query_emb for this state doesn't exist");
    //   }
    // }
    SearchExecutionState s = parent->state_explore_frontier(state);
    // parent->state_print_detailed(state);
    if (s == SearchExecutionState::FINISHED) {
      // state->end_time = std::chrono::steady_clock::now();
      if (state->stats != nullptr) {
        state->stats->total_us += (double)state->query_timer.elapsed();
      }
      this->parent->notify_client(state);
      number_concurrent_queries--;
      if (parent->dist_search_mode == DistributedSearchMode::SCATTER_GATHER) {
        parent->query_emb_map.erase(state->query_id);
      }
      delete state;
    } else if (s == SearchExecutionState::TOP_CAND_NODE_ON_SERVER) {
      // LOG(INFO) << "Issuing io";
      if (state->stats != nullptr) {
        state->stats->n_4k++;
        state->stats->n_ios++;
	state->io_timer.reset();
      }
      assert(state->frontier.size() != 0);
      bool read_issued = parent->state_issue_next_io_batch(state, ctx);
      assert(read_issued);
    } else if (s == SearchExecutionState::TOP_CAND_NODE_OFF_SERVER) {
      assert(parent->num_partitions > 1);
      parent->send_state(state);
      delete state;
      number_concurrent_queries--;
      // need to send this bitch

      // TODO: serialize the state and send that bitch
      // complication: how to make it work both locally and distributed/
      // make the iothread unique ptr, if local then nullptr
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
