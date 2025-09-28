#include "query_buf.h"
#include "ssd_partition_index.h"

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::SearchThread::SearchThread(
    SSDPartitionIndex *parent, uint64_t thread_id)
: parent(parent), thread_id(thread_id) {}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::start() {
  running = true;
  real_thread =
    std::thread(&SSDPartitionIndex<T, TagT>::SearchThread::main_loop, this);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::main_loop() {
  this->parent->reader->register_thread();
  ctx = this->parent->reader->get_ctx();
  if (ctx == nullptr) {
    throw std::runtime_error("ctx given by get_ctx is nullptr");
  }
  int num_requests = 0;
  while (running) {
    IORequest *req = this->parent->reader->poll_wait(ctx);
    num_requests++;
    if (req->search_state == nullptr) {
      delete req;
      break;
    }
    SearchState *state = reinterpret_cast<SearchState*>(req->search_state);
    SearchExecutionState s = state->explore_frontier();
    if (s == SearchExecutionState::FINISHED) {
      // TODO:send results to client, delete the state
      this->parent->notify_client_local(state);
    } else if (s == SearchExecutionState::TOP_CAND_NODE_ON_SERVER) {
      // TODO:issue io
      state->issue_next_io_batch(ctx);
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
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::SearchThread::join() {
  if (real_thread.joinable()) {
    assert(running == false);
    real_thread.join();
  }
}


template class SSDPartitionIndex<float>::SearchThread;
template class SSDPartitionIndex<uint8_t>::SearchThread;
template class SSDPartitionIndex<int8_t>::SearchThread;
