#include "ssd_partition_index.h"

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::IOSubmissionThread::IOSubmissionThread(
    SSDPartitionIndex *parent, uint32_t num_search_threads)
    : parent(parent), num_search_threads(num_search_threads) {
  if (!parent)
    throw std::invalid_argument("parent cannot be null");
  if (num_search_threads > MAX_SEARCH_THREADS)
    throw std::invalid_argument("too many search threads");
  for (auto i = 0; i < num_search_threads; i++) {
    thread_io_requests[i].reserve(max_requests);
  }

  thread_io_ctx.fill(nullptr);
  LOG(INFO) << "created the io submission thread";
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::IOSubmissionThread::start() {
  running = true;
  this->real_thread = std::thread(
      &SSDPartitionIndex<T, TagT>::IOSubmissionThread::main_loop, this);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::IOSubmissionThread::signal_stop() {
  running = false;
  this->concurrent_io_req_queue.enqueue(
      {nullptr, std::numeric_limits<uint64_t>::max(), {}});
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::IOSubmissionThread::join() {
  if (this->real_thread.joinable())
    this->real_thread.join();
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::IOSubmissionThread::push_io_request(
    thread_io_req_t thread_io_req) {
  this->concurrent_io_req_queue.enqueue(thread_io_req);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::IOSubmissionThread::submit_requests(
    std::vector<IORequest> &io_requests, void *ctx) {
  parent->reader->send_io(io_requests, ctx, false);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::IOSubmissionThread::main_loop() {
  std::array<thread_io_req_t, max_requests> requests;
  while (running) {
    size_t num_dequeued_requests =
        this->concurrent_io_req_queue.wait_dequeue_bulk(requests.begin(),
                                                        max_requests);
    for (size_t i = 0; i < num_dequeued_requests; i++) {
      if (requests[i].thread_id == std::numeric_limits<uint64_t>::max() &&
          requests[i].ctx == nullptr) {
        continue;
      }

      if (requests[i].thread_id >= MAX_SEARCH_THREADS) {
        throw std::runtime_error("invalid thread id " +
                                 std::to_string(requests[i].thread_id));
      }

      thread_io_requests[requests[i].thread_id].emplace_back(
          requests[i].io_request);

      if (thread_io_ctx[requests[i].thread_id] == nullptr) {
        thread_io_ctx[requests[i].thread_id] = requests[i].ctx;
      }
    }
    for (auto thread_id = 0; thread_id < num_search_threads; thread_id++) {
      if (!thread_io_requests[thread_id].empty()) {
        submit_requests(thread_io_requests[thread_id],
                        thread_io_ctx[thread_id]);
        thread_io_requests[thread_id].clear();
      }
    }
  }
}

