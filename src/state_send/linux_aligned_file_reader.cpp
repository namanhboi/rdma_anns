#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <thread>
#define MAX_EVENTS 4096
namespace {
constexpr uint64_t kNoUserData = 0;
void execute_io(void *context, int fd, std::vector<IORequest> &reqs,
                uint64_t n_retries = 0, bool write = false) {
  io_uring *ring = (io_uring *)context;
  while (true) {
    for (uint64_t j = 0; j < reqs.size(); j++) {
      auto sqe = io_uring_get_sqe(ring);
      sqe->user_data = kNoUserData;
      if (write) {
        io_uring_prep_write(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
      } else {
        io_uring_prep_read(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
      }
    }
    io_uring_submit(ring);

    io_uring_cqe *cqe = nullptr;
    bool fail = false;
    for (uint64_t j = 0; j < reqs.size(); j++) {
      int ret = 0;
      do {
        ret = io_uring_wait_cqe(ring, &cqe);
      } while (ret == -EINTR);

      if (ret < 0 || cqe->res < 0) {
        fail = true;
        LOG(ERROR) << "Failed " << strerror(-ret) << " " << ring << " " << j
                   << " " << reqs[j].buf << " " << reqs[j].len << " "
                   << reqs[j].offset;
        break; // CQE broken.
      }
      io_uring_cqe_seen(ring, cqe);
    }
    if (!fail) { // repeat until no fails.
      break;
    }
  }
}
} // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() { this->file_desc = -1; }

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                  << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

void *LinuxAlignedFileReader::get_ctx(int flag) {
  std::unique_lock<std::mutex> lk(ctx_mut);
  // perform checks only in DEBUG mode
  if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end()) {
    std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
    return this->bad_ctx;
  } else {
    return ctx_map[std::this_thread::get_id()];
  }
}

void LinuxAlignedFileReader::register_thread(int flag) {
  auto my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> l(ctx_mut);
  if (ctx_map.find(my_id) != ctx_map.end()){
    std::cerr << "multiple calls to register_thread from the same thread" << std::endl;
    return;
  }
  io_uring* ctx = new io_uring();
  io_uring_queue_init(MAX_EVENTS, ctx, flag);
  ctx_map[my_id] = ctx;
  ctx_submission_mutex_map[reinterpret_cast<void *>(ctx)] = std::make_unique<std::mutex>();
  assert(ctx_map.size() == ctx_submission_mutex_map.size());
  l.unlock();
}


void LinuxAlignedFileReader::deregister_thread() {
  auto my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  assert(ctx_map.find(my_id) != ctx_map.end());
  lk.unlock();
  io_uring *ctx = reinterpret_cast<io_uring *>(this->get_ctx());
  
  io_uring_queue_exit(ctx);

  lk.lock();
  ctx_map.erase(my_id);
  ctx_submission_mutex_map.erase(reinterpret_cast<void*>(ctx));
  if (ctx_map.size() != ctx_submission_mutex_map.size()) {
    throw std::runtime_error("map is different sizes " +
                             std::to_string(ctx_map.size()) +
                             std::to_string(ctx_submission_mutex_map.size()));
  }
  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
  lk.unlock();
}

void LinuxAlignedFileReader::deregister_all_threads() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
    io_uring* ctx= x->second;
    io_uring_queue_exit(ctx);
  }
  ctx_map.clear();
  ctx_submission_mutex_map.clear();
}

void LinuxAlignedFileReader::open(const std::string &fname,
                                  bool enable_writes = false,
                                  bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags, 0644);
  // error checks
  assert(this->file_desc != -1);
  //  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx,
                                  bool async) {
  std::unique_lock<std::mutex> l(ctx_mut);
  std::unique_ptr<std::mutex> &ctx_sub_mut = ctx_submission_mutex_map[ctx];
  l.unlock();
  std::unique_lock<std::mutex> lock(*ctx_sub_mut);
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs,
                                   void *ctx, bool async) {
  std::unique_lock<std::mutex> l(ctx_mut);
  std::unique_ptr<std::mutex> &ctx_sub_mut = ctx_submission_mutex_map[ctx];
  l.unlock();
  std::unique_lock<std::mutex> lock(*ctx_sub_mut);  
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs,
                                     void *ctx) {
  std::unique_lock<std::mutex> l(ctx_mut);
  std::unique_ptr<std::mutex> &ctx_sub_mut = ctx_submission_mutex_map[ctx];
  l.unlock();
  std::unique_lock<std::mutex> lock(*ctx_sub_mut);  
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd,
                                      std::vector<IORequest> &write_reqs,
                                      void *ctx) {
  std::unique_lock<std::mutex> l(ctx_mut);
  std::unique_ptr<std::mutex> &ctx_sub_mut = ctx_submission_mutex_map[ctx];
  l.unlock();
  std::unique_lock<std::mutex> lock(*ctx_sub_mut);
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
  std::unique_lock<std::mutex> l(ctx_mut);
  std::unique_ptr<std::mutex> &ctx_sub_mut = ctx_submission_mutex_map[ctx];
  l.unlock();
  std::unique_lock<std::mutex> lock(*ctx_sub_mut);
  io_uring *ring = (io_uring *)ctx;
  auto sqe = io_uring_get_sqe(ring);
  req.finished = false;
  sqe->user_data = (uint64_t)&req;
  if (write) {
    io_uring_prep_write(sqe, this->file_desc, req.buf, req.len, req.offset);
  } else {
    io_uring_prep_read(sqe, this->file_desc, req.buf, req.len, req.offset);
  }
  io_uring_submit(ring);
}

void LinuxAlignedFileReader::send_noop(IORequest *req,void *ctx) {
  std::unique_lock<std::mutex> l(ctx_mut);
  std::unique_ptr<std::mutex> &ctx_sub_mut = ctx_submission_mutex_map[ctx];
  l.unlock();
  std::unique_lock<std::mutex> lock(*ctx_sub_mut);
  io_uring *ring = (io_uring *)ctx;
  auto sqe = io_uring_get_sqe(ring);
  req->finished = false;
  sqe->user_data = (uint64_t)req;
  io_uring_prep_nop(sqe);
  io_uring_submit(ring);
}


void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx,
                                     bool write) {
  std::unique_lock<std::mutex> l(ctx_mut);
  std::unique_ptr<std::mutex> &ctx_sub_mut = ctx_submission_mutex_map[ctx];
  l.unlock();
  std::unique_lock<std::mutex> lock(*ctx_sub_mut);
  io_uring *ring = (io_uring *)ctx;
  for (uint64_t j = 0; j < reqs.size(); j++) {
    auto sqe = io_uring_get_sqe(ring);
    reqs[j].finished = false;
    sqe->user_data = (uint64_t)&reqs[j];
    if (write) {
      io_uring_prep_write(sqe, this->file_desc, reqs[j].buf, reqs[j].len,
                          reqs[j].offset);
    } else {
      io_uring_prep_read(sqe, this->file_desc, reqs[j].buf, reqs[j].len,
                         reqs[j].offset);
    }
  }
  int ret = io_uring_submit(ring);
  if (ret < 0) {
    LOG(INFO) << "Submit failed: " << strerror(-ret);
  }
}

IORequest * LinuxAlignedFileReader::poll(void *ctx) {
  io_uring *ring = (io_uring *)ctx;
  io_uring_cqe *cqe = nullptr;
  int ret = io_uring_peek_cqe(ring, &cqe);
  if (ret < 0) {
    return nullptr; // not finished yet.
  }
  if (cqe->res < 0) {
    LOG(ERROR) << "Failed " << strerror(-cqe->res);
  }
  IORequest *req = (IORequest *)cqe->user_data;
  if (req != nullptr) {
    req->finished = true;
  }
  io_uring_cqe_seen(ring, cqe);
  return req;
}

void LinuxAlignedFileReader::poll_all(void *ctx) {
  io_uring *ring = (io_uring *)ctx;
  static __thread io_uring_cqe *cqes[MAX_EVENTS];
  int ret = io_uring_peek_batch_cqe(ring, cqes, MAX_EVENTS);
  if (ret < 0) {
    return; // not finished yet.
  }
  for (int i = 0; i < ret; i++) {
    if (cqes[i]->res < 0) {
      LOG(ERROR) << "Failed " << strerror(-cqes[i]->res);
    }
    IORequest *req = (IORequest *)cqes[i]->user_data;
    if (req != nullptr) {
      req->finished = true;
    }
    io_uring_cqe_seen(ring, cqes[i]);
  }
}

IORequest* LinuxAlignedFileReader::poll_wait(void *ctx) {
  io_uring *ring = (io_uring *)ctx;
  io_uring_cqe *cqe = nullptr;
  int ret = 0;
  do {
    ret = io_uring_wait_cqe(ring, &cqe);
  } while (ret == -EINTR);
  if (ret < 0 || cqe->res < 0) {
    LOG(ERROR) << "Failed " << strerror(-cqe->res);
  }
  IORequest *req = (IORequest *)cqe->user_data;
  if (req != nullptr) {
    req->finished = true;
  }
  io_uring_cqe_seen(ring, cqe);
  return req;
}
