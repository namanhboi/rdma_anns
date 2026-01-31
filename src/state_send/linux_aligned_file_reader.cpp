#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>

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

      if (ret < 0) {
        fail = true;
        LOG(ERROR) << "io_uring_wait_cqe failed: " << strerror(-ret)
                   << " ring=" << ring << " request=" << j;
        break;
      }
      
      if (cqe->res < 0) {
        fail = true;
        LOG(ERROR) << "I/O operation failed: " << strerror(-cqe->res)
                   << " ring=" << ring << " request=" << j
                   << " buf=" << reqs[j].buf 
                   << " len=" << reqs[j].len 
                   << " offset=" << reqs[j].offset;
        break;
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

namespace ioctx {
  static thread_local io_uring *ring = nullptr;
};

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::ring == nullptr)) {
    register_thread(flag);
  }
  return ioctx::ring;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::ring == nullptr) {
    ioctx::ring = new io_uring();
    io_uring_queue_init(MAX_EVENTS, ioctx::ring, flag);
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  io_uring_queue_exit(ioctx::ring);
  delete ioctx::ring;
  ioctx::ring = nullptr;
}

// not sure if this is correct?
void LinuxAlignedFileReader::deregister_all_threads() { return; }

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
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs,
                                   void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs,
                                     void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd,
                                      std::vector<IORequest> &write_reqs,
                                      void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
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
  io_uring *ring = (io_uring *)ctx;
  auto sqe = io_uring_get_sqe(ring);
  req->finished = false;
  sqe->user_data = (uint64_t)req;
  io_uring_prep_nop(sqe);
  io_uring_submit(ring);
}


void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx,
                                     bool write) {
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
