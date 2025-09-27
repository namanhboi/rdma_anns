#pragma once

#define MAX_IO_DEPTH 128

#include <fcntl.h>
#include <unistd.h>
#include "query_buf.h"

#include <malloc.h>
#include <cstdio>

class AlignedFileReader {
 public:
  // returns the thread-specific io ring.
  // If not constructed, it will register the thread (using the flag) and return the context.
  // For io_uring reader, the flag is used to set up the ring (e.g., IORING_SETUP_SQPOLL).
  // For PipeSearch, we use IO_RING_SETUP_POLL to enable polling.
  // For all the other algorithms, we use 0 to disable polling.
  virtual void *get_ctx(int flag = 0) = 0;

  virtual ~AlignedFileReader() {};

  // Open & close ops
  // Blocking calls
  virtual void open(const std::string &fname, bool enable_writes, bool enable_create) = 0;
  virtual void close() = 0;

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  virtual void read(std::vector<IORequest> &read_reqs, void *ctx, bool async = false) = 0;
  virtual void write(std::vector<IORequest> &write_reqs, void *ctx, bool async = false) = 0;
  virtual void read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) = 0;
  virtual void write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) = 0;

  virtual void send_io(IORequest &reqs, void *ctx, bool write) = 0;
  virtual void send_io(std::vector<IORequest> &reqs, void *ctx, bool write) = 0;

  virtual int poll(void *ctx) = 0;
  virtual void poll_all(void *ctx) = 0;
  virtual void poll_wait(void *ctx) = 0;

 protected:
  // register thread-id for a context
  virtual void register_thread(int flag = 0) = 0;
  // de-register thread-id for a context
  virtual void deregister_thread() = 0;
  virtual void deregister_all_threads() = 0;
};
