#pragma once

#include "aligned_file_reader.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t file_sz;
  FileHandle file_desc;
  void *bad_ctx = nullptr;

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  void *get_ctx(int flag = 0);

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname, bool enable_writes, bool enable_create);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<IORequest> &read_reqs, void *ctx, bool async = false);
  void write(std::vector<IORequest> &write_reqs, void *ctx, bool async = false);
  void read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx);
  void write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx);

  /**
     search_state is io request will be null which is a signal to stop the main
     loop of search thread
  */
  void send_noop(IORequest *req, void *ctx);
  void send_io(IORequest &reqs, void *ctx, bool write);
  void send_io(std::vector<IORequest> &reqs, void *ctx, bool write);
  int poll(void *ctx);
  void poll_all(void *ctx);
  IORequest *poll_wait(void *ctx);

  // register thread-id for a context
  void register_thread(int flag = 0);

  // de-register thread-id for a context
  void deregister_thread();

  void deregister_all_threads();
};

