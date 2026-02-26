#pragma once
#define ZMQ_BUILD_DRAFT_API 1
#include <cstdint>
#include <functional>
#include <string_view>
#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>
#include <zmq.h>
#include <iostream>
#include <atomic>
#include <thread>
#include "../state_send/utils.h"
#include "../state_send/types.h"
// for convenience

using json = nlohmann::json;


constexpr uint32_t max_num_servers = 128;


struct Region {
  static constexpr size_t MAX_BYTES_REGION = 80000;
  static constexpr size_t MAX_PRE_ALLOC_ELEMENTS =10000;
  
  char *addr; // address to whatever is sent must be allocated with new []
  uint32_t length;

  // for rdma, specifically this : https://arxiv.org/pdf/2212.09134
  uint64_t context;
  uint32_t lkey;


  // passed into delete addr, used to detemine whether to allow zmq to delete
  // data or not
  // bool self_manage_data = false;

  void *prealloc_queue;


  // by default, we don't manage our memeory
  Region() : prealloc_queue(nullptr) {}

  static void reset(Region *r) { r->length = 0; }

  // hint used to pass in the address to prealloc queue
  static void assign_addr(Region *r, char *prealloacted_addr, void *hint) {
    r->addr = prealloacted_addr;
    r->prealloc_queue = hint;
  }

  /**
   * used for zmq zmq_msg_init_data, which will use this function to free addr,
   so no need to manually free it if use zmq. Hint here represents the pointer
   to the Region r you're freeing.
   If the region is self managed, then should pass in pointer to it. Else pass
   in nullptr, which will trigger a delete[] call to free addr
   */
  static void delete_addr(void *data, void *hint) {
    // reinterpretting the hint as a bool representing self_managed_data
    if (hint == nullptr) {
      delete[] reinterpret_cast<char *>(data);
    } else {
      Region *r = reinterpret_cast<Region *>(hint);
      if (unlikely(r->prealloc_queue == nullptr)) {
        throw std::runtime_error(
				 "prealloc queue is nullptr even tho ptr to Region is not nullptr");
      }
      PreallocatedQueue<Region> *q =
        reinterpret_cast<PreallocatedQueue<Region> *>(r->prealloc_queue);
      q->free(r);
    }
  }
};


using recv_handler_t = std::function<void(const char *, size_t)>;
/**
   this class handlers inter-server communication by wrapping some existing
   communication library via tcp/rdma. It needs to work for both
   client-server and server-server connection.

   Clients need to send the query to the server and servers needs to exchange
   states.

   This class may parse a json file to get the list of all server ids + their
   address/(rdma equivalent). This is unecessary for cascade wrapper.
*/
class P2PCommunicator {
protected:
public:
  virtual void send_to_peer(uint64_t peer_id, Region *r) =0;
  virtual void recv_loop() = 0;
  virtual void register_receive_handler(recv_handler_t handler) = 0;
  // starts running recv loop on a different thread
  virtual void start_recv_thread() = 0;
  virtual void stop_recv_thread() = 0;
  virtual uint64_t get_my_id() = 0; 
  /** including our own */
  virtual uint64_t get_num_peers() =0;

  /** doesn't include your own */
  virtual std::vector<uint64_t> get_other_peer_ids() = 0;
  
};


/**
   zeromq wrapper. Uses ZMQ_PEER. Binds to ip/port given in json file and
   zmq_connect_peer to all others. Included in the json file is also the id of
   its partition of the graph + the ids of all other partitions.
   Used for both client<->server and server<->server communication
*/
class ZMQP2PCommunicator : virtual public P2PCommunicator {
private:
  uint64_t my_id;
protected:
  void* ctx;
  void *sock;

  // result of zmq_connect_peer is the routing id of the other servers.
  // zmq_msg_send uses routing id to send stuff.
  std::unordered_map<uint64_t, uint32_t> peer_to_routing_id;

  std::unordered_map<uint64_t, std::string> peer_id_to_address;

  std::string my_address;
  recv_handler_t recv_handler;
  std::atomic<bool> running{false};
  std::thread real_thread;
  void parse_config(const std::string &config_path);
  void bind_and_connect_peers();
public:
  void send_to_peer(uint64_t peer_id, Region *r) override;
  void recv_loop() override;
  void register_receive_handler(recv_handler_t handler) override;
  void start_recv_thread() override;
  void stop_recv_thread() override;
  uint64_t get_my_id() override;
  /** including our own */
  uint64_t get_num_peers() override;

  /** doesn't include your own */
  std::vector<uint64_t> get_other_peer_ids() override;

  ZMQP2PCommunicator(uint64_t my_id, const std::string &config_path);

  /**
     includes ip addreeses of all clients and servers.
     client peer id should correspond to the last ip address
  */
  ZMQP2PCommunicator(uint64_t my_id, const std::vector<std::string> &peer_ips);
  
  ~ZMQP2PCommunicator();
};




