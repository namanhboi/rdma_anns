#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "VerbsEP.hpp"
#include "concurrentqueue.h"
#include "ring.hpp"
#include "reverse_ring_communicators.hpp"
#include "consts.hpp"
#include "utils.hpp"
#include "reverse_ring.hpp"


using recv_handler_t = std::function<void(const char *, size_t)>;
/* Error Macro*/
#define rdma_error(msg, args...) do {                                   \
fprintf(stderr, "%s : %d : ERROR : " msg, __FILE__, __LINE__, ## args); \
}while(0);

/**
   assign every server a unique, sortable identifier. Then, you establish a hard
   rule: A server only actively initiates a connection to peers with a higher
   ID. It only passively listens for peers with a lower ID.

   Wait first, connect second.
 */

/**
   Uses librdmacm to manage connections. Each connected client will have 2
   uni-directional channels per client  using write ring buffer with inlined
   bell and w/o. 1 channel used for actual data sending, another is for acks.
 */
class RDMAManager {
  struct rdma_event_channel *_ec;

  // id is only used for connection establishment, not for data exchange
  struct rdma_cm_id* _listen_id;
  struct ibv_pd* _pd;

  std::vector<union ibv_gid > valid_gids;
  uint32_t udqpn;

  std::vector<VerbsEP *> eps; // contain client endpoints with their qp and cp

  std::vector<connect_info *>
    connect_buffers; // contain init messages from client when they connect

  std::vector<std::unique_ptr<CircularReverseReceiver>>
    receivers; // for each client, we need a ring buffer for receiving data
  std::vector<std::unique_ptr<CircularConnectionReverse>>
    senders; // for each client, we need a ring buffer for sending data

  std::vector<struct ibv_mr *> local_mrs; // used for misc stuff, tracked because we need to free it
  std::vector<char*> local_mems;

  // ip address and ports of all servers
  std::vector<std::pair<std::string, std::string>> address_list;
  uint64_t my_id;
  size_t num_servers; // including your own

private:
  recv_handler_t handler;
  std::unique_ptr<std::atomic<uint32_t>[]> pending_freed_bytes;
  std::vector<uint64_t> pending_ack_ids;
  std::vector<moodycamel::ConcurrentQueue<Region *>> outgoing_queues;

private:
  struct ibv_mr *preallocated_region_mr;
  char *preallocated_region_addr;
private:
  // thread resources for send/recv thread
  std::thread send_thread;
  std::thread recv_thread;

  std::atomic<bool> running = false;
private:
  std::pair<std::string, std::string> parse_ip_port(const std::string& ip_port);
  std::vector<std::vector<Region *>> in_flight_regions;

public:

  RDMAManager(){ // fake server
    _pd = NULL;
  }

  // setup local variables and call bind_server
  RDMAManager(uint64_t my_id, const std::vector<std::string> &addresses);

  ~RDMAManager();
  void register_receive_handler(recv_handler_t handler) {
    this->handler = handler;
  }

  void bind_server(const char *ip, const char *port);


  // passive for all servers with id below my_id, active connect to all servers
  // with servers with id above my_id
  void connect_to_all_servers();

  struct ibv_pd* getPD() const { return this->_pd; }

  uint8_t get_ibport() const { return _listen_id->port_num; }

  uint32_t get_mtu(){
    struct ibv_port_attr port_info = {};
    if (ibv_query_port(_pd->context, get_ibport(), &port_info)) {
      fprintf(stderr, "Unable to query port info for port %d\n", get_ibport());
      return 1;
    }
    int mtu = 1 << (port_info.active_mtu + 7);
    printf("The Maximum payload size is %d\n",mtu);
    printf("The port lid is 0 for roce. Let's check: %d\n",port_info.lid);
    return (uint32_t)mtu;
  }

  int get_pkeyindex() const {
    struct ibv_port_attr port_info = {};
    if (ibv_query_port(_pd->context, get_ibport(), &port_info)) {
      fprintf(stderr, "Unable to query port info for port %d\n", get_ibport());
      return 0;
    }
    if(port_info.link_layer == IBV_LINK_LAYER_ETHERNET){
      printf("Link is ethernet. table size %u\n",port_info.pkey_tbl_len);
      return 0;
    }

    __be16 pkey = _listen_id->route.addr.addr.ibaddr.pkey;
    printf("pkey is %u\n", pkey );
    int index = ibv_get_pkey_index(_pd->context, get_ibport(), pkey);
    printf("pkeyindex is %u\n", index);
    return index;
  }

  static int null_gid(union ibv_gid *gid) {
    return !(gid->raw[8] | gid->raw[9] | gid->raw[10] | gid->raw[11] |
             gid->raw[12] | gid->raw[13] | gid->raw[14] | gid->raw[15]);
  }

  void query_all_gids() {
    struct ibv_port_attr port_info = {};
    if (ibv_query_port(_pd->context, get_ibport(), &port_info)) {
      fprintf(stderr, "Unable to query port info for port %d\n", get_ibport());
    }

    for(int i=0; i < port_info.gid_tbl_len; i++){
      union ibv_gid   gid;
      if(ibv_query_gid(_pd->context, get_ibport(), i, &gid)){
        fprintf(stderr, "Failed to query GID[%d]\n",i);
        continue;
      }
      if (!null_gid(&gid)){
        valid_gids.push_back(gid);
        printf("GID[%d].  subnet %llu\n\t inteface %llu\n",i,gid.global.subnet_prefix, gid.global.interface_id);
      }
    }
  }

  std::pair<struct rdma_cm_id *, void *> wait_for_connect_request();

  std::pair<VerbsEP *, connect_info *>
    get_client_ep_and_info(struct ibv_qp_init_attr attr, void *my_info,
                           uint32_t recv_batch, bool recv_with_data);

  void receive_connections(int starting_client_id, int num_clients,int max_send_size, int max_recv_size);


  std::pair<VerbsEP *, connect_info *>
    get_server_ep_and_info(int server_id, struct ibv_qp_init_attr attr,
                           void *my_info, uint32_t recv_batch,
                           bool recv_with_data);

  // return the newly created id and the connect_buffer pointer for the client connection
  struct rdma_cm_id *send_connect_request(const char *ip, const char *port);

  void send_connections(int starting_server_id, int num_servers,
                        int max_send_size, int max_recv_size);




  // the actual sending and receiving operations
  void enqueue_region_to_send(uint64_t peer_id, Region *r);


  //// [START] thread stuff ////
  void start_send_recv_threads();
  void shutdown_threads();
  void send_loop();
  void recv_loop();
  //// [END] thread stuff  ////

  std::pair<char *, uint32_t>
  get_preallocated_region_ptr_lkey(size_t block_size_per_element,
                                   size_t num_elements);


  // need to disconnect and destroy id + all mr and qp + cq
  void cleanup();
};
