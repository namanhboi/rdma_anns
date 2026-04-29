#include "rdma_com/rdma_manager.hpp"
#include "rdma_com/consts.hpp"
#include "rdma_com/reverse_ring_communicators.hpp"
#include <infiniband/verbs.h>
#include <chrono>
static void drain_cq(struct ibv_cq *cq) {
  if (!cq) return;

  struct ibv_wc wc[32];
  while (true) {
    int n = ibv_poll_cq(cq, 32, wc);
    if (n <= 0) break;
  }
}

inline uint32_t local_log2(const uint32_t x) {
  uint32_t y;
  asm ( "\tbsr %1, %0\n"
          : "=r"(y)
          : "r" (x)
   );
  return y;
}


std::pair<std::string, std::string>
RDMAManager::parse_ip_port(const std::string &ip_port) {
  std::string addr = ip_port;

  // Strip protocol prefix if present (e.g., "tcp://")
  size_t proto_pos = addr.find("://");
  if (proto_pos != std::string::npos) {
    addr = addr.substr(proto_pos + 3);
  }

  // Split on the LAST ':' to avoid issues
  size_t colon_pos = addr.rfind(':');
  if (colon_pos == std::string::npos) {
    throw std::runtime_error("Invalid address format: " + ip_port);
  }

  std::string ip = addr.substr(0, colon_pos);
  std::string port = addr.substr(colon_pos + 1);

  return {ip, port};
}
RDMAManager::RDMAManager(uint64_t my_id, const std::vector<std::string> &addresses)
: my_id(my_id), num_servers(addresses.size()) {
  // throw std::runtime_error("Need to address how to handle threads and such");
  std::cout << "Binding to " << addresses[my_id] << std::endl;
  for (const std::string &ip_port : addresses) {
    address_list.push_back(parse_ip_port(ip_port));
  }

  std::string my_ip = address_list[my_id].first;
  std::string my_port = address_list[my_id].second;
  bind_server(my_ip.c_str(), my_port.c_str());
  outgoing_queues.resize(num_servers);
  std::cout << "Done binding to " << addresses[my_id] << std::endl;
}

void RDMAManager::bind_server(const char *ip, const char *port) {
  this->_ec = rdma_create_event_channel();
  if (!this->_ec) {
    perror("failed to create event channel\n");
    exit(1);
  }

  if (rdma_create_id(this->_ec, &this->_listen_id, NULL, RDMA_PS_TCP)) {
    printf("failed to create listen id\n");
    exit(1);
  }

  rdma_addrinfo *addrinfo;
  rdma_addrinfo hints;

  memset(&hints, 0, sizeof(hints));
  hints.ai_port_space = RDMA_PS_TCP;
  hints.ai_flags = RAI_PASSIVE;

  if (rdma_getaddrinfo(ip, port, &hints, &addrinfo)) {
    printf("failed to get addr info\n");
    std::cout << std::string(ip) << ":" << std::string(port) << std::endl;
    exit(1);
  }

  if (rdma_bind_addr(this->_listen_id, addrinfo->ai_src_addr)) {
    printf("failed to bind addr\n");
    exit(1);
  }

  // Start listening
  if (rdma_listen(this->_listen_id, 0)) {
    printf("failed to start listening\n");
    exit(1);
  }

  if (_listen_id->pd) {
    _pd = _listen_id->pd;
  } else {
    _pd = ibv_alloc_pd(_listen_id->verbs);
  }
}

// old reverse ring buffer
// void RDMAManager::connect_to_all_servers() {
//   eps.resize(num_servers);
//   eps[my_id] = nullptr;

//   receivers.resize(num_servers);
//   receivers[my_id] = nullptr;

//   senders.resize(num_servers);
//   senders[my_id] = nullptr;

//   connect_buffers.resize(num_servers);
//   connect_buffers[my_id] = nullptr;

//   in_flight_regions.resize(num_servers);
//   pending_ack_ids.resize(num_servers);

//   // Replace pending_freed_bytes.resize(num_servers) with this:
//   pending_freed_bytes.reset(new std::atomic<uint32_t>[num_servers]);
//   for (size_t i = 0; i < num_servers; i++) {
//     pending_freed_bytes[i].store(0, std::memory_order_relaxed);
//   }
//   if (my_id != 0) {
//     receive_connections(0, my_id, MAX_SEND_RECV_WR, MAX_SEND_RECV_WR);
//   }
//   if (my_id != num_servers - 1) {
//     send_connections(my_id + 1, num_servers - my_id - 1, MAX_SEND_RECV_WR,
//                      MAX_SEND_RECV_WR);
//   }
// }

void RDMAManager::connect_to_all_servers() {
  eps.resize(num_servers);
  receivers.resize(num_servers);
  senders.resize(num_servers);
  connect_buffers.resize(num_servers);
  in_flight_regions.resize(num_servers);
  pending_ack_ids.resize(num_servers);

  pending_freed_bytes.reset(new std::atomic<uint32_t>[num_servers]);
  incoming_credits.reset(new std::atomic<uint32_t>[num_servers]); // ADDED

  for (size_t i = 0; i < num_servers; i++) {
    pending_freed_bytes[i].store(0, std::memory_order_relaxed);
    incoming_credits[i].store(0, std::memory_order_relaxed); // ADDED
  }

  if (my_id != 0) {
    receive_connections(0, my_id, MAX_SEND_RECV_WR, MAX_SEND_RECV_WR);
  }
  if (my_id != num_servers - 1) {
    send_connections(my_id + 1, num_servers - my_id - 1, MAX_SEND_RECV_WR,
                     MAX_SEND_RECV_WR);
  }
}

void RDMAManager::receive_connections(int starting_client_id, int num_clients,
                                      int max_send_size, int max_recv_size) {
  for (uint32_t i = 0; i < static_cast<uint32_t>(num_clients); i++) {
    // prepare_qp() allocates CQs. Keep it inside this loop so each QP has
    // private send/recv CQs; the per-peer SendConnection/ReceiveReceiver code
    // assumes completions cannot be mixed with another peer's QP.
    struct ibv_qp_init_attr attr =
        prepare_qp(this->getPD(), max_send_size, max_recv_size, false);
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;

    connect_info info = {};
    info.code = 4;
    info.server_id = my_id;

    VerbsEP *ep;
    connect_info *connect_buffer;
    std::tie(ep, connect_buffer) =
        get_client_ep_and_info(attr, &info, 16, true);
    uint64_t client_id = connect_buffer->server_id;

    size_t buffer_size = Region::MAX_BYTES_REGION;
    size_t local_recv_size = buffer_size * MAX_SEND_RECV_WR;

    // aligned_alloc requires the allocation size to be a multiple of alignment.
    size_t alloc_size = ((local_recv_size + 4095) / 4096) * 4096;

    char *recv_mem = (char *)aligned_alloc(4096, alloc_size);
    if (!recv_mem) {
      perror("aligned_alloc recv_mem");
      exit(1);
    }

    memset(recv_mem, 0, alloc_size);

    struct ibv_mr *recv_mr =
      ibv_reg_mr(this->getPD(),
                 recv_mem,
                 alloc_size,
                 IBV_ACCESS_LOCAL_WRITE);

    if (!recv_mr) {
      perror("ibv_reg_mr recv_mr");
      free(recv_mem);
      exit(1);
    }

    local_mrs.push_back(recv_mr);
    local_mems.push_back(recv_mem);

    receivers[client_id] = std::make_unique<ReceiveReceiver>(
        ep, recv_mem, buffer_size, MAX_SEND_RECV_WR, recv_mr->lkey);
    senders[client_id] = std::make_unique<SendConnection>(ep);
    eps[client_id] = ep;
    connect_buffers[client_id] = connect_buffer;

    in_flight_regions[client_id] = std::vector<Region *>(MAX_SEND_RECV_WR, nullptr);
    pending_ack_ids[client_id] = 1;

    printf("RDMA passive peer=%lu qp=%p qp_num=%u send_cq=%p recv_cq=%p recv_mem=%p lkey=0x%x\n",
           client_id,
           ep->qp,
           ep->qp->qp_num,
           ep->qp->send_cq,
           ep->qp->recv_cq,
           recv_mem,
           recv_mr->lkey);
  }
}

void RDMAManager::send_connections(int starting_server_id,
                                   int num_servers,
                                   int max_send_size,
                                   int max_recv_size) {
  for (uint32_t i = 0; i < num_servers; i++) {
    int server_id = starting_server_id + i;

    connect_info info = {};
    info.code = 4;
    info.server_id = my_id;

    VerbsEP *ep = nullptr;
    connect_info *connect_buffer = nullptr;

    std::tie(ep, connect_buffer) =
        get_server_ep_and_info(server_id,
                               max_send_size,
                               max_recv_size,
                               &info,
                               16,
                               true);

    size_t buffer_size = Region::MAX_BYTES_REGION;
    size_t local_recv_size = buffer_size * MAX_SEND_RECV_WR;

    // aligned_alloc requires the allocation size to be a multiple of alignment.
    size_t alloc_size = ((local_recv_size + 4095) / 4096) * 4096;

    char *recv_mem = (char *)aligned_alloc(4096, alloc_size);
    if (!recv_mem) {
      perror("aligned_alloc recv_mem");
      exit(1);
    }

    memset(recv_mem, 0, alloc_size);

    struct ibv_mr *recv_mr =
      ibv_reg_mr(this->getPD(),
                 recv_mem,
                 alloc_size,
                 IBV_ACCESS_LOCAL_WRITE);

    if (!recv_mr) {
      perror("ibv_reg_mr recv_mr");
      free(recv_mem);
      exit(1);
    }
    local_mrs.push_back(recv_mr);
    local_mems.push_back(recv_mem);

    receivers[server_id] = std::make_unique<ReceiveReceiver>(
        ep,
        recv_mem,
        buffer_size,
        MAX_SEND_RECV_WR,
        recv_mr->lkey);

    senders[server_id] = std::make_unique<SendConnection>(ep);
    eps[server_id] = ep;
    connect_buffers[server_id] = connect_buffer;

    in_flight_regions[server_id] =
        std::vector<Region *>(MAX_SEND_RECV_WR, nullptr);

    pending_ack_ids[server_id] = 1;

    printf("RDMA active peer=%d qp=%p qp_num=%u send_cq=%p recv_cq=%p recv_mem=%p lkey=0x%x\n",
           server_id,
           ep->qp,
           ep->qp->qp_num,
           ep->qp->send_cq,
           ep->qp->recv_cq,
           recv_mem,
           recv_mr->lkey);
  }
}
struct rdma_cm_id *RDMAManager::send_connect_request(const char *ip,
                                                     const char *port) {
  struct rdma_cm_id *id = nullptr;
  struct rdma_addrinfo *addrinfo = nullptr;
  struct rdma_addrinfo hints;

  memset(&hints, 0, sizeof hints);
  hints.ai_port_space = RDMA_PS_TCP;

  if (rdma_getaddrinfo(ip, port, &hints, &addrinfo)) {
    return nullptr; // Just fail gracefully, let the loop retry
  }

  struct rdma_event_channel *channel = rdma_create_event_channel();
  if (!channel) {
    rdma_freeaddrinfo(addrinfo);
    return nullptr;
  }

  if (rdma_create_id(channel, &id, NULL, RDMA_PS_TCP)) {
    rdma_destroy_event_channel(channel);
    rdma_freeaddrinfo(addrinfo);
    return nullptr;
  }

  // 1. Resolve Address
  if (rdma_resolve_addr(id, NULL, addrinfo->ai_dst_addr, 2000)) {
    rdma_destroy_id(id);
    rdma_destroy_event_channel(channel);
    rdma_freeaddrinfo(addrinfo);
    return nullptr;
  }

  struct rdma_cm_event *event;
  if (rdma_get_cm_event(channel, &event)) {
    rdma_destroy_id(id);
    rdma_destroy_event_channel(channel);
    rdma_freeaddrinfo(addrinfo);
    return nullptr;
  }

  // CRITICAL FIX: Make sure it actually resolved!
  if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
    rdma_ack_cm_event(event);
    rdma_destroy_id(id);
    rdma_destroy_event_channel(channel);
    rdma_freeaddrinfo(addrinfo);
    return nullptr;
  }
  rdma_ack_cm_event(event);

  // 2. Resolve Route
  if (rdma_resolve_route(id, 2000)) {
    rdma_destroy_id(id);
    rdma_destroy_event_channel(channel);
    rdma_freeaddrinfo(addrinfo);
    return nullptr;
  }

  if (rdma_get_cm_event(channel, &event)) {
    rdma_destroy_id(id);
    rdma_destroy_event_channel(channel);
    rdma_freeaddrinfo(addrinfo);
    return nullptr;
  }

  // CRITICAL FIX: Make sure the route actually resolved!
  if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
    rdma_ack_cm_event(event);
    rdma_destroy_id(id);
    rdma_destroy_event_channel(channel);
    rdma_freeaddrinfo(addrinfo);
    return nullptr;
  }
  rdma_ack_cm_event(event);

  if (addrinfo) {
    rdma_freeaddrinfo(addrinfo);
  }

  return id; // Success! Return the valid ID to the caller loop
}


std::pair<struct rdma_cm_id *, void *> RDMAManager::wait_for_connect_request() {
  int has_pending = 0;
  rdma_cm_event *event;
  struct rdma_cm_id *id = nullptr;
  void *connect_buffer = nullptr;

  while (!has_pending) {
    // call blocks until an event shows up
    if (rdma_get_cm_event(this->_ec, &event)) {
      printf("Event poll unsuccesful, reason %d %s\n", errno, strerror(errno));
      exit(1);
    }

    switch (event->event) {
    case RDMA_CM_EVENT_CONNECT_REQUEST:
      has_pending = 1;
      id = event->id;

      if (event->param.conn.private_data_len) {
        printf("connect request had data with it: %u bytes\n",
               event->param.conn.private_data_len);
        connect_buffer = malloc(event->param.conn.private_data_len);
        memcpy(connect_buffer, event->param.conn.private_data,
               event->param.conn.private_data_len);
      }
      break;
    case RDMA_CM_EVENT_ESTABLISHED:
      printf("connection is esteblished for id %p\n", event->id);
      break;
    case RDMA_CM_EVENT_DISCONNECTED:
      printf("connection is disconnected for id %p\n", event->id);
      break;
    case RDMA_CM_EVENT_ADDR_ERROR:
    case RDMA_CM_EVENT_ROUTE_ERROR:
    case RDMA_CM_EVENT_CONNECT_ERROR:
    case RDMA_CM_EVENT_UNREACHABLE:
    case RDMA_CM_EVENT_REJECTED:
    case RDMA_CM_EVENT_ADDR_RESOLVED:
    case RDMA_CM_EVENT_ROUTE_RESOLVED:
      printf("[RDMAPassive]: Unexpected to receive;\n");
      break;
    case RDMA_CM_EVENT_DEVICE_REMOVAL:
      printf("[TODO]:  need to disconnect everything;\n");
      break;

    default:
      break;
    }
    rdma_ack_cm_event(event);
  }

  return std::make_pair(id, connect_buffer);
}

std::pair<VerbsEP *, connect_info *>
RDMAManager::get_client_ep_and_info(struct ibv_qp_init_attr attr, void *my_info,
                                    uint32_t recv_batch, bool recv_with_data) {
  struct ibv_pd *pd = this->getPD();
  struct rdma_cm_id *id;
  void *buf = nullptr;

  std::tie(id, buf) = this->wait_for_connect_request();

  uint32_t max_recv_size = attr.cap.max_recv_wr;
  if (attr.srq)
    attr.cap.max_recv_wr = 0;

  if (rdma_create_qp(id, pd, &attr)) {
    perror("rdma_create_qp");
    exit(1);
  }

  connect_info *connect_buffer = (connect_info *)buf;
  id->context = (void *)connect_buffer->server_id;

  attr.cap.max_recv_wr = max_recv_size;

  struct rdma_conn_param conn_param;
  memset(&conn_param, 0, sizeof(conn_param));
  conn_param.responder_resources = 16;
  conn_param.initiator_depth = 16;
  conn_param.retry_count = 3;
  conn_param.rnr_retry_count = 7;
  conn_param.private_data = my_info;
  conn_param.private_data_len = sizeof(connect_info); // Safe limit

  if (rdma_accept(id, &conn_param)) {
    printf(" failed to accept\n");
    exit(1);
  }
  printf("Accepted one\n");

  return std::make_pair(new VerbsEP(id, attr, recv_batch, recv_with_data),
                        (connect_info *)buf);
}

// void RDMAManager::receive_connections(int starting_client_id, int num_clients,
//                                       int max_send_size, int max_recv_size) {
//   struct ibv_qp_init_attr attr =
//     prepare_qp(this->getPD(), max_send_size, max_recv_size, false);
//   attr.cap.max_send_sge = 3; // this has to be 3 for the ring buffer impl we are using

//   // 2048 because of the size of each slot for each client (16) * the number of
//   // slows allowed on hw (128)
//   size_t mem2_size = 2048 * num_clients;

//   // 2. aligned_alloc strict size constraint
//   size_t alloc_size = ((mem2_size + 4095) / 4096) * 4096;
//   if (alloc_size == 0) alloc_size = 4096;

//   char *mem2 = (char *)aligned_alloc(4096, alloc_size);
//   memset(mem2, 0, alloc_size);

//   struct ibv_mr *mr2 =
//       ibv_reg_mr(this->getPD(), mem2, alloc_size,
//                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
//                  IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

//   local_mrs.push_back(mr2);
//   local_mems.push_back(mem2);
//   for (uint32_t i = 0; i < num_clients; i++) {
//     char *mem = (char *)GetMagicBuffer(RING_BUFFER_SIZE);
//     struct ibv_mr *mr =
//         ibv_reg_mr(this->getPD(), mem, RING_BUFFER_SIZE * 2,
//                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
//                        IBV_ACCESS_REMOTE_READ);

//     printf("MagicBuffer at %p and it repeats at %p\n", mem,
//            mem + RING_BUFFER_SIZE);

//     Buffer *lb = new ReverseRingBuffer(mr, local_log2(RING_BUFFER_SIZE));
//     BufferContext bc = lb->GetContext();

//     // 2. Zero-initialize the struct to prevent stack garbage leaks
//     connect_info info = {};
//     info.code = 4;
//     info.ctx = bc;
//     info.rkey_magic = mr2->rkey;
//     info.server_id = my_id;

//     // 3. Offset the remote doorbell address for this specific client
//     uint64_t base_addr = (uint64_t)mr2->addr;
//     info.addr_magic = base_addr + (i * 2048);
//     info.addr_magic2 = info.addr_magic;

//     VerbsEP *ep;
//     connect_info *connect_buffer;
//     std::tie(ep, connect_buffer) =
//         get_client_ep_and_info(attr, &info, 16, false);

//     info = *connect_buffer;
//     uint64_t client_id = connect_buffer->server_id;

//     RemoteBuffer *rb = new ReverseRemoteBuffer(info.ctx);

//     // 4. Offset the local staging memory for the SGE metadata
//     uint64_t local_mem = base_addr + (i * 2048);
//     uint32_t local_mem_lkey = mr2->lkey;

//     receivers[client_id] = std::make_unique<CircularReverseReceiver>(lb);
//     senders[client_id] = std::make_unique<CircularConnectionReverse>(
//         ep, rb, local_mem, local_mem_lkey);
//     eps[client_id] = ep;
//     connect_buffers[client_id] = connect_buffer;

//     in_flight_regions[client_id] = std::vector<Region*>(128, nullptr); // 128 slots
//     pending_ack_ids[client_id] = 1;                                    // Matches next_id_
//     pending_freed_bytes[client_id].store(0, std::memory_order_relaxed); // Zero out counter
//   }
// }

std::pair<VerbsEP *, connect_info *>
RDMAManager::get_server_ep_and_info(int server_id,
                                    int max_send_size,
                                    int max_recv_size,
                                    void *my_info,
                                    uint32_t recv_batch,
                                    bool recv_with_data) {
  std::string server_ip = address_list[server_id].first;
  std::string server_port = address_list[server_id].second;

  while (true) {
    struct rdma_cm_id *client_cm_id =
        send_connect_request(server_ip.c_str(), server_port.c_str());

    if (!client_cm_id) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    struct rdma_event_channel *temp_channel = client_cm_id->channel;
    struct ibv_pd *pd = this->getPD();

    if (!pd) {
      fprintf(stderr, "getPD() returned null\n");
      rdma_destroy_id(client_cm_id);
      if (temp_channel) {
        rdma_destroy_event_channel(temp_channel);
      }
      exit(1);
    }

    if (client_cm_id->verbs && client_cm_id->verbs != pd->context) {
      fprintf(stderr,
              "RDMA device context mismatch for peer %d\n",
              server_id);

      rdma_destroy_id(client_cm_id);
      if (temp_channel) {
        rdma_destroy_event_channel(temp_channel);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    struct ibv_qp_init_attr attr =
        prepare_qp(pd, max_send_size, max_recv_size, false);

    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;

    struct ibv_cq *send_cq = attr.send_cq;
    struct ibv_cq *recv_cq = attr.recv_cq;

    auto cleanup_failed_attempt = [&]() {
      if (client_cm_id) {
        if (client_cm_id->qp) {
          rdma_destroy_qp(client_cm_id);
        }

        rdma_destroy_id(client_cm_id);
        client_cm_id = nullptr;
      }

      drain_cq(send_cq);
      if (recv_cq != send_cq) {
        drain_cq(recv_cq);
      }

      if (send_cq) {
        int ret = ibv_destroy_cq(send_cq);
        if (ret) {
          printf("WARNING: failed to destroy failed-attempt send CQ for peer %d: ret=%d errno=%d\n",
                 server_id,
                 ret,
                 errno);
        }
        send_cq = nullptr;
      }

      if (recv_cq && recv_cq != send_cq) {
        int ret = ibv_destroy_cq(recv_cq);
        if (ret) {
          printf("WARNING: failed to destroy failed-attempt recv CQ for peer %d: ret=%d errno=%d\n",
                 server_id,
                 ret,
                 errno);
        }
        recv_cq = nullptr;
      }

      if (temp_channel) {
        rdma_destroy_event_channel(temp_channel);
        temp_channel = nullptr;
      }
    };

    uint32_t saved_max_recv_size = attr.cap.max_recv_wr;
    if (attr.srq) {
      attr.cap.max_recv_wr = 0;
    }

    if (rdma_create_qp(client_cm_id, pd, &attr)) {
      perror("rdma_create_qp active");
      attr.cap.max_recv_wr = saved_max_recv_size;

      cleanup_failed_attempt();
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    client_cm_id->context = (void *)(uintptr_t)server_id;
    attr.cap.max_recv_wr = saved_max_recv_size;

    VerbsEP *ep =
        new VerbsEP(client_cm_id, attr, recv_batch, recv_with_data);

    struct rdma_conn_param conn_param;
    memset(&conn_param, 0, sizeof(conn_param));

    conn_param.responder_resources = 16;
    conn_param.initiator_depth = 16;
    conn_param.retry_count = 3;
    conn_param.rnr_retry_count = 7;
    conn_param.private_data = my_info;
    conn_param.private_data_len = sizeof(connect_info);

    if (rdma_connect(client_cm_id, &conn_param)) {
      perror("rdma_connect active");

      delete ep;
      cleanup_failed_attempt();

      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    struct rdma_cm_event *event = nullptr;
    if (rdma_get_cm_event(client_cm_id->channel, &event)) {
      perror("rdma_get_cm_event active");

      delete ep;
      cleanup_failed_attempt();

      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    if (event->event != RDMA_CM_EVENT_ESTABLISHED) {
      enum rdma_cm_event_type ev = event->event;
      rdma_ack_cm_event(event);

      delete ep;
      cleanup_failed_attempt();

      std::cout << "Waiting for Node " << server_id
                << " to boot... event=" << ev << std::endl;

      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    if (!event->param.conn.private_data_len) {
      rdma_ack_cm_event(event);

      delete ep;
      cleanup_failed_attempt();

      fprintf(stderr, "Error: did not get private data on connect to peer %d\n",
              server_id);
      exit(1);
    }

    void *connect_buffer = malloc(event->param.conn.private_data_len);
    if (!connect_buffer) {
      rdma_ack_cm_event(event);

      delete ep;
      cleanup_failed_attempt();

      perror("malloc connect_buffer");
      exit(1);
    }

    memcpy(connect_buffer,
           event->param.conn.private_data,
           event->param.conn.private_data_len);

    rdma_ack_cm_event(event);

    return std::make_pair(ep, (connect_info *)connect_buffer);
  }
}
// void RDMAManager::send_connections(int starting_server_id, int num_servers,
//                                    int max_send_size, int max_recv_size) {
//   struct ibv_qp_init_attr attr =
//     prepare_qp(this->getPD(), max_send_size, max_recv_size, false);
//   attr.cap.max_send_sge = 3; // has to be 3 for the ring buffer impl

//   size_t mem2_size = 2048 * num_servers;

//   // 2. aligned_alloc strictly requires size to be a multiple of alignment
//   size_t alloc_size = ((mem2_size + 4095) / 4096) * 4096;
//   if (alloc_size == 0) alloc_size = 4096; // Guard against 0 servers

//   char *mem2 = (char *)aligned_alloc(4096, alloc_size);
//   memset(mem2, 0, alloc_size);

//   struct ibv_mr *mr2 =
//       ibv_reg_mr(this->getPD(), mem2, alloc_size,
//                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
//                  IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
//   local_mrs.push_back(mr2);
//   local_mems.push_back(mem2);
//   for (uint32_t i = 0; i < num_servers; i++) {
//     int server_id = starting_server_id + i;
//     std::string server_ip = address_list[server_id].first,
//                 server_port = address_list[server_id].second;

//     // this->send_connect_request(server_ip.c_str(), server_port.c_str());

//     char *mem = (char *)GetMagicBuffer(RING_BUFFER_SIZE);
//     struct ibv_mr *mr =
//         ibv_reg_mr(this->getPD(), mem, RING_BUFFER_SIZE * 2,
//                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
//                        IBV_ACCESS_REMOTE_READ);
//     printf("MagicBuffer at %p and it repeats at %p\n", mem,
//            mem + RING_BUFFER_SIZE);
//     Buffer *lb = new ReverseRingBuffer(mr, local_log2(RING_BUFFER_SIZE));
//     BufferContext bc = lb->GetContext();

//     connect_info info = {};
//     info.code = 4;
//     info.ctx = bc;
//     info.rkey_magic = mr2->rkey;
//     info.server_id = my_id;

//     uint64_t base_addr = (uint64_t)mr2->addr;
//     info.addr_magic = base_addr + (i * 2048);
//     info.addr_magic2 = info.addr_magic;

//     VerbsEP *ep;
//     connect_info *connect_buffer;

//     std::tie(ep, connect_buffer) =
//         get_server_ep_and_info(server_id, attr, &info, 16, false);

//     info = *(connect_info *)connect_buffer;

//     RemoteBuffer *rb = new ReverseRemoteBuffer(info.ctx);
//     uint64_t local_mem = base_addr + (i * 2048);
//     uint32_t local_mem_lkey = mr2->lkey;

//     receivers[server_id] = (std::make_unique<CircularReverseReceiver>(lb));
//     senders[server_id] = std::make_unique<CircularConnectionReverse>(
//         ep, rb, local_mem, local_mem_lkey);
//     eps[server_id] = ep;
//     connect_buffers[server_id] = connect_buffer;

//     in_flight_regions[server_id] = std::vector<Region*>(128, nullptr); // 128 slots
//     pending_ack_ids[server_id] = 1;                                    // Matches next_id_
//     pending_freed_bytes[server_id].store(0, std::memory_order_relaxed); // Zero out counter
//   }
// }

void RDMAManager::cleanup() {
  int ret = 0;

  // Threads should already be stopped by shutdown_threads().
  // These wrappers have empty destructors, but clearing them prevents accidental
  // future use while we tear down QPs/CQs/MRs.
  receivers.clear();
  senders.clear();

  for (int i = 0; i < num_servers; i++) {
    if (eps[i] == nullptr) continue;

    struct rdma_cm_id *cm_id = eps[i]->id;
    struct rdma_event_channel *channel = cm_id ? cm_id->channel : nullptr;

    struct ibv_qp *qp = cm_id ? cm_id->qp : nullptr;
    struct ibv_cq *send_cq = qp ? qp->send_cq : nullptr;
    struct ibv_cq *recv_cq = qp ? qp->recv_cq : nullptr;

    bool active_side = (i > my_id);

    if (cm_id) {
      rdma_disconnect(cm_id);

      // Required before rdma_destroy_id().
      if (cm_id->qp) {
        rdma_destroy_qp(cm_id);  // returns void
      }
    }

    drain_cq(send_cq);
    if (recv_cq != send_cq) {
      drain_cq(recv_cq);
    }

    if (send_cq) {
      ret = ibv_destroy_cq(send_cq);
      if (ret) {
        printf("WARNING: failed to destroy send CQ for peer %d: ret=%d errno=%d\n",
               i,
               ret,
               errno);
      }
    }

    if (recv_cq && recv_cq != send_cq) {
      ret = ibv_destroy_cq(recv_cq);
      if (ret) {
        printf("WARNING: failed to destroy recv CQ for peer %d: ret=%d errno=%d\n",
               i,
               ret,
               errno);
      }
    }

    if (cm_id) {
      ret = rdma_destroy_id(cm_id);
      if (ret) {
        printf("Failed to destroy client id cleanly for peer %d: errno=%d\n",
               i,
               errno);
      }
    }

    if (active_side && channel) {
      rdma_destroy_event_channel(channel);
    }

    if (connect_buffers[i]) {
      free(connect_buffers[i]);
      connect_buffers[i] = nullptr;
    }

    delete eps[i];
    eps[i] = nullptr;
  }

  for (size_t i = 0; i < local_mrs.size(); i++) {
    if (local_mrs[i]) {
      ibv_dereg_mr(local_mrs[i]);
    }

    if (local_mems[i]) {
      free(local_mems[i]);
    }
  }

  local_mrs.clear();
  local_mems.clear();

  if (this->_listen_id) {
    rdma_destroy_id(this->_listen_id);
    this->_listen_id = nullptr;
  }

  if (preallocated_region_mr != nullptr) {
    ibv_dereg_mr(preallocated_region_mr);
    preallocated_region_mr = nullptr;
  }

  if (preallocated_region_addr != nullptr) {
    ::free(preallocated_region_addr);
    preallocated_region_addr = nullptr;
  }

  if (this->_pd) {
    ibv_dealloc_pd(this->_pd);
    this->_pd = nullptr;
  }

  if (this->_ec) {
    rdma_destroy_event_channel(this->_ec);
    this->_ec = nullptr;
  }
}
void RDMAManager::enqueue_region_to_send(uint64_t peer_id, Region *r) {
  outgoing_queues[peer_id].enqueue(r);
}
// void RDMAManager::recv_loop() {
//     std::vector<Region> recvs;

//     while (this->running) {
//       // std::cout << "Running recv loop" << std::endl;
//         for (int i = 0; i < num_servers; i++) {
//             if (!receivers[i]) continue;

//             uint32_t freed_this_batch = 0;

//             // Single argument exactly as requested!
//             int received_messages = receivers[i]->Receive(recvs);

//             if (received_messages > 0) {
//                 for (auto& recv : recvs) {
//                     char* msg_ptr = (char*)recv.addr;

//                     // msg_ptr[0] is the Magic 0 Reset Bell
//                     msg_opcode_t opcode = (msg_opcode_t)msg_ptr[1]; // Your Application Opcode
//                     char* payload_ptr = msg_ptr + 2; // Pointer to the pure user data

//                     // Route based on Opcode
//                     if (opcode == msg_opcode_t::USER_DATA) {
//                       // User Data
//                       // std::cout << "received data " <<std::endl;
//                       this->handler(payload_ptr, recv.length);
//                     } else if (opcode == msg_opcode_t::FREED_BYTES) {

//                         // Flow Control ACK
//                         uint32_t freed_ack = *(uint32_t *)payload_ptr;
//                         std::cout << "received ack, num freed " << freed_ack << std::endl;

//                         // Tell our Sender that the REMOTE buffer has cleared space!
//                         senders[i]->AckSentBytes(freed_ack);
//                     }

//                     // THE FIX:
//                     // Footprint = 1 (Opcode) + payload length + 4 (Length) + 1 (Shared Magic Byte)
//                     uint32_t message_footprint = 1 + recv.length + 5;
//                     freed_this_batch += message_footprint;

//                     // Recycle the pointer
//                     receivers[i]->FreeReceive(recv);
//                 }
//                 recvs.clear();
//             }

//             // Lock-free handoff: Tell the Send Thread to ACK these freed bytes
//             if (freed_this_batch > 0) {
//                 pending_freed_bytes[i].fetch_add(freed_this_batch, std::memory_order_relaxed);
//             }
//         }
//     }
// }
// void RDMAManager::send_loop() {
//     std::vector<int32_t> can_send(num_servers, 120);

//     // THE FIX: Track in-flight ACKs to enforce the 8-slot limit
//     std::vector<int32_t> acks_in_flight(num_servers, 0);

//     std::vector<Region*> stalled_regions(num_servers, nullptr);
//     const uint32_t ACK_THRESHOLD = 1024;

//     while (this->running) {
//         for (int i = 0; i < num_servers; i++) {
//             if (!senders[i]) continue;

//             // =================================================================
//             // 1. SEND PENDING FLOW-CONTROL ACKs (VIP TRAFFIC)
//             // =================================================================
//             uint32_t current_freed = pending_freed_bytes[i].load(std::memory_order_relaxed);

//             if (current_freed >= ACK_THRESHOLD && acks_in_flight[i] < 8) {
//               uint32_t bytes_to_ack = pending_freed_bytes[i].exchange(0, std::memory_order_relaxed);
//               uint64_t ack_id = senders[i]->SendAckAsync(bytes_to_ack);

//               if (ack_id == (uint64_t)-1) {
//                 // Yield and put the bytes back!
//                 pending_freed_bytes[i].fetch_add(bytes_to_ack, std::memory_order_relaxed);
//               } else {
//                 std::cout << "sending bytes to ack " << bytes_to_ack << std::endl;
//                 in_flight_regions[i][ack_id % MAX_SEND_RECV_WR] = nullptr;
//                 acks_in_flight[i]++;
//               }
//             }

//             // =================================================================
//             // 2. RECYCLE HARDWARE SLOTS & GARBAGE COLLECTION
//             // =================================================================
//             while (senders[i]->TestSend(pending_ack_ids[i])) {
//                 Region* completed_region = in_flight_regions[i][pending_ack_ids[i] % MAX_SEND_RECV_WR];

//                 if (completed_region != nullptr) {
//                     // It was a Data Message
//                     Region::delete_addr(completed_region->addr, (void *)completed_region);
//                     can_send[i]++;
//                 } else {
//                     // THE FIX: It was an ACK. Refund the ACK slot!
//                     acks_in_flight[i]--;
//                 }

//                 pending_ack_ids[i]++;
//             }

//             // =================================================================
//             // 3. TRANSMIT USER DATA
//             // =================================================================
//             while (can_send[i] > 0) {
//                 Region* target_region = stalled_regions[i];

//                 if (target_region == nullptr) {
//                     if (!outgoing_queues[i].try_dequeue(target_region)) {
//                         break;
//                     }
//                 }

//                 uint64_t msg_id = senders[i]->SendDataAsync(target_region);

//                 if (msg_id == (uint64_t)-1) {
//                     stalled_regions[i] = target_region;
//                     break;
//                 }

//                 stalled_regions[i] = nullptr;
//                 in_flight_regions[i][msg_id % MAX_SEND_RECV_WR] = target_region;
//                 can_send[i]--;
//             }
//         }
//     }
// }

void RDMAManager::recv_loop() {
    std::vector<Region> recvs;
    std::vector<uint32_t> incoming_credits_batch;

    while (this->running) {
        for (int i = 0; i < num_servers; i++) {
            if (!receivers[i]) continue;

            uint32_t freed_this_batch = 0;

            // Poll for both Data AND incoming ACKs
            int received_messages = receivers[i]->Receive(recvs, incoming_credits_batch);

            // 1. Process received credits (from peers we sent data to)
            if (!incoming_credits_batch.empty()) {
                uint32_t total_credits = 0;
                for (uint32_t c : incoming_credits_batch) total_credits += c;

                // Add credits to sender loop via atomic
                incoming_credits[i].fetch_add(total_credits, std::memory_order_relaxed);
                incoming_credits_batch.clear();
            }

            // 2. Process received user data (from peers sending to us)
            if (received_messages > 0) {
                for (auto& recv : recvs) {
                    char* payload_ptr = (char*)recv.addr;

                    // Route application data
                    this->handler(payload_ptr, recv.length);

                    // Free the buffer right back to the NIC!
                    receivers[i]->FreeReceive(payload_ptr);

                    // Instead of tracking bytes, we track the NUMBER of messages processed (credits)
                    freed_this_batch++;
                }
                recvs.clear();
            }

            // Tell the Send Thread to ACK these freed buffers
            if (freed_this_batch > 0) {
                pending_freed_bytes[i].fetch_add(freed_this_batch, std::memory_order_relaxed);
            }
        }
    }
}

void RDMAManager::send_loop() {
  constexpr int32_t ACK_SEND_RESERVE = 8;
  constexpr int32_t RECV_CONTROL_RESERVE = 24;
  constexpr int32_t DATA_CREDIT_LIMIT =
      MAX_SEND_RECV_WR - ACK_SEND_RESERVE - RECV_CONTROL_RESERVE;

  static_assert(DATA_CREDIT_LIMIT > 0,
                "DATA_CREDIT_LIMIT must be positive");

  // ACK less aggressively than every 4 messages so ACK traffic does not eat
  // too many remote receive WQEs under full-duplex traffic.
  const uint32_t ACK_THRESHOLD = 16;

  // Fallback: do not strand a small number of credits forever at the end of
  // a burst. If fewer than ACK_THRESHOLD credits are pending, flush them after
  // this interval.
  using Clock = std::chrono::steady_clock;
  const auto ACK_FLUSH_INTERVAL = std::chrono::microseconds(100);

  std::vector<int32_t> can_send(num_servers, DATA_CREDIT_LIMIT);
  std::vector<int32_t> acks_in_flight(num_servers, 0);
  std::vector<Region *> stalled_regions(num_servers, nullptr);
  std::vector<Clock::time_point> last_ack_flush(num_servers, Clock::now());

  while (this->running) {
    for (int i = 0; i < num_servers; i++) {
      if (!senders[i]) continue;

      // =================================================================
      // 0. READ INCOMING CREDITS
      // =================================================================
      // These are credits sent by peer i telling us that peer i has freed
      // receive slots, so we may send more DATA messages to peer i.
      uint32_t new_credits =
          incoming_credits[i].exchange(0, std::memory_order_relaxed);

      if (new_credits > 0) {
        can_send[i] += static_cast<int32_t>(new_credits);

        // Never allow data sends to consume the peer's entire receive queue.
        if (can_send[i] > DATA_CREDIT_LIMIT) {
          can_send[i] = DATA_CREDIT_LIMIT;
        }
      }

      // =================================================================
      // 1. RECYCLE COMPLETED SENDS
      // =================================================================
      // This frees local Region buffers after DATA send completion and also
      // frees ACK in-flight slots after ACK send completion.
      while (senders[i]->TestSend(pending_ack_ids[i])) {
        Region *completed_region =
            in_flight_regions[i][pending_ack_ids[i] % MAX_SEND_RECV_WR];

        if (completed_region != nullptr) {
          // DATA send completed locally. This only means the NIC is done with
          // the local buffer; it does NOT mean the remote app processed it.
          Region::delete_addr(completed_region->addr,
                              reinterpret_cast<void *>(completed_region));

          in_flight_regions[i][pending_ack_ids[i] % MAX_SEND_RECV_WR] =
              nullptr;
        } else {
          // ACK send completed.
          if (acks_in_flight[i] > 0) {
            acks_in_flight[i]--;
          }
        }

        pending_ack_ids[i]++;
      }

      // =================================================================
      // 2. SEND PENDING FLOW-CONTROL ACKs
      // =================================================================
      // These are credits we send to peer i, telling peer i that we processed
      // messages from it and freed our receive slots.
      uint32_t current_freed =
          pending_freed_bytes[i].load(std::memory_order_relaxed);

      auto now = Clock::now();

      bool threshold_ready = current_freed >= ACK_THRESHOLD;
      bool timeout_ready =
          current_freed > 0 &&
          (now - last_ack_flush[i]) >= ACK_FLUSH_INTERVAL;

      if ((threshold_ready || timeout_ready) &&
          acks_in_flight[i] < ACK_SEND_RESERVE) {
        uint32_t credits_to_ack =
            pending_freed_bytes[i].exchange(0, std::memory_order_relaxed);

        if (credits_to_ack > 0) {
          uint64_t ack_id = senders[i]->SendAckAsync(credits_to_ack);

          if (ack_id == static_cast<uint64_t>(-1)) {
            // Could not post the ACK. Put the credits back and retry later.
            pending_freed_bytes[i].fetch_add(credits_to_ack,
                                             std::memory_order_relaxed);
          } else {
            // nullptr means this WR ID belongs to an ACK, not a DATA Region.
            in_flight_regions[i][ack_id % MAX_SEND_RECV_WR] = nullptr;
            acks_in_flight[i]++;
            last_ack_flush[i] = now;

            // Optional debug:
            // std::cout << "Fired an ACK for " << credits_to_ack
            //           << " credits back to Sender!" << std::endl;
          }
        }
      }

      // =================================================================
      // 3. TRANSMIT USER DATA
      // =================================================================
      // DATA sends are controlled only by remote credits. Local send completion
      // does not replenish can_send[i].
      while (can_send[i] > 0) {
        Region *target_region = stalled_regions[i];

        if (target_region == nullptr) {
          if (!outgoing_queues[i].try_dequeue(target_region)) {
            break;
          }
        }

        uint64_t msg_id = senders[i]->SendAsync(target_region);

        if (msg_id == static_cast<uint64_t>(-1)) {
          // Send queue is temporarily full. Keep this exact Region and retry
          // later instead of dequeuing another one.
          stalled_regions[i] = target_region;
          break;
        }

        stalled_regions[i] = nullptr;
        in_flight_regions[i][msg_id % MAX_SEND_RECV_WR] = target_region;
        can_send[i]--;
      }
    }
  }
}
void RDMAManager::start_send_recv_threads() {
  running = true;
  send_thread = std::thread(&RDMAManager::send_loop, this);
  recv_thread = std::thread(&RDMAManager::recv_loop, this);
}

void RDMAManager::shutdown_threads() {
  running = false;
  if (send_thread.joinable()) {
    send_thread.join();
  }
  if (recv_thread.joinable()) {
    recv_thread.join();
  }
}


std::pair<char *, uint32_t> RDMAManager::get_preallocated_region_ptr_lkey(size_t block_size_per_element, size_t num_elements) {
  size_t total_size = block_size_per_element * num_elements;
  size_t alloc_size = ((total_size + 4095) / 4096) * 4096;
  if (alloc_size == 0) alloc_size = 4096;

  char *mem = (char *)aligned_alloc(4096, alloc_size);
  memset(mem, 0, alloc_size);

  // Register the memory
  preallocated_region_mr = ibv_reg_mr(
                                   _pd, mem, alloc_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
  preallocated_region_addr = mem;
  return {preallocated_region_addr, preallocated_region_mr->lkey};
}


RDMAManager::~RDMAManager() {
  shutdown_threads();
  cleanup();
}
