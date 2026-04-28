#include "rdma_com/rdma_manager.hpp"
#include "rdma_com/consts.hpp"
#include "rdma_com/reverse_ring_communicators.hpp"
#include <infiniband/verbs.h>


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
  std::string ip, port;

  size_t colon_id = ip_port.find(":");
  ip = ip_port.substr(0, colon_id);
  port = ip_port.substr(colon_id + 1);
  return std::make_pair(ip, port);
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

void RDMAManager::connect_to_all_servers() {
  eps.resize(num_servers);
  eps[my_id] = nullptr;

  receivers.resize(num_servers);
  receivers[my_id] = nullptr;

  senders.resize(num_servers);
  senders[my_id] = nullptr;

  connect_buffers.resize(num_servers);
  connect_buffers[my_id] = nullptr;

  in_flight_regions.resize(num_servers);
  pending_ack_ids.resize(num_servers);

  // Replace pending_freed_bytes.resize(num_servers) with this:
  pending_freed_bytes.reset(new std::atomic<uint32_t>[num_servers]);
  for (size_t i = 0; i < num_servers; i++) {
    pending_freed_bytes[i].store(0, std::memory_order_relaxed);
  }
  if (my_id != 0) {
    receive_connections(0, my_id, MAX_SEND_RECV_WR, MAX_SEND_RECV_WR);
  }
  if (my_id != num_servers - 1) {
    send_connections(my_id + 1, num_servers - my_id - 1, MAX_SEND_RECV_WR,
                     MAX_SEND_RECV_WR);
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
  conn_param.rnr_retry_count = 3;
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

void RDMAManager::receive_connections(int starting_client_id, int num_clients,
                                      int max_send_size, int max_recv_size) {
  struct ibv_qp_init_attr attr =
    prepare_qp(this->getPD(), max_send_size, max_recv_size, false);
  attr.cap.max_send_sge = 3; // this has to be 3 for the ring buffer impl we are using

  // 2048 because of the size of each slot for each client (16) * the number of
  // slows allowed on hw (128)
  size_t mem2_size = 2048 * num_clients;

  // 2. aligned_alloc strict size constraint
  size_t alloc_size = ((mem2_size + 4095) / 4096) * 4096;
  if (alloc_size == 0) alloc_size = 4096;

  char *mem2 = (char *)aligned_alloc(4096, alloc_size);
  memset(mem2, 0, alloc_size);

  struct ibv_mr *mr2 =
      ibv_reg_mr(this->getPD(), mem2, alloc_size,
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                 IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

  local_mrs.push_back(mr2);
  local_mems.push_back(mem2);
  for (uint32_t i = 0; i < num_clients; i++) {
    char *mem = (char *)GetMagicBuffer(RING_BUFFER_SIZE);
    struct ibv_mr *mr =
        ibv_reg_mr(this->getPD(), mem, RING_BUFFER_SIZE * 2,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ);

    printf("MagicBuffer at %p and it repeats at %p\n", mem,
           mem + RING_BUFFER_SIZE);

    Buffer *lb = new ReverseRingBuffer(mr, local_log2(RING_BUFFER_SIZE));
    BufferContext bc = lb->GetContext();

    // 2. Zero-initialize the struct to prevent stack garbage leaks
    connect_info info = {};
    info.code = 4;
    info.ctx = bc;
    info.rkey_magic = mr2->rkey;
    info.server_id = my_id;

    // 3. Offset the remote doorbell address for this specific client
    uint64_t base_addr = (uint64_t)mr2->addr;
    info.addr_magic = base_addr + (i * 2048);
    info.addr_magic2 = info.addr_magic;

    VerbsEP *ep;
    connect_info *connect_buffer;
    std::tie(ep, connect_buffer) =
        get_client_ep_and_info(attr, &info, 16, false);

    info = *connect_buffer;
    uint64_t client_id = connect_buffer->server_id;

    RemoteBuffer *rb = new ReverseRemoteBuffer(info.ctx);

    // 4. Offset the local staging memory for the SGE metadata
    uint64_t local_mem = base_addr + (i * 2048);
    uint32_t local_mem_lkey = mr2->lkey;

    receivers[client_id] = std::make_unique<CircularReverseReceiver>(lb);
    senders[client_id] = std::make_unique<CircularConnectionReverse>(
        ep, rb, local_mem, local_mem_lkey);
    eps[client_id] = ep;
    connect_buffers[client_id] = connect_buffer;

    in_flight_regions[client_id] = std::vector<Region*>(128, nullptr); // 128 slots
    pending_ack_ids[client_id] = 1;                                    // Matches next_id_
    pending_freed_bytes[client_id].store(0, std::memory_order_relaxed); // Zero out counter
  }
}

std::pair<VerbsEP *, connect_info *>
RDMAManager::get_server_ep_and_info(int server_id, struct ibv_qp_init_attr attr,
                                    void *my_info, uint32_t recv_batch,
                                    bool recv_with_data) {
  std::string server_ip = address_list[server_id].first;
  std::string server_port = address_list[server_id].second;

  while (true) {
    // 1. Attempt to resolve route and get an ID
    struct rdma_cm_id *client_cm_id =
        send_connect_request(server_ip.c_str(), server_port.c_str());

    // If address resolution failed, sleep and retry
    if (!client_cm_id) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    struct ibv_pd *pd = client_cm_id->pd;
    uint32_t max_recv_size = attr.cap.max_recv_wr;
    if (attr.srq) attr.cap.max_recv_wr = 0;

    struct rdma_event_channel *temp_channel = client_cm_id->channel;

    // 2. Create the Queue Pair
    if (rdma_create_qp(client_cm_id, pd, &attr)) {
      rdma_destroy_id(client_cm_id);
      if (temp_channel)rdma_destroy_event_channel(temp_channel);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    client_cm_id->context = (void *)(uintptr_t)server_id;
    attr.cap.max_recv_wr = max_recv_size;

    VerbsEP *ep = new VerbsEP(client_cm_id, attr, recv_batch, recv_with_data);

    // 3. Prepare connection parameters
    struct rdma_conn_param conn_param;
    memset(&conn_param, 0, sizeof(conn_param));
    conn_param.responder_resources = 16;
    conn_param.initiator_depth = 16;
    conn_param.retry_count = 3;
    conn_param.rnr_retry_count = 3;
    conn_param.private_data = my_info;
    conn_param.private_data_len = sizeof(connect_info);

    // 4. Initiate the connection
    if (rdma_connect(client_cm_id, &conn_param)) {
      // Failed to send the connect request packet
      delete ep; // Clean up the wrapper and QP
      rdma_destroy_id(client_cm_id);
      if (temp_channel)rdma_destroy_event_channel(temp_channel);
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    // 5. Block until the target responds
    struct rdma_cm_event *event;
    if (rdma_get_cm_event(client_cm_id->channel, &event)) {
      perror("rdma_get_cm_event");
      exit(1); // Genuine OS failure (e.g., file descriptor broken)
    }

    // 6. THE MAGIC CHECK: Did they accept or reject?
    if (event->event != RDMA_CM_EVENT_ESTABLISHED) {
      // The target node is not listening yet! (REJECTED or TIMEOUT)

      struct rdma_event_channel *temp_channel = client_cm_id->channel;
      rdma_ack_cm_event(event); // Acknowledge the rejection
      delete ep;                // Destroy the EP/QP

      rdma_destroy_id(client_cm_id); // Destroy the dead CM ID
      if (temp_channel) rdma_destroy_event_channel(temp_channel);
      std::cout << "Waiting for Node " << server_id << " to boot..." << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue; // Loop back to the top and try again!
    }

    // 7. SUCCESS! We are connected.
    if (!event->param.conn.private_data_len) {
      printf("Error did not get data on connect \n");
      exit(1);
    }

    void *connect_buffer = malloc(event->param.conn.private_data_len);
    memcpy(connect_buffer, event->param.conn.private_data,
           event->param.conn.private_data_len);

    rdma_ack_cm_event(event);
    return std::make_pair(ep, (connect_info *)connect_buffer);
  }
}
void RDMAManager::send_connections(int starting_server_id, int num_servers,
                                   int max_send_size, int max_recv_size) {
  struct ibv_qp_init_attr attr =
    prepare_qp(this->getPD(), max_send_size, max_recv_size, false);
  attr.cap.max_send_sge = 3; // has to be 3 for the ring buffer impl

  size_t mem2_size = 2048 * num_servers;

  // 2. aligned_alloc strictly requires size to be a multiple of alignment
  size_t alloc_size = ((mem2_size + 4095) / 4096) * 4096;
  if (alloc_size == 0) alloc_size = 4096; // Guard against 0 servers

  char *mem2 = (char *)aligned_alloc(4096, alloc_size);
  memset(mem2, 0, alloc_size);

  struct ibv_mr *mr2 =
      ibv_reg_mr(this->getPD(), mem2, alloc_size,
                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                 IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
  local_mrs.push_back(mr2);
  local_mems.push_back(mem2);
  for (uint32_t i = 0; i < num_servers; i++) {
    int server_id = starting_server_id + i;
    std::string server_ip = address_list[server_id].first,
                server_port = address_list[server_id].second;

    // this->send_connect_request(server_ip.c_str(), server_port.c_str());

    char *mem = (char *)GetMagicBuffer(RING_BUFFER_SIZE);
    struct ibv_mr *mr =
        ibv_reg_mr(this->getPD(), mem, RING_BUFFER_SIZE * 2,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ);
    printf("MagicBuffer at %p and it repeats at %p\n", mem,
           mem + RING_BUFFER_SIZE);
    Buffer *lb = new ReverseRingBuffer(mr, local_log2(RING_BUFFER_SIZE));
    BufferContext bc = lb->GetContext();

    connect_info info = {};
    info.code = 4;
    info.ctx = bc;
    info.rkey_magic = mr2->rkey;
    info.server_id = my_id;

    uint64_t base_addr = (uint64_t)mr2->addr;
    info.addr_magic = base_addr + (i * 2048);
    info.addr_magic2 = info.addr_magic;

    VerbsEP *ep;
    connect_info *connect_buffer;

    std::tie(ep, connect_buffer) =
        get_server_ep_and_info(server_id, attr, &info, 16, false);

    info = *(connect_info *)connect_buffer;

    RemoteBuffer *rb = new ReverseRemoteBuffer(info.ctx);
    uint64_t local_mem = base_addr + (i * 2048);
    uint32_t local_mem_lkey = mr2->lkey;

    receivers[server_id] = (std::make_unique<CircularReverseReceiver>(lb));
    senders[server_id] = std::make_unique<CircularConnectionReverse>(
        ep, rb, local_mem, local_mem_lkey);
    eps[server_id] = ep;
    connect_buffers[server_id] = connect_buffer;

    in_flight_regions[server_id] = std::vector<Region*>(128, nullptr); // 128 slots
    pending_ack_ids[server_id] = 1;                                    // Matches next_id_
    pending_freed_bytes[server_id].store(0, std::memory_order_relaxed); // Zero out counter
  }
}

void RDMAManager::cleanup() {
  // based on https://github.com/animeshtrivedi/rdma-example
  int ret = -1;
  // =================================================================
  // 1. DISCONNECT PEERS & DESTROY QUEUE PAIRS
  // =================================================================
  for (int i = 0; i < num_servers; i++) {
    if (eps[i] == nullptr) continue;

    struct rdma_cm_id *client_cm_id = eps[i]->id;

    struct rdma_event_channel *temp_channel = client_cm_id->channel;
    struct ibv_cq *client_send_cq = eps[i]->qp->send_cq;
    struct ibv_cq *client_recv_cq = eps[i]->qp->recv_cq;

    // Disconnect safely before destroying
    rdma_disconnect(client_cm_id);
    rdma_destroy_qp(client_cm_id);

    ret = rdma_destroy_id(client_cm_id);
    if (ret) {
      printf("Failed to destroy client id cleanly, %d \n", -errno);
    }
    if (i > my_id && temp_channel) {
      rdma_destroy_event_channel(temp_channel);
    }

    ret = ibv_destroy_cq(client_send_cq);
    if (ret) {
      printf("Failed to destroy client send cq cleanly, %d \n", -errno);
    }

    ret = ibv_destroy_cq(client_recv_cq);
    if (ret) {
      printf("Failed to destroy client rev cq cleanly, %d \n", -errno);
    }

    free(connect_buffers[i]);

    // We don't have a completion channel and just poll from the ring buffer
    // so we don't need to destroy it.
    delete eps[i];
    eps[i] = nullptr;
  }
  receivers.clear();
  senders.clear();

  // =================================================================
  // 2. DEREGISTER MEMORY REGIONS (Sliding Window Metadata)
  // =================================================================
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

  // =================================================================
  // 3. TEARDOWN GLOBAL LISTENING INFRASTRUCTURE
  // =================================================================

  // Destroy the global listener to free the TCP port
  if (this->_listen_id) {
    rdma_destroy_id(this->_listen_id);
    this->_listen_id = nullptr;
  }
  if (preallocated_region_mr != nullptr) {
    ibv_dereg_mr(preallocated_region_mr);
    preallocated_region_mr = nullptr;
  }

  if (preallocated_region_addr != nullptr) {
    // MUST use free() for aligned_alloc, not delete
    ::free(preallocated_region_addr);
    preallocated_region_addr = nullptr;
  }


  // Free the Protection Domain if we manually allocated it
  // (In bind_server, we only allocated it if _listen_id->pd was null)
  if (this->_pd) {
    ibv_dealloc_pd(this->_pd);
    this->_pd = nullptr;
  }

  // Finally, destroy the Event Channel
  if (this->_ec) {
    rdma_destroy_event_channel(this->_ec);
    this->_ec = nullptr;
  }

}
void RDMAManager::enqueue_region_to_send(uint64_t peer_id, Region *r) {
  outgoing_queues[peer_id].enqueue(r);
}
void RDMAManager::recv_loop() {
    std::vector<Region> recvs;

    while (this->running) {
      // std::cout << "Running recv loop" << std::endl;
        for (int i = 0; i < num_servers; i++) {
            if (!receivers[i]) continue;

            uint32_t freed_this_batch = 0;

            // Single argument exactly as requested!
            int received_messages = receivers[i]->Receive(recvs);

            if (received_messages > 0) {
                for (auto& recv : recvs) {
                    char* msg_ptr = (char*)recv.addr;

                    // msg_ptr[0] is the Magic 0 Reset Bell
                    msg_opcode_t opcode = (msg_opcode_t)msg_ptr[1]; // Your Application Opcode
                    char* payload_ptr = msg_ptr + 2; // Pointer to the pure user data

                    // Route based on Opcode
                    if (opcode == msg_opcode_t::USER_DATA) {
                      // User Data
                      std::cout << "received data " <<std::endl;
                        this->handler(payload_ptr, recv.length);
                    }
                    else if (opcode == msg_opcode_t::FREED_BYTES) {
                        // Flow Control ACK
                        uint32_t freed_ack = *(uint32_t*)payload_ptr;

                        // Tell our Sender that the REMOTE buffer has cleared space!
                        senders[i]->AckSentBytes(freed_ack);
                    }

                    // THE FIX:
                    // Footprint = 1 (Opcode) + payload length + 4 (Length) + 1 (Shared Magic Byte)
                    uint32_t message_footprint = 1 + recv.length + 5;
                    freed_this_batch += message_footprint;

                    // Recycle the pointer
                    receivers[i]->FreeReceive(recv);
                }
                recvs.clear();
            }

            // Lock-free handoff: Tell the Send Thread to ACK these freed bytes
            if (freed_this_batch > 0) {
                pending_freed_bytes[i].fetch_add(freed_this_batch, std::memory_order_relaxed);
            }
        }
    }
}
void RDMAManager::send_loop() {
    // 120 tokens per connection. Leaves 8 hardware slots permanently reserved for ACKs.
    std::vector<int32_t> can_send(num_servers, 120);

    // ADDED: Track messages that were dequeued but couldn't be sent because the remote buffer is full
    std::vector<Region*> stalled_regions(num_servers, nullptr);

    // Threshold to prevent ACK ping-pong.
    const uint32_t ACK_THRESHOLD = 1024;

    while (this->running) {
      // std::cout << "running send loop" << std::endl;
        for (int i = 0; i < num_servers; i++) {
            if (!senders[i]) continue;

            // =================================================================
            // 1. SEND PENDING FLOW-CONTROL ACKs (VIP TRAFFIC)
            // =================================================================
            uint32_t current_freed = pending_freed_bytes[i].load(std::memory_order_relaxed);

            if (current_freed >= ACK_THRESHOLD) {
                uint32_t bytes_to_ack = pending_freed_bytes[i].exchange(0, std::memory_order_relaxed);
                uint64_t ack_id = senders[i]->SendAckAsync(bytes_to_ack);
                in_flight_regions[i][ack_id % 128] = nullptr;
            }

            // =================================================================
            // 2. RECYCLE HARDWARE SLOTS & GARBAGE COLLECTION
            // =================================================================
            while (senders[i]->TestSend(pending_ack_ids[i])) {
                Region* completed_region = in_flight_regions[i][pending_ack_ids[i] % 128];
                if (completed_region != nullptr) {
                  std::cout << "Hardware confirmed message " << pending_ack_ids[i]
                  << " was delivered to remote RAM!" << std::endl;
                  Region::delete_addr(completed_region->addr, (void *)completed_region);
                }
                pending_ack_ids[i]++;
                can_send[i]++;
            }

            // =================================================================
            // 3. TRANSMIT USER DATA (NON-BLOCKING BACKPRESSURE)
            // =================================================================
            while (can_send[i] > 0) {
                Region* target_region = stalled_regions[i];

                // If we don't have a stalled region waiting, try to grab a new one
                if (target_region == nullptr) {
                    if (!outgoing_queues[i].try_dequeue(target_region)) {
                        break; // Queue is empty, nothing to send right now
                    }
                }

                // Try to send it to the network
                std::cout << "sending data " << std::endl;
                uint64_t msg_id = senders[i]->SendDataAsync(target_region);

                if (msg_id == (uint64_t)-1) {
                    // THE REMOTE BUFFER IS FULL! (Avoid Deadlock)
                    // Save this region so we don't lose it, and break out of Phase 3.
                    // This allows the loop to go back to Phase 1 and send ACKs!
                    stalled_regions[i] = target_region;
                    break;
                }

                // Success! Clear the stalled state and record the flight
                stalled_regions[i] = nullptr;
                in_flight_regions[i][msg_id % 128] = target_region;
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
  cleanup();
}
