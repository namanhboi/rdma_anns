#include "rdma_com/rdma_manager.hpp"
#include "rdma_com/consts.hpp"
#include "rdma_com/ring.hpp"
#include "rdma_com/utils.hpp"
#include <cstdint>
#include <cstdlib>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

std::pair<std::string, std::string>
RDMAManager::parse_ip_port(const std::string &ip_port) {
  std::string ip, port;

  size_t colon_id = ip_port.find(":");
  ip = ip_port.substr(0, colon_id);
  port = ip_port.substr(colon_id + 1);
  return std::make_pair(ip, port);
}

RDMAManager::RDMAManager(uint64_t my_id, std::vector<std::string> addresses)
    : my_id(my_id), num_servers(addresses.size()) {
  for (const std::string &ip_port : addresses) {
    address_list.push_back(parse_ip_port(ip_port));
  }

  std::string my_ip = address_list[my_id].first;
  std::string my_port = address_list[my_id].second;
  bind_server(my_ip.c_str(), my_port.c_str());
}

void RDMAManager::bind_server(const char *ip, const char *port) {
  this->_ec = rdma_create_event_channel();
  if (!this->_ec) {
    printf("failed to create event channel\n");
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

  if (my_id != 0) {
    receive_connections(0, my_id, MAX_SEND_LEN, MAX_SEND_LEN);
  }
  if (my_id != num_servers - 1) {
    send_connections(my_id + 1, num_servers - my_id - 1, MAX_SEND_LEN,
                     MAX_SEND_LEN);
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
    perror("rdma_getaddrinfo\n");
    if (addrinfo) {
      rdma_freeaddrinfo(addrinfo);
    }
    exit(1);
  }

  struct rdma_event_channel *channel = rdma_create_event_channel();

  if (!channel) {
    printf("failed to create event channel\n");
    exit(1);
  }

  if (rdma_create_id(channel, &id, NULL, RDMA_PS_TCP)) {
    printf("failed to create listen id\n");
    exit(1);
  }

  if (rdma_resolve_addr(id, NULL, addrinfo->ai_dst_addr, 2000)) {
    printf("failed to rdma_resolve_addr\n");
    exit(1);
  }

  struct rdma_cm_event *event;

  if (rdma_get_cm_event(channel, &event)) {
    perror("rdma_get_cm_event");
    exit(1);
  }
  rdma_ack_cm_event(event);

  if (rdma_resolve_route(id, 2000)) {
    printf("failed to rdma_resolve_route\n");
    exit(1);
  }

  if (rdma_get_cm_event(channel, &event)) {
    perror("rdma_get_cm_event");
    exit(1);
  }
  rdma_ack_cm_event(event);

  if (addrinfo) {
    rdma_freeaddrinfo(addrinfo);
  }
  return id;
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

  // 1. Allocate 64 bytes per client (CPU cache-line aligned) to prevent false
  // sharing
  size_t mem2_size = 64 * num_clients;
  char *mem2 = (char *)aligned_alloc(4096, 4096);
  memset(mem2, 0, mem2_size);
  struct ibv_mr * mr2 = ibv_reg_mr(this->getPD(), mem2, 4096,
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

    Buffer *lb = new ReverseRingBuffer(mr, log2(RING_BUFFER_SIZE));
    BufferContext bc = lb->GetContext();

    // 2. Zero-initialize the struct to prevent stack garbage leaks
    connect_info info = {};
    info.code = 4;
    info.ctx = bc;
    info.rkey_magic = mr2->rkey;
    info.server_id = my_id;

    // 3. Offset the remote doorbell address for this specific client
    uint64_t base_addr = (uint64_t)mr2->addr;
    info.addr_magic = base_addr + (i * 64);
    info.addr_magic2 = info.addr_magic;

    VerbsEP *ep;
    connect_info *connect_buffer;
    std::tie(ep, connect_buffer) =
        get_client_ep_and_info(attr, &info, 16, false);

    info = *connect_buffer;
    uint64_t client_id = connect_buffer->server_id;

    RemoteBuffer *rb = new ReverseRemoteBuffer(info.ctx);

    // 4. Offset the local staging memory for the SGE metadata
    uint64_t local_mem = base_addr + (i * 64);
    uint32_t local_mem_lkey = mr2->rkey;

    receivers[client_id] = std::make_unique<CircularReverseReceiver>(lb);
    senders[client_id] = std::make_unique<CircularConnectionReverse>(
        ep, rb, local_mem, local_mem_lkey);
    eps[client_id] = ep;
    connect_buffers[client_id] = connect_buffer;
  }
}

std::pair<VerbsEP *, connect_info *>
RDMAManager::get_server_ep_and_info(int server_id, struct ibv_qp_init_attr attr,
                                    void *my_info, uint32_t recv_batch,
                                    bool recv_with_data) {
  std::string server_ip = address_list[server_id].first,
              server_port = address_list[server_id].second;

  struct rdma_cm_id *client_cm_id =
      send_connect_request(server_ip.c_str(), server_port.c_str());
  VerbsEP *ep = nullptr;
  struct ibv_pd *pd = client_cm_id->pd;

  uint32_t max_recv_size = attr.cap.max_recv_wr;
  if (attr.srq)
    attr.cap.max_recv_wr = 0;

  if (rdma_create_qp(client_cm_id, pd, &attr)) {
    perror("rdma_create_qp");
    exit(1);
  }

  client_cm_id->context = (void *)(uintptr_t)server_id;

  attr.cap.max_recv_wr = max_recv_size;
  ep = new VerbsEP(client_cm_id, attr, recv_batch, recv_with_data);
  // attr.cap.max_inline_data, attr.cap.max_send_wr, max_recv_size, , attr.srq
  // );

  struct rdma_conn_param conn_param;
  memset(&conn_param, 0, sizeof(conn_param));
  conn_param.responder_resources = 16; // up to 16 reads

  conn_param.initiator_depth = 16;
  conn_param.retry_count = 3;
  conn_param.rnr_retry_count = 3;
  conn_param.private_data = my_info;
  conn_param.private_data_len = sizeof(connect_info);

  if (rdma_connect(client_cm_id, &conn_param)) {
    printf(" failed to accept\n");
    exit(1);
  }

  struct rdma_cm_event *event;

  if (rdma_get_cm_event(client_cm_id->channel, &event)) {
    perror("rdma_get_cm_event");
    exit(1);
  }

  if (event->event != RDMA_CM_EVENT_ESTABLISHED) {
    perror("event not RDMA_CM_EVENT_ESTABLISHED");
    exit(1);
  }

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

void RDMAManager::send_connections(int starting_server_id, int num_servers,
                                   int max_send_size, int max_recv_size) {
  struct ibv_qp_init_attr attr =
      prepare_qp(this->getPD(), max_send_size, max_recv_size, false);

  size_t mem2_size = 64 * num_servers;
  char *mem2 = (char *)aligned_alloc(4096, 4096);
  memset(mem2, 0, mem2_size);
  struct ibv_mr * mr2 = ibv_reg_mr(this->getPD(), mem2, 4096,
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
    Buffer *lb = new ReverseRingBuffer(mr, log2(RING_BUFFER_SIZE));
    BufferContext bc = lb->GetContext();

    connect_info info = {};
    info.code = 4;
    info.ctx = bc;
    info.rkey_magic = mr2->rkey;
    info.server_id = my_id;

    uint64_t base_addr = (uint64_t)mr2->addr;
    info.addr_magic = base_addr + (i * 64);
    info.addr_magic2 = info.addr_magic;

    VerbsEP *ep;
    connect_info *connect_buffer;

    std::tie(ep, connect_buffer) =
        get_server_ep_and_info(server_id, attr, &info, 16, false);

    info = *(connect_info *)connect_buffer;

    RemoteBuffer *rb = new ReverseRemoteBuffer(info.ctx);
    uint64_t local_mem = base_addr + (i * 64);
    uint32_t local_mem_lkey = mr2->rkey;

    receivers[server_id] = (std::make_unique<CircularReverseReceiver>(lb));
    senders[server_id] = std::make_unique<CircularConnectionReverse>(
        ep, rb, local_mem, local_mem_lkey);
    eps[server_id] = ep;
    connect_buffers[server_id] = connect_buffer;
  }
}

void RDMAManager::cleanup() {
  // based on https://github.com/animeshtrivedi/rdma-example
  // rdma_destroy_qp();
  int num_clients = eps.size();
  int ret = -1;
  for (int i = 0; i < num_clients; i++) {
    if (eps[i] == nullptr) continue;
    // destroy the qp
    struct rdma_cm_id *client_cm_id = eps[i]->id;
    struct ibv_cq *client_send_cq = eps[i]->qp->send_cq;
    struct ibv_cq *client_recv_cq = eps[i]->qp->recv_cq;
    rdma_disconnect(client_cm_id);

    rdma_destroy_qp(client_cm_id);

    ret = rdma_destroy_id(client_cm_id);
    if (ret) {
      rdma_error("Failed to destroy client id cleanly, %d \n", -errno);
    }

    ret = ibv_destroy_cq(client_send_cq);
    if (ret) {
      rdma_error("Failed to destroy client send cq cleanly, %d \n", -errno);
    }

    ret = ibv_destroy_cq(client_recv_cq);
    if (ret) {
      rdma_error("Failed to destroy client rev cq cleanly, %d \n", -errno);
    }
    free(connect_buffers[i]);
    // we don't have a completion channel and just polls from the ring buffer so
    // don't need to destroy
    delete eps[i];
  }
  for (size_t i =0; i < local_mrs.size(); i++) {
    ibv_dereg_mr(local_mrs[i]);
    free(local_mems[i]);
  }
}
