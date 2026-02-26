#include "communicator.h"
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <zmq.h>

#define CLIENT_MASK 1ll << 32

ZMQP2PCommunicator::ZMQP2PCommunicator(uint64_t my_id, const std::string &config_path)
: my_id(my_id) {
  parse_config(config_path);
  bind_and_connect_peers();
}



void ZMQP2PCommunicator::parse_config(const std::string &config_path) {
  std::ifstream f(config_path);
  json data = json::parse(f);
  auto address_list =
    data["address_list"].get<std::vector<std::string>>();
  for (size_t i = 0; i < address_list.size(); i++) {
    peer_id_to_address[i] = address_list[i];
  }
  my_address = address_list[my_id];
}


void ZMQP2PCommunicator::bind_and_connect_peers() {
  ctx = zmq_ctx_new();
  if (!ctx) {
    std::stringstream err;
    err << "Failed to create ctx : " << zmq_strerror(zmq_errno());
    throw std::runtime_error(err.str());
  }
  sock = zmq_socket(ctx, ZMQ_PEER);
  if (!sock) {
    std::stringstream err;
    err << "Failed to create socket : " << zmq_strerror(zmq_errno());
    throw std::runtime_error(err.str());
  }
  // Set 1-second timeout
  int timeout = 1000;
  zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
  
  std::cout << "my address is " << my_address << std::endl;
  int ret = zmq_bind(sock, my_address.c_str());
  if (ret != 0) {
    throw std::runtime_error("error binding socket to " + my_address +
                             zmq_strerror(zmq_errno()));
  }
  std::cout << "Done binding" << std::endl;

  for (const auto &[peer_id, address] : peer_id_to_address) {
    if (peer_id != my_id) {
      uint32_t routing_id = zmq_connect_peer(sock, address.c_str());
      if (routing_id == 0) {
        throw std::runtime_error("error connecting socket to " + address +
                                 zmq_strerror(zmq_errno()));
      }
      peer_to_routing_id[peer_id] = routing_id;
      std::cout << "Done connecting with " << address << std::endl;
    }
  }
  std::cout << "Done connecting to peers" << std::endl;
}


void ZMQP2PCommunicator::send_to_peer(uint64_t peer_id, Region *r) {
  zmq_msg_t msg;
  uint32_t length = r->length;
  int rc = zmq_msg_init_data(&msg, r->addr, r->length, Region::delete_addr,
                             (r->prealloc_queue == nullptr) ? nullptr : (void*)r);
  if (rc != 0) {
    std::stringstream err;
    err << __func__ << " error initializing msg to send to peer "
    << (int)peer_id << " : " << zmq_strerror(zmq_errno());
    throw std::runtime_error(err.str());
  }
  zmq_msg_set_routing_id (&msg, peer_to_routing_id[peer_id]);
  int num_bytes_sent = zmq_msg_send(&msg, sock, 0);
  if (num_bytes_sent == -1) {
    zmq_msg_close(&msg);
    std::stringstream err;
    err << __func__ << " error sending msg to partition " << (int)peer_id
    << " : " << zmq_strerror(zmq_errno());
    throw std::runtime_error(err.str());
  }
  if (num_bytes_sent != length) {
    std::stringstream err;
    err << __func__ << "num bytes sent " << num_bytes_sent
    << " different from length of msg " << length;
    throw std::runtime_error(err.str());
  }
}

void ZMQP2PCommunicator::recv_loop() {
  while (running) {
    zmq_msg_t msg;
    int rc = zmq_msg_init(&msg);
    assert(rc == 0);
    int num_bytes_recv = zmq_msg_recv(&msg, sock, 0);
    if (num_bytes_recv < 0) {
      zmq_msg_close(&msg);
      if (zmq_errno() == EAGAIN) {
        //timeout
	continue;
      }
      if (!running) {
	break;
      }
      std::stringstream ss;
      ss << __func__ << " error when receiving bytes " << zmq_strerror(zmq_errno());
      throw std::runtime_error(ss.str());
    }
    if (recv_handler) {
      this->recv_handler(reinterpret_cast<char *>(zmq_msg_data(&msg)),
                         zmq_msg_size(&msg));
    }
    zmq_msg_close(&msg);
  }
}

void ZMQP2PCommunicator::start_recv_thread() {
  running = true;
  real_thread = std::thread(&ZMQP2PCommunicator::recv_loop, this);
}

void ZMQP2PCommunicator::stop_recv_thread() {
  running = false;
  if (sock) {
    zmq_close(sock);
    sock = nullptr;
  }
  if (real_thread.joinable())
    real_thread.join();
}

void ZMQP2PCommunicator::register_receive_handler(recv_handler_t handler) {
  this->recv_handler = handler;
}

ZMQP2PCommunicator::~ZMQP2PCommunicator() {
  stop_recv_thread();
  zmq_ctx_destroy(ctx);
}

uint64_t ZMQP2PCommunicator::get_my_id() { return this->my_id; }
uint64_t ZMQP2PCommunicator::get_num_peers() {
  return this->peer_to_routing_id.size() + 1;
}

std::vector<uint64_t> ZMQP2PCommunicator::get_other_peer_ids() {
  std::vector<uint64_t> peer_ids;
  uint64_t num_peers = get_num_peers();
  for (uint64_t i = 0; i < num_peers; i++) {
    if (i != my_id) peer_ids.push_back(i);
  }
  return peer_ids;
}

ZMQP2PCommunicator::ZMQP2PCommunicator(
				       uint64_t my_peer_id, const std::vector<std::string> &address_list): my_id(my_peer_id) {
  for (size_t i = 0; i < address_list.size(); i++) {
    peer_id_to_address[i] = address_list[i];
    std::cout <<peer_id_to_address[i] << std::endl;
  }
  my_address = address_list[my_id];
  std::cout << my_address << std::endl;
  bind_and_connect_peers();
}



