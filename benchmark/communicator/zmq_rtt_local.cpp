#include "communicator.h"
#include <chrono>
#include <thread>
#include "zmq_sender_ack.hpp"

int main(int argc, char **argv) {
  // the first num_ack_nodes addresses in json file are for ack nodes
  // the last address specified in json is the sender node address, 
  uint64_t num_ack_nodes = std::stoull(argv[1]); 
  std::string config_file(argv[2]);
  uint32_t num_msg = std::stoul(argv[3]);
  uint32_t msg_size = std::stoul(argv[4]);
  std::vector<std::unique_ptr<AckNode>> ack_nodes;
  for (uint64_t i = 0; i < num_ack_nodes; i++) {
    ack_nodes.emplace_back(std::make_unique<AckNode>(i, config_file));
  }
  for (uint64_t i = 0; i < num_ack_nodes; i++) {
    ack_nodes[i]->start();
  }
  SenderNode sender_node(num_ack_nodes, config_file, num_msg, msg_size);
  sender_node.start_recv_thread();
  sender_node.blocking_send_msgs(0);
  sender_node.blocking_wait_all_acks();
}
