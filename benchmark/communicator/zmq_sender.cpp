#include "zmq_sender_ack.hpp"

int main(int argc, char **argv) {
  uint64_t my_id = std::stoull(argv[1]);
  std::string config_file(argv[2]);
  uint32_t num_msg = std::stoul(argv[3]);
  uint32_t num_warmup_msg = std::stoul(argv[4]);  
  uint32_t msg_size = std::stoul(argv[5]);
  SenderNode sender(my_id, config_file, num_msg, msg_size);
  sender.start_recv_thread();

  sender.blocking_send_msgs(num_warmup_msg);
  
  sender.blocking_wait_all_acks();
}

