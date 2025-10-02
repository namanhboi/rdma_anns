#include "communicator.h"
#include <thread>



/**
   test: send the messages round robin to other peers in the network
*/
int main(int argc, char **argv) {
  uint64_t my_id = std::stoul(argv[1]);
  std::string config_file(argv[2]);
  uint32_t num_msg = std::stoul(argv[3]);
  uint32_t msg_size = std::stoul(argv[4]);
  ZMQP2PCommunicator server(my_id, config_file);
  std::atomic<int> num_received{0};
  server.register_receive_handler(
				  [&num_received](const char *data, size_t len) { num_received++; });
  std::cout << "number of peers " << server.get_num_peers() << std::endl;
  server.start_recv_loop();
  for (size_t i = 0; i < num_msg; i++) {
    for (uint64_t peer_id = 0; peer_id < server.get_num_peers(); peer_id++) {
      if (peer_id != server.get_my_id()) {
        char * arr = new char[msg_size];
        Region r = {.addr = arr, .length = msg_size, .context = 0, .lkey = 0};
	// std::cout << "Sending to peer id " << peer_id << std::endl;
        server.send_to_peer(peer_id, r);
      }
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(10));
  std::cout << num_received << std::endl;
}

