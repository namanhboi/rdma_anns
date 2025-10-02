#include "zmq_sender_ack.hpp"
#include <chrono>
#include <thread>


int main(int argc, char **argv) {
  uint64_t my_id = std::stoull(argv[1]);
  std::string config_file(argv[2]);
  AckNode ack(my_id, config_file);
  ack.start();
  std::cout << "Press q to shutdown" << std::endl;
  std::string end;
  while (end != "q") {
    std::cin >> end;
  }
}


