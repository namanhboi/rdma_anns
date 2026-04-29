#include "communicator.h"
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>

namespace po = boost::program_options;
int main(int argc, char **argv) {
  po::options_description desc("Options here mate");
  uint64_t server_peer_id;
  std::vector<std::string> address_list;
  uint32_t num_msg;
  uint32_t msg_size;

  desc.add_options()("server_peer_id",
                     po::value<uint64_t>(&server_peer_id)->required(),
                     "Server peer ID")(
      "address_list",
      po::value<std::vector<std::string>>(&address_list)
          ->multitoken()
          ->required(),
      "Address list")("num_msg", po::value<uint32_t>(&num_msg)->required(),
                      "num msg to send between the servers")(
      "msg_size", po::value<uint32_t>(&msg_size)->required(), "msg_size");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);
  if (msg_size > Region::MAX_BYTES_REGION) {
    std::cerr << "Error: msg_size exceeds MAX_BYTES_REGION!" << std::endl;
    return 1;
  }
  std::cout << "server_peer_id:" << server_peer_id << std::endl;
  std::cout << "address list of " << address_list.size() << " servers"
            << std::endl;
  for (int i = 0; i < address_list.size(); i++) {
    std::cout << address_list[i] << ",";
  }
  std::cout << std::endl;

  RDMARingBufferP2PCommunicator communicator(server_peer_id, address_list);
  std::atomic<int> num_received{0};

  communicator.register_receive_handler(
      [&num_received](const char *data, size_t len) {
        num_received++;
        if (num_received % 10000 == 0) {
          std::cout << "received: " << num_received << std::endl;
        }
      });

  // need to make the prealloc queue fuckkkkk mateee
  PreallocatedQueue<Region> prealloc_region_queue(
                                                  1000, Region::reset);
  std::pair<char *, uint32_t> ptr_lkey =
      communicator.get_preallocated_region_ptr_lkey(
                                                    Region::MAX_BYTES_REGION, 1000);
  char *region_addr = ptr_lkey.first;
  uint32_t lkey = ptr_lkey.second;
  prealloc_region_queue.assign_additional_block_mr(
                                                   region_addr, lkey, Region::MAX_BYTES_REGION, Region::assign_addr);

  communicator.start_recv_thread();
  for (size_t i = 0; i < num_msg; i++) {
    for (uint64_t peer_id = 0; peer_id < communicator.get_num_peers(); peer_id++) {
      if (peer_id != communicator.get_my_id()) {
        // char * arr = new char[msg_size];
        Region *r;
        prealloc_region_queue.dequeue_exact(1, &r);
        r->length = msg_size;
        communicator.send_to_peer(peer_id, r);
      }
    }
  }

  std::string shutdown;

  while (shutdown != "q") {
    std::cin >> shutdown;
  }


  return 0;
}
