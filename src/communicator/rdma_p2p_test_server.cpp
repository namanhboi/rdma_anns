#include "communicator.h"
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>

namespace po = boost::program_options;
int main(int argc, char **argv) {
  po::options_description desc("Options here mate");
  uint64_t server_peer_id;
  std::vector<std::string> address_list;
  desc.add_options()(
      "server_peer_id", po::value<uint64_t>(&server_peer_id)->required(),
      "Server peer ID")("address_list",
                        po::value<std::vector<std::string>>(&address_list)
                            ->multitoken()
                            ->required(),
                        "Address list");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);
  std::cout << "server_peer_id:" << server_peer_id << std::endl;
  std::cout << "address list of " << address_list.size() << " servers"
  << std::endl;
  for (int i = 0; i < address_list.size(); i++) {
    std::cout << address_list[i] << ",";
  }
  std::cout << std::endl;

  RDMARingBufferP2PCommunicator communicator(server_peer_id, address_list);


  return 0;
}
