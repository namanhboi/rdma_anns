#include "serialize_utils.hpp"
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <cascade/service_client_api.hpp>
#include "udl_path_and_index.hpp"
#include <random>

namespace po = boost::program_options;
namespace derecho {
namespace cascade {

  
void benchmark(int num_clusters, uint64_t num_bytes, uint64_t num_msg) {
  ServiceClientAPI &capi = ServiceClientAPI::get_service_client();

  uint64_t client_node_id = capi.get_my_id();
  std::random_device rd;
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<uint64_t> uint64_t_gen(
						       0, std::numeric_limits<uint64_t>::max());
  for (uint8_t cluster_sender_id = 0; cluster_sender_id < num_clusters;
       cluster_sender_id++) {
    send_query_t query = {client_node_id, num_bytes, num_msg, cluster_sender_id,
                          static_cast<uint8_t>(num_clusters)};
    std::shared_ptr<Blob> blob = query.get_blob();
    ObjectWithStringKey obj;
    obj.blob = std::move(*blob);
    obj.key = UDL_PATHNAME_CLUSTER + std::to_string(cluster_sender_id) + "_" +
              std::to_string(uint64_t_gen(gen)) + "_query";
    capi.trigger_put<UDL_OBJ_POOL_TYPE>(
					obj, UDL_SUBGROUP_INDEX, static_cast<uint32_t>(cluster_sender_id));
  }
}

} // namespace cascade
} // namespace derecho



int main(int argc, char **argv) {
  po::options_description desc("Options here mate");
  int num_clusters;
  uint64_t num_bytes, num_msg;

  desc.add_options()("help,h", "show help message")(
      "num_clusters", po::value<int>(&num_clusters)->required(),
      "number of clusters")("num_msg",
                            po::value<uint64_t>(&num_msg)->required(),
                            "num send_requests")(
      "num_bytes", po::value<uint64_t>(&num_bytes)->required(),
						 "max bytes in send request");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);
  derecho::cascade::benchmark(num_clusters, num_bytes, num_msg);
  return 0;
}

