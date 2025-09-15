#include "serialize_utils.hpp"
#include "udl_path_and_index.hpp"
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <cascade/service_client_api.hpp>
#include <random>

namespace po = boost::program_options;
namespace derecho {
namespace cascade {

void benchmark(uint64_t num_bytes, uint64_t num_msg) {
  ServiceClientAPI &capi = ServiceClientAPI::get_service_client();
  std::random_device rd;
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<uint64_t> uint64_t_gen(
      0, std::numeric_limits<uint64_t>::max());
  int cluster_sender_id = 0;
  auto client_id = capi.get_my_id();

  for (auto i = 0; i < num_msg; i++) {
    std::shared_ptr<Blob> blob = std::make_shared<derecho::cascade::Blob>(
        [num_bytes](uint8_t *buffer, size_t size) {
          uint8_t *tmp = new uint8_t[num_bytes];
          std::memcpy(buffer, tmp, num_bytes);
          delete[] tmp;
          return size;
        },
        num_bytes);
    uint64_t batch_id = uint64_t_gen(gen);
    ObjectWithStringKey obj;
    obj.blob = std::move(*blob);
    obj.key = UDL_PATHNAME "/" + std::to_string(client_id) + "_" +
              std::to_string(batch_id);
    TimestampLogger::log(LOG_CLIENT_SEND_START, client_id, batch_id,
                         obj.blob.size);
    
    capi.trigger_put<UDL_OBJ_POOL_TYPE>(
					obj, UDL_SUBGROUP_INDEX, static_cast<uint32_t>(cluster_sender_id));
  }

  std::cout << "Done sending messages" << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(5));
  TimestampLogger::flush("client" + std::to_string(client_id) + ".dat");
  
  auto shards = capi.get_subgroup_members<VolatileCascadeStoreWithStringKey>(
      UDL_SUBGROUP_INDEX);
  uint32_t shard_id = 0;
  for (auto &shard : shards) {
    ObjectWithStringKey obj;
    obj.key = UDL_PATHNAME "/flush_logs";
    for (int j = 0; j < shard.size(); j++) {
      auto res = capi.trigger_put<VolatileCascadeStoreWithStringKey>(
          obj, UDL_SUBGROUP_INDEX, shard_id);
    }
    shard_id++;
  }
}

} // namespace cascade
} // namespace derecho

int main(int argc, char **argv) {
  po::options_description desc("Options here mate");
  uint64_t num_bytes, num_msg;

  desc.add_options()("help,h", "show help message")(
      "num_msg", po::value<uint64_t>(&num_msg)->required(),
      "num send_requests")("num_bytes",
                           po::value<uint64_t>(&num_bytes)->required(),
                           "max bytes in send request");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);
  derecho::cascade::benchmark(num_bytes, num_msg);
  return 0;
}
