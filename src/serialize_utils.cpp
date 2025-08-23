#include "serialize_utils.hpp"
#include "udl_path_and_index.hpp"


uint8_t get_cluster_id(const std::string &key) {
  std::string cluster_prefix = "cluster";
  if (key.rfind(cluster_prefix, 0) != 0) {
    // doesn't start with the correct prefix
    throw std::invalid_argument(key + " doesn't have the correct prefix " +
                                cluster_prefix);
  }
  int num = std::stoul(key.substr(cluster_prefix.size()));
  if (num > 255) {
    throw std::invalid_argument("cluster id parsed from key " + key + " is bigger than uint8_t: " + std::to_string(num));
  }
  return static_cast<uint8_t>(num);
}

void free_const(const void* ptr) noexcept {
    free(const_cast<void*>(ptr));
}
std::pair<uint32_t, uint64_t>
parse_client_and_batch_id(const std::string &key_string) {
  size_t pos = key_string.find("_");
  uint32_t client_id = std::stoll(key_string.substr(0,pos));
  uint64_t batch_id = std::stoull(key_string.substr(pos+1));
  return std::make_pair(client_id,batch_id);
}
