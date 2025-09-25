#include "serialize_utils.hpp"
#include "udl_path_and_index.hpp"
#include <limits>


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

std::pair<uint8_t, uint64_t>
parse_cluster_and_batch_id(const std::string &key_string) {
  std::string cluster_prefix = "cluster";
  uint8_t cluster_id = static_cast<uint8_t>(
					    std::stoll(key_string.substr(cluster_prefix.size())));
  std::string cluster_id_str = std::to_string(cluster_id);

  size_t pos = key_string.find("_");
  uint64_t batch_id;
  if (pos != std::string::npos) {
     batch_id = std::stoull(key_string.substr(pos + 1));
  } else {
    batch_id = std::numeric_limits<uint64_t>::max();
  }
  return std::make_pair(cluster_id, batch_id);
}
