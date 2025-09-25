#pragma once

#include <cstdint>
#include <memory>
/*
  TagT is the node id type
*/


using query_id_t = uint64_t;
using client_id_t = uint64_t;

template <typename TagT> struct result_t {
  query_id_t query_id;
  client_id_t client_id;
  uint32_t K;
  uint32_t L;
  std::shared_ptr<TagT[]> result_tags;
  std::shared_ptr<float[]> result_dists;
};
