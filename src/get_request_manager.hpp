#pragma once

#include <cascade/object.hpp>
#include <list>
#include "serialize_utils.hpp"


template <typename T, typename ObjectType > class GetRequestManager {
private:
  std::list<std::pair<uint64_t, derecho::rpc::QueryResults<const ObjectType>>>
      _get_requests;

public:
  GetRequestManager() {}


  void submit_request(
		      uint64_t id,
      derecho::rpc::QueryResults<const ObjectType>
          request) {
    _get_requests.emplace_back(id, std::move(request));
  }


  std::pair<std::vector<uint64_t>, std::vector<std::shared_ptr<const T>>> get_available_requests() {
    std::vector<uint64_t> results_id;
    std::vector<std::shared_ptr<const T>> results_data;
    
    for (auto it = _get_requests.begin(); it != _get_requests.end();) {
      uint64_t id = it->first;
      auto &request = it->second;
      if (request.is_ready()) {
        auto &reply = request.get().begin()->second.get();
        derecho::cascade::Blob blob =
          std::move(const_cast<derecho::cascade::Blob &>(reply.blob));
        blob.memory_mode = derecho::cascade::object_memory_mode_t::
            EMPLACED; // need to do this i think so that the destructor for blob
        // won't clean up the memory we want
        std::shared_ptr<const T> data(reinterpret_cast<const T *>(blob.bytes),
                                      free_const);
        results_data.emplace_back(std::move(data));
        results_id.emplace_back(id);
        // get data from request then delete
	it = _get_requests.erase(it);
      } else {
	it++;
      }
    }
    return {std::move(results_id), std::move(results_data)};
    
  }

  std::pair<std::vector<uint64_t>, std::vector<std::shared_ptr<const T>>>
  get_all_requests() {
    std::vector<uint64_t> results_id;
    std::vector<std::shared_ptr<const T>> results_data;
    for (auto it = _get_requests.begin(); it != _get_requests.end();) {
      uint64_t id = it->first;
      auto &request = it->second;
      auto &reply = request.get().begin()->second.get();
      derecho::cascade::Blob blob =
        std::move(const_cast<derecho::cascade::Blob &>(reply.blob));
      blob.memory_mode = derecho::cascade::object_memory_mode_t::
            EMPLACED; // need to do this i think so that the destructor for blob
      // won't clean up the memory we want
      std::shared_ptr<const T> data(reinterpret_cast<const T *>(blob.bytes),
                                    free_const);
      results_data.emplace_back(std::move(data));
      results_id.emplace_back(id);
      // get data from request then delete
      it = _get_requests.erase(it);
    }
    return {std::move(results_id), std::move(results_data)};
    
  }
  
};
