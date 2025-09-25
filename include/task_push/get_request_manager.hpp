#pragma once

#include <cascade/object.hpp>
#include <cascade/service_types.hpp>
#include <list>
#include "serialize_utils.hpp"


template <typename T, typename ObjectType> class GetRequestVector {
  std::vector<std::pair<uint64_t, derecho::rpc::QueryResults<const ObjectType>>>
      _get_requests;
  derecho::cascade::DefaultCascadeContextType *typed_ctxt;
public:
  GetRequestVector(derecho::cascade::DefaultCascadeContextType *typed_ctxt,
                   uint32_t size_hint): typed_ctxt(typed_ctxt) {
    _get_requests.reserve(size_hint);
  }
  
  void submit_request(uint64_t id, const std::string &key) {
    _get_requests.emplace_back(
			       id, typed_ctxt->get_service_client_ref().get(key, CURRENT_VERSION, true));
  }

  std::vector<uint64_t> get_requests_ids() {
    std::vector<uint64_t> ids;
    for (auto &[id, _] : _get_requests) {
      ids.emplace_back(id);
    }
    return ids;
  }

  // this clears all get requests
  std::vector<std::pair<uint64_t, std::shared_ptr<const T>>> get_results() {
    std::vector<std::pair<uint64_t, std::shared_ptr<const T>>> results;
    for (auto i = 0; i < _get_requests.size(); i++) {
      auto &reply = _get_requests[i].second.get().begin()->second.get();
      derecho::cascade::Blob blob =
        std::move(const_cast<derecho::cascade::Blob &>(reply.blob));
      blob.memory_mode = derecho::cascade::object_memory_mode_t::
            EMPLACED; // need to do this i think so that the destructor for blob
      // won't clean up the memory we want
      std::shared_ptr<const T> data(reinterpret_cast<const T *>(blob.bytes),
                                    free_const);
      results.emplace_back(_get_requests[i].first, data);
    }
    _get_requests.clear();
    return results;
  }
  size_t size() const{
    return _get_requests.size();
  }    
};  




// this is somewhat uncessarily complicated if we don't need to remove a
// completed request and instead just wait for  all requests, in that case just
// use std::vector to take advantage of locality
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
