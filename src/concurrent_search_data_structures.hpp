#include "neighbor.h"
#include <limits>
#include <mutex>
#include "serialize_utils.hpp"

/**
   end goal is to have both the search function and the push results function
   from another thread access the data structures in a safe way.

what i will do is to have a concurrent class for each retset, fullretset,
visited where all operations involve a mutex. These 3 classes have access to the
underlying data structures (neighbor pq, vector, tsl robin set) through
references + the mutex is also a reference.


*/




  
class ConcurrentNeighborPriorityQueue {
  std::mutex &query_mutex;
  diskann::NeighborPriorityQueue &nbr_queue;
public:
  ConcurrentNeighborPriorityQueue(diskann::NeighborPriorityQueue &nbr_queue,
                                  std::mutex &query_mutex)
  : nbr_queue(nbr_queue), query_mutex(query_mutex) {}

  void insert(const diskann::Neighbor &nbr) {
    std::scoped_lock lock(query_mutex);
    nbr_queue.insert(nbr);
  }
  
  void insert_nbrs(const std::vector<diskann::Neighbor> &nbrs) {
    std::scoped_lock lock(query_mutex);
    for (const auto &nbr : nbrs) {
      nbr_queue.insert(nbr);
    }
  }

  diskann::Neighbor closest_unexpanded() {
    std::scoped_lock lock(query_mutex);
    return nbr_queue.closest_unexpanded();
  }

  bool has_unexpanded_node() const {
    std::scoped_lock lock(query_mutex);
    return nbr_queue.has_unexpanded_node();
  }

  size_t size() const {
    std::scoped_lock lock(query_mutex);
    return nbr_queue.size();
  }
  size_t capacity() const {
    std::scoped_lock lock(query_mutex);
    return nbr_queue.capacity();
  }
  void reserve(size_t capacity) {
    std::scoped_lock lock(query_mutex);
    nbr_queue.reserve(capacity);
  }
  diskann::Neighbor operator[](size_t i) const {
    std::scoped_lock lock(query_mutex);
    return nbr_queue[i];
  }

  bool insert_nbrs(uint32_t num_neighbors, const uint32_t *nbr_ids,
                   const float *nbr_distances) {
    std::scoped_lock lock(query_mutex);
    for (auto i = 0; i < num_neighbors; i++) {
      diskann::Neighbor nbr(nbr_ids[i], nbr_distances[i]);
      nbr_queue.insert(nbr);
    }
    return true;
  }

  void clear() {
    std::scoped_lock lock(query_mutex);
    nbr_queue.clear();
  }    
};

/**
   used to store diskann::Neighbor with expanded distance 
*/
class ConcurrentResultVector {
  std::vector<diskann::Neighbor> &results;
  std::mutex &query_mutex;
public:
  ConcurrentResultVector(std::vector<diskann::Neighbor> &results,
                         std::mutex &query_mutex)
  : results(results), query_mutex(query_mutex) {}

  void push_back(const diskann::Neighbor &nbr) {
    std::scoped_lock l(query_mutex);
    results.push_back(nbr);
  }
  void sort_and_write(uint64_t *indices, float *distances, uint32_t K) {
    std::scoped_lock l(query_mutex);
    std::sort(results.begin(), results.end());
    for (uint64_t i = 0; i < K; i++) {
      indices[i] = results[i].id;
      if (distances != nullptr)
        distances[i] = results[i].distance;
    }
  }

  void clear() {
    std::scoped_lock l(query_mutex);
    results.clear();
  }    
};


class ConcurrentVisitedSet {
  tsl::robin_set<size_t> &visited;
  std::mutex &query_mutex;
public:
  ConcurrentVisitedSet(tsl::robin_set<size_t> &visited, std::mutex &query_mutex)
  : visited(visited), query_mutex(query_mutex) {}

  bool insert(size_t node_id) {
    std::scoped_lock l(query_mutex);
    return visited.insert(node_id).second;
  }
  
  void clear() {
    std::scoped_lock l(query_mutex);
    visited.clear();
  }
};

/*
  global search baseline: includes the concurrent candidate q, full result ret
  (expanded data)
  cotraversal: candidate queues, stop tokens, etc
*/
class ConcurrentSearchData {
  uint64_t query_id = std::numeric_limits<uint64_t>::max();
  std::mutex query_mutex;
  
  diskann::NeighborPriorityQueue retset;
  std::vector<diskann::Neighbor> full_retset;
  tsl::robin_set<size_t> visited;
  
public:
  ConcurrentSearchData(uint32_t visited_reserve) {
    full_retset.reserve(visited_reserve);
    visited.reserve(visited_reserve);
  }

  ConcurrentNeighborPriorityQueue get_retset() {
    return ConcurrentNeighborPriorityQueue(retset, query_mutex);
  }

  ConcurrentResultVector get_full_retset() {
    return ConcurrentResultVector(full_retset, query_mutex);
  }
  ConcurrentVisitedSet get_visited() {
    return ConcurrentVisitedSet(visited, query_mutex);
  }

  // only used after a query has finished
  void clear() {
    std::scoped_lock l(query_mutex);
    query_id = std::numeric_limits<uint64_t>::max();
    retset.clear();
    full_retset.clear();
    visited.clear();
  }

  void receive_result(
      std::shared_ptr<ComputeResult> compute_result,
		      std::unordered_map<uint32_t, uint32_t> &query_id_order_map) {
    std::scoped_lock l(query_mutex);
    if (compute_result->get_query_id() != query_id) {
      // std::cout << "outdated compute result, compute result id is "
                // << query_id_order_map[compute_result->get_query_id()] << ", current query id is "
      // << query_id_order_map[query_id] << std::endl;
      return;
    }
    std::cout << "compute result not outdated "
              << compute_result->get_query_id() << ", current query id is "
    << query_id << std::endl;
    auto nbr_ids = compute_result->get_nbr_ids();
    auto nbr_distances = compute_result->get_nbr_distances();
    for (auto i = 0; i < compute_result->get_num_neighbors(); i++) {
      retset.insert({nbr_ids[i], nbr_distances[i]});
      visited.insert(nbr_ids[i]);
    }
    full_retset.push_back(
			  {compute_result->get_node_id(), compute_result->get_expanded_dist()});
  }

  void start_new_query(uint64_t query_id) {
    std::scoped_lock l(query_mutex);
    this->query_id = query_id;
    retset.clear();
    full_retset.clear();
    visited.clear();
  }
};


