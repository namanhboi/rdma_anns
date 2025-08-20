#include "neighbor.h"
#include <limits>
#include <mutex>

namespace diskann {
class ConcurrentNeighborPriorityQueue: NeighborPriorityQueue {
  mutable std::mutex nbr_queue_mutex;
  uint32_t query_id;

public:
  ConcurrentNeighborPriorityQueue()
  : NeighborPriorityQueue(), query_id(0) {}

  explicit ConcurrentNeighborPriorityQueue(uint32_t query_id)
  : NeighborPriorityQueue(), query_id(query_id) {}

  explicit ConcurrentNeighborPriorityQueue(uint32_t query_id, size_t capacity)
  : NeighborPriorityQueue(capacity), query_id(query_id) {}

  void insert(const Neighbor &nbr) {
    std::scoped_lock lock(nbr_queue_mutex);
    NeighborPriorityQueue::insert(nbr);
  }
  
  void insert_nbrs(const std::vector<Neighbor> &nbrs) {
    std::scoped_lock lock(nbr_queue_mutex);
    for (const auto &nbr : nbrs) {
      NeighborPriorityQueue::insert(nbr);
    }
  }

  Neighbor closest_unexpanded() {
    std::scoped_lock lock(nbr_queue_mutex);
    return NeighborPriorityQueue::closest_unexpanded();
  }

  bool has_unexpanded_node() const {
    std::scoped_lock lock(nbr_queue_mutex);
    return NeighborPriorityQueue::has_unexpanded_node();
  }

  size_t size() const {
    std::scoped_lock lock(nbr_queue_mutex);
    return NeighborPriorityQueue::size();
  }
  size_t capacity() const {
    std::scoped_lock lock(nbr_queue_mutex);
    return NeighborPriorityQueue::capacity();
  }
  void reserve(size_t capacity) {
    std::scoped_lock lock(nbr_queue_mutex);
    NeighborPriorityQueue::reserve(capacity);
  }
  Neighbor operator[](size_t i) const {
    std::scoped_lock lock(nbr_queue_mutex);
    return NeighborPriorityQueue::operator[](i);
  }
  void clear() {
    std::scoped_lock lock(nbr_queue_mutex);
    query_id = std::numeric_limits<uint32_t>::max();
    NeighborPriorityQueue::clear();
  }
  uint32_t get_query_id() const {
    std::scoped_lock lock(nbr_queue_mutex);
    return query_id;
  }

  /**checks that the query id matches internal query id before adding nbrs*/
  bool check_query_id_insert_nbrs(uint32_t query_id, uint32_t num_neighbors,
                                  const uint32_t* nbr_ids,
                                  const float* nbr_distances) {
    std::scoped_lock lock(nbr_queue_mutex);
    if (query_id != this->query_id) {
      return false;
    }
    for (auto i = 0; i < num_neighbors; i++) {
      diskann::Neighbor nbr(nbr_ids[i], nbr_distances[i]);
      NeighborPriorityQueue::insert(nbr);
    }
    return true;
  }

  /** used when starting a new query */
  void start_new_query(uint32_t query_id) {
    std::scoped_lock<std::mutex> lock(nbr_queue_mutex);
    this->query_id = query_id;
    NeighborPriorityQueue::clear();
  }
};

} // namespace diskann

