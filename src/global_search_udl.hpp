#include <cascade/object.hpp>
#include <chrono>
#include <immintrin.h> // needed to include this to make sure that the code compiles since in DiskANN/include/utils.h it uses this library.
#include "in_mem_data_store.h"
#include "in_mem_graph_store.h"
#include "index.h"
#include "serialize_utils.hpp"
#include <cascade/cascade_interface.hpp>
#include <cascade/service_types.hpp>
#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include "concurrent_queue.h"
#include "neighbor.h"
#include <stdexcept>
#include "tsl/robin_set.h"
#include <boost/dynamic_bitset.hpp>
#include "udl_path_and_index.hpp"
#include "get_request_manager.hpp"

#define MAX_POINTS_FOR_USING_BITSET 10000000

namespace derecho {
namespace cascade {

#define MY_UUID "837425c0-a204-435c-bf6f-00ae40c8039f"
#define MY_DESC                                                                \
  "UDL for searching the head index + pull push mode as described in cotra "   \
  "paper"
#define DATA_TYPE "float"

std::string get_uuid();

std::string get_description();

template <typename data_type>
class GlobalSearchOCDPO : public DefaultOffCriticalDataPathObserver {
  class GlobalIndex {
    uint32_t num_points;

    //contains the mapping of all the nodes to clusters.
    std::vector<uint8_t>  cluster_assignment;

    uint32_t dim;
    uint8_t cluster_id;
    uint8_t num_clusters;
    GlobalSearchOCDPO
        *parent; // used to send distance compute queries to other servers.
  public:
    GlobalIndex(GlobalSearchOCDPO *parent,
                 uint32_t dim, uint8_t cluster_id) {
      this->parent = parent;
      this->cluster_id = cluster_id;
      std::string cluster_assignment_file =
        parent->cluster_assignment_bin_file;

      std::ifstream in(cluster_assignment_file, std::ios::binary);
      in.read((char *)&num_points, sizeof(num_points));
      in.read((char *)&num_clusters, sizeof(num_clusters));
      in.read((char *)cluster_assignment.data(), sizeof(uint8_t) * num_points);
    }

    bool is_in_cluster(uint32_t node_id) const {
      return cluster_assignment[node_id] == this->cluster_id;
    }

    


    /** get neighbor of a node in the cluster
        returns shared ptr with deleter free_const pointing to the number of neighbors (first uint32_t) and the neighbors of the requested node (rest of the uint32_ts)
	this method is only for retrieving neighbors of nodes that are in this cluster.
       */
    std::shared_ptr<const uint32_t>
    get_neighbors(uint32_t node_id,
                  DefaultCascadeContextType *typed_ctxt) const {
      if (!is_in_cluster(node_id))
	throw std::runtime_error(
				 "Node id " + std::to_string(node_id) + " not in cluster " +
				 std::to_string(this->cluster_id) + " " + __PRETTY_FUNCTION__);
    
      std::string nbr_key = parent->cluster_data_prefix + "_nbr_" + std::to_string(node_id);
      auto nbr = typed_ctxt->get_service_client_ref().get(nbr_key,
                                                          CURRENT_VERSION, true);
      auto &nbr_reply = nbr.get().begin()->second.get();
      Blob nbr_blob = std::move(const_cast<Blob &>(nbr_reply.blob));
      nbr_blob.memory_mode =
	object_memory_mode_t::EMPLACED; 
      uint32_t num_nbrs = nbr_blob.size / sizeof(uint32_t);
      std::shared_ptr<const uint32_t> nbr_ptr(
					reinterpret_cast<const uint32_t *>(nbr_blob.bytes), free_const);
      
      return nbr_ptr;
    }

    // these shared_ptrs must have free_const as deleter
    /*
      this method is only for retrieving embeddings that are in this cluster.
     */
    std::vector<std::shared_ptr<const data_type>>
    retrieve_local_embeddings(const std::vector<uint32_t> &node_ids, DefaultCascadeContextType *typed_ctxt) const {
      std::vector<std::shared_ptr<const data_type>> embeddings;
      for (size_t i = 0; i < node_ids.size(); i++) {
	uint32_t node_id = node_ids[i];
	if (!is_in_cluster(node_id))
          throw std::runtime_error(
				   "Node id " + std::to_string(node_id) + " not in cluster " +
				   std::to_string(this->cluster_id) + " " + __PRETTY_FUNCTION__ + " " + __FILE__);
            
	std::string emb_key = parent->cluster_data_prefix + "_emb_" + std::to_string(node_id);
	auto emb = typed_ctxt->get_service_client_ref().get(emb_key,
                                                            CURRENT_VERSION, true);
	auto &emb_reply = emb.get().begin()->second.get();
	Blob emb_blob = std::move(const_cast<Blob &>(emb_reply.blob));
        emb_blob.memory_mode =
            object_memory_mode_t::EMPLACED; // we now manage this object's
        // memory, not cascade
        std::shared_ptr<const data_type> embedding(
					     reinterpret_cast<const data_type*>(emb_blob.bytes), free_const);
        embeddings.emplace_back(std::move(embedding));
      }
      return embeddings;
    }

    
    
    std::pair<uint32_t, uint32_t>
    search_global_baseline(DefaultCascadeContextType *typed_ctxt,
                           std::shared_ptr<GreedySearchQuery<data_type>> query,
                           const uint32_t K, const uint32_t L, uint32_t *indices,
                           float *distances) {
      if (!query)
        throw std::invalid_argument("Query cannot be null");
      if (!indices)
        throw std::invalid_argument("Indices array cannot be null");

      if (L == 0 || K == 0 || K > L) {
        throw std::invalid_argument("Invalid L/K parameters");
      }
    
      if (query->get_candidate_queue_size() == 0) {
	throw std::invalid_argument("query " + std::to_string(query->get_query_id()) + " has candidate queue empty");
      }    

      diskann::NeighborPriorityQueue candidate_queue;
      candidate_queue.reserve(L);
      tsl::robin_set<uint32_t> inserted_into_pool_rs;
      boost::dynamic_bitset<> inserted_into_pool_bs;

      bool fast_iterate = this->num_points <= MAX_POINTS_FOR_USING_BITSET;

      if (fast_iterate)
	{
          if (inserted_into_pool_bs.size() < num_points)
            {
              // hopefully using 2X will reduce the number of allocations.
              auto resize_size =
                2 * num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 * num_points;
              inserted_into_pool_bs.resize(resize_size);
            }
	}

      // Lambda to determine if a node has been visited
      auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
        return fast_iterate ? inserted_into_pool_bs[id] == 0
               : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
      };
      auto mark_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](uint32_t node_id) {
	if (fast_iterate) {
          inserted_into_pool_bs[node_id] = 1;
	} else {
          inserted_into_pool_rs.insert(node_id);
	}
      };

      // Lambda to batch compute query<-> node distances
      auto compute_dists = [this, &query](const std::vector<std::shared_ptr<const data_type>> &embs,
					 std::vector<float> &dists_out					 ) {
	for (size_t i = 0; i < embs.size(); i++) {
          dists_out.push_back(parent->dist_fn->compare(
						       query->get_embedding_ptr(), embs[i].get(), this->dim));
	}
      };
      std::vector<uint32_t> init_node_ids(query->get_candidate_queue_ptr(),
                                          query->get_candidate_queue_ptr() +
                                          query->get_candidate_queue_size());

      std::vector<std::shared_ptr<const data_type>> init_embs =
        retrieve_local_embeddings(init_node_ids, typed_ctxt);
      std::vector<float> init_distances;

      compute_dists(init_embs, init_distances);
      for (uint32_t i = 0; i < query->get_candidate_queue_size(); i++) {
	diskann::Neighbor nbr(query->get_candidate_queue_ptr()[i], init_distances[i]);
	candidate_queue.insert(nbr);
      }

      uint32_t hops = 0;
      uint32_t cmps = 0;

      //used during search
      std::vector<uint32_t> primary_node_ids;
      std::vector<float> primary_dist;

      std::vector<float> secondary_dist;

      GetRequestManager<float, derecho::cascade::ObjectWithStringKey>
      emb_get_request_manager;
      

      // GetRequestManager<uint32_t,
                        // decltype(typed_ctxt->get_service_client_ref().get)>
      // nbr_get_request_manager;

      while (candidate_queue.has_unexpanded_node()) {
	diskann::Neighbor node =  candidate_queue.closest_unexpanded();
	uint32_t node_id = node.id;
	hops++;
	primary_node_ids.clear();
        primary_dist.clear();

        secondary_dist.clear();

	if (!is_in_cluster(node_id)) {
          throw std::runtime_error(
				   "Node id " + std::to_string(node_id) + " not in cluster " +
				   std::to_string(this->cluster_id) + " " + __PRETTY_FUNCTION__ + " " +
				   __FILE__);
	}

        std::shared_ptr<const uint32_t> neighbors =
          get_neighbors(node_id, typed_ctxt);
        uint32_t num_nbrs = neighbors.get()[0];
        const uint32_t *neighbors_ptr = neighbors.get() + 1;
	for (size_t i = 0; i < num_nbrs; i++) {
          uint32_t nbr_node_id = neighbors_ptr[i];
	  uint8_t nbr_cluster_id = cluster_assignment[nbr_node_id];
          if (nbr_cluster_id == this->cluster_id) {
            if (is_not_visited(nbr_node_id)) {
              primary_node_ids.push_back(nbr_node_id);
            }
          } else {
            // need to get the neighbors embeddings
            const std::string &emb_key = UDL2_DATA_PREFIX "/cluster_" +
                                         std::to_string(nbr_cluster_id) +
                                         "_emb_" + std::to_string(nbr_node_id);
            emb_get_request_manager.submit_request(
                nbr_node_id, typed_ctxt->get_service_client_ref().get(
								      emb_key, CURRENT_VERSION, true));
          }
        }

        // Mark nodes visited
        for (auto id : primary_node_ids) {
	  mark_visited(id);
        }

        std::vector<std::shared_ptr<const data_type>> embs =
          retrieve_local_embeddings(primary_node_ids, typed_ctxt);
	primary_dist.resize(embs.size());
        compute_dists(embs, primary_dist);
        cmps += primary_dist.size();
        for (size_t m = 0; m < primary_node_ids.size(); m++) {
	  candidate_queue.insert(diskann::Neighbor(primary_node_ids[m], primary_dist[m]));
        }

        std::pair<std::vector<uint64_t>,
                  std::vector<std::shared_ptr<const data_type>>>
            data_secondary_partition =
              emb_get_request_manager.get_available_requests();
        const std::vector<uint64_t> id = data_secondary_partition.first;
        const std::vector<std::shared_ptr<const data_type>> embeddings = data_secondary_partition.second;
	
        secondary_dist.resize(embeddings.size());
        compute_dists(embeddings, secondary_dist);
        cmps += secondary_dist.size();
        for (size_t m = 0; m < embeddings.size(); m++) {
	  candidate_queue.insert(diskann::Neighbor(id[m], secondary_dist[m]));
        }
                        
      }
      for (size_t i = 0; i < std::min(K, (uint32_t)candidate_queue.size()); i++) {
        indices[i] = candidate_queue[i].id;
        if (distances != nullptr)
          distances[i] = candidate_queue[i].distance;
      }
      return std::make_pair(hops, cmps);

      
    }


    /**
       does task_push search on the global graph index.
       the server that is doing the search is called the primary partition while
       the servers doing distance compute tasks are called secondary partitions.

       If the current node in the candidate queue is stored in the local
       server's storage then explore their neighbors as usual. If the current
       node is not local then issue a distance computation request to the server
       with that node, which in turn will send back node_id, distance pairs
       to be added to the candidate queue as well as its neighbors.
       For this, we need to know which nodes are local and which are not. Also
       there should be a queue where the 2nd partitions can send back results which we
       will check upon each new iteration of the search loop.


     */
    std::pair<uint32_t, uint32_t> search_task_push(
        DefaultCascadeContextType *typed_ctxt,
        std::shared_ptr<GreedySearchQuery<data_type>> query,
        std::shared_ptr<diskann::ConcurrentQueue<ComputeResult>> compute_res_q,
        uint32_t thread_id, const uint32_t K, const uint32_t L,
					 uint32_t *indices, float *distances) const {
      // this can only contain nodes that are in the current cluster, while
      // expanding the nodes in here, if a neighbor of the node being expanded is
      // not in the cluster then send a compute request or a request to get its
      // data. (pull/push mode)
      if (!query) throw std::invalid_argument("Query cannot be null");
      if (!compute_res_q) throw std::invalid_argument("Compute queue cannot be null");
      if (!indices) throw std::invalid_argument("Indices array cannot be null");
      // if (!distances) throw std::invalid_argument("Distances array cannot be null");
    
      if (L == 0 || K == 0 || K > L) {
        throw std::invalid_argument("Invalid L/K parameters");
      }
    
      if (query->get_candidate_queue_size() == 0) {
	throw std::invalid_argument("query " + std::to_string(query->get_query_id()) + " has candidate queue empty");
      }    

      diskann::NeighborPriorityQueue candidate_queue;
      candidate_queue.reserve(L);
      tsl::robin_set<uint32_t> inserted_into_pool_rs;
      boost::dynamic_bitset<> inserted_into_pool_bs;

      bool fast_iterate = this->num_points <= MAX_POINTS_FOR_USING_BITSET;

      if (fast_iterate)
	{
          if (inserted_into_pool_bs.size() < num_points)
            {
              // hopefully using 2X will reduce the number of allocations.
              auto resize_size =
                2 * num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 * num_points;
              inserted_into_pool_bs.resize(resize_size);
            }
	}

      // Lambda to determine if a node has been visited
      auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
        return fast_iterate ? inserted_into_pool_bs[id] == 0
               : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
      };
      auto mark_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](uint32_t node_id) {
	if (fast_iterate) {
          inserted_into_pool_bs[node_id] = 1;
	} else {
          inserted_into_pool_rs.insert(node_id);
	}
      };

      // Lambda to batch compute query<-> node distances
      auto compute_dists = [this, query](std::vector<std::shared_ptr<const data_type>> embs,
					 std::vector<float> &dists_out					 ) {
	for (size_t i = 0; i < embs.size(); i++) {
          dists_out.push_back(parent->dist_fn->compare(
						       query->get_embedding_ptr(), embs[i].get(), this->dim));
	}
      };
      std::vector<uint32_t> init_node_ids(query->get_candidate_queue_ptr(),
                                          query->get_candidate_queue_ptr() +
                                          query->get_candidate_queue_size());

      std::vector<std::shared_ptr<const data_type>> init_embs =
        retrieve_local_embeddings(init_node_ids, typed_ctxt);
      std::vector<float> init_distances;

      compute_dists(init_embs, init_distances);
      for (uint32_t i = 0; i < query->get_candidate_queue_size(); i++) {
	diskann::Neighbor nbr(query->get_candidate_queue_ptr()[i], init_distances[i]);
	candidate_queue.insert(nbr);
      }

      uint32_t hops = 0;
      uint32_t cmps = 0;

      //used during search
      std::vector<uint32_t> primary_node_ids;
      std::vector<float> primary_dist;

      std::unordered_map<uint8_t, std::vector<uint32_t>> second_partition_nodes;

      while (candidate_queue.has_unexpanded_node()) {
	while (!compute_res_q->empty()) {
          const ComputeResult &res = compute_res_q->pop();
          if (res.get_cluster_sender_id() == 0 && res.get_cluster_receiver_id() == 0 &&
              res.get_node().id == 0 && res.get_node().distance == 0 && res.get_query_id() == 0 &&
              res.get_num_neighbors() == 0 &&
              res.get_neighbors_ptr() == nullptr) { // not popping an empty queue, check the
            // impl of concurrent queue
            continue;
          }
          candidate_queue.insert(res.get_node());
        }
	diskann::Neighbor node =  candidate_queue.closest_unexpanded();
	uint32_t node_id = node.id;
	hops++;
	primary_node_ids.clear();
        primary_dist.clear();

        second_partition_nodes.clear();
      
	if (!is_in_cluster(node_id)) {
          throw std::runtime_error(
				   "Node id " + std::to_string(node_id) + " not in cluster " +
				   std::to_string(this->cluster_id) + " " + __PRETTY_FUNCTION__ + " " +
				   __FILE__);
	}

        std::shared_ptr<const uint32_t> neighbors =
          get_neighbors(node_id, typed_ctxt);
        uint32_t num_nbrs = neighbors.get()[0];
        const uint32_t *neighbors_ptr = neighbors.get() + 1;
	for (size_t i = 0; i < num_nbrs; i++) {
          uint32_t nbr_node_id = neighbors_ptr[i];
	  uint8_t nbr_cluster_id = cluster_assignment[nbr_node_id];
          if (nbr_cluster_id == this->cluster_id) {
            if (is_not_visited(nbr_node_id)) {
              primary_node_ids.push_back(nbr_node_id);
            }
          } else {
            if (second_partition_nodes.count(nbr_cluster_id) == 0) {
              second_partition_nodes[nbr_cluster_id] = std::vector<uint32_t>();
            }
            second_partition_nodes[nbr_cluster_id].push_back(nbr_node_id);
	    mark_visited(nbr_node_id);
          }
        }

        // Mark nodes visited
        for (auto id : primary_node_ids) {
	  mark_visited(id);
        }

        std::vector<std::shared_ptr<const data_type>> embs =
          retrieve_local_embeddings(primary_node_ids, typed_ctxt);
	primary_dist.reserve(embs.size());
        compute_dists(embs, primary_dist);
        cmps += primary_dist.size();
        for (size_t m = 0; m < primary_node_ids.size(); m++) {
	  candidate_queue.insert(diskann::Neighbor(primary_node_ids[m], primary_dist[m]));
        }
        float min_distance = candidate_queue[candidate_queue.size() - 1].distance;

        for (auto &[cluster_id, node_ids] : second_partition_nodes) {
          for (uint32_t node_id : node_ids) {
            parent->batch_thread->push_compute_query(
                node_id, query->get_query_id(), min_distance, cluster_id,
						     thread_id);
	  }
        }
      }
      for (size_t i = 0; i < std::min(K, (uint32_t)candidate_queue.size()); i++) {
        indices[i] = candidate_queue[i].id;
        if (distances != nullptr)
          distances[i] = candidate_queue[i].distance;
      }
      return std::make_pair(hops, cmps);
    }
  };

  
  class SearchThread {
    uint64_t my_thread_id;
    GlobalSearchOCDPO *parent;
    std::thread real_thread;
    bool running = false;

    std::condition_variable_any query_queue_cv;
    std::mutex query_queue_mutex;
    std::queue<std::shared_ptr<GreedySearchQuery<data_type>>> query_queue;
    
    // query_id and the concurrent queue that accepts distance compute results
    std::unordered_map<uint32_t, std::shared_ptr<diskann::ConcurrentQueue<ComputeResult>>> compute_res_queues;
    std::mutex compute_res_queues_mtx;

    void main_loop(DefaultCascadeContextType *typed_ctxt) {
      std::unique_lock<std::mutex> lock(query_queue_mutex, std::defer_lock);
      while (running) {
        lock.lock();

        if (query_queue.empty()) {
	  query_queue_cv.wait(lock);
        }
        
        if (!running) 
          break;
	std::shared_ptr<GreedySearchQuery<data_type>> query = query_queue.front();
        query_queue.pop();

        uint32_t query_id = query->get_query_id();
        std::unique_lock<std::mutex> compute_res_lock(compute_res_queues_mtx);
	ComputeResult empty_res;
        compute_res_queues[query_id] =
          std::make_shared<diskann::ConcurrentQueue<ComputeResult>>(empty_res);
        compute_res_lock.unlock();
        lock.unlock();

        const uint32_t &K = query->get_K();
        const uint32_t &L = query->get_L();
        std::shared_ptr<uint32_t[]> result(new uint32_t[query->get_K()]);
#ifdef TASK_PUSH
        parent->index->search_task_push(
            typed_ctxt, query, compute_res_queues[query_id], this->my_thread_id,
					      K, L, result.get(), nullptr);
#elif GLOBAL_BASELINE
	        // parent->index->search_global_baseline(
            // typed_ctxt, query, compute_res_queues[query_id], this->my_thread_id,
					      // K, L, result.get(), nullptr);
#endif
        parent->batch_thread->push_ann_result(
            query->get_query_id(), query->get_client_node_id(), K, L,
					      std::move(result), parent->cluster_id);
      }
    }

  public:
    SearchThread(uint64_t thread_id, GlobalSearchOCDPO<data_type> *parent)
    : my_thread_id(thread_id), parent(parent) {}

    /**
push the compute result to the queue for that query id.
     */
    void push_compute_res(ComputeResult result) {
      uint32_t query_id = result.get_query_id();
      if (this->compute_res_queues.count(query_id) == 0) {
        throw std::runtime_error("Compute result for query id " +
                                 std::to_string(query_id) +
                                 " arrived but there is no matching compute "
                                 "result queue to put it into ");
      }
      this->compute_res_queues[query_id]->push(result);
      // this is fine since concurrent queue has internal locks
    }
    void push_search_query(std::shared_ptr<GreedySearchQuery<data_type>> search_query) {
      std::scoped_lock<std::mutex> lock(query_queue_mutex);
      query_queue.emplace(std::move(search_query));
      query_queue_cv.notify_all();
    }

    void start(DefaultCascadeContextType *typed_ctxt) {
      running = true;
      real_thread = std::thread(&GlobalSearchOCDPO::SearchThread::main_loop, this, typed_ctxt);
    }
    void join() {
      if (real_thread.joinable())
        real_thread.join();
    }
    void signal_stop() {
      std::scoped_lock l(query_queue_mutex);
      running = false;
      query_queue.push(nullptr);
      query_queue_cv.notify_all();
    }
  };

  // all queries embeddings needs to be sent to all nodes running global search
  // udl because it needs to compute the distance
  // then we have to think about how to overlap getting the data with
  // computation
  // Down the line, this won't be a thread but instead a co corountine running
  // with the main greedy search algo as described in the cotra paper:
  // 
  // The greedy search will explore 1 candidate then yield, then the distance
  // compute will execute its queries then yield,...

  class DistanceComputeThread {
    uint64_t thread_id;
    std::thread real_thread;
    
    // for a given compute query, if query embedding data hasn't arrived yet
    // then pop the query and reinsert it then move on the next query
    std::queue<compute_query_t> compute_queue;
    std::mutex compute_queue_mutex;
    std::condition_variable_any compute_queue_cv;
    bool running = false;

    std::unordered_map<uint32_t, std::shared_ptr<EmbeddingQuery<data_type>>>
        query_map;
    std::mutex query_map_mutex;

    GlobalSearchOCDPO<data_type> *parent;
    void main_loop(DefaultCascadeContextType *typed_ctxt) {
      // since this is the in memory version, just get the data and do compute
      // in the disk version (hopefully that's possible) we do all requests
      // first then get which ever one arrives
      std::unique_lock<std::mutex> queue_lock(compute_queue_mutex, std::defer_lock);
      std::unique_lock<std::mutex> map_lock(query_map, std::defer_lock);
      
      while (running) {
        // need to check if queue empty then waitfor 
	std::shared_ptr<EmbeddingQuery<data_type>> query_emb_ptr;
        queue_lock.lock();
        if (compute_queue.empty()) { // should this be while or if (ask alicia)
	  compute_queue_cv.wait(queue_lock);
        }
        if (!running)
          break;
        
        compute_query_t query = compute_queue.front();
        compute_queue.pop();
        // prolly could pop until the next query id available?

	map_lock.lock();
        if (query_map.count(query.query_id) == 0) {
          std::cerr << "[compute distance] " << "cluster " << parent->cluster_id << " query embedding for query "
          << query.query_id << " has not arrived" << std::endl;
          compute_queue.push(query);
        } else {
	  query_emb_ptr = query_map[query.query_id];
        }
        map_lock.unlock();
        queue_lock.unlock();
        if (!query_emb_ptr) {
	  continue;
        }
        // after this, the embedding for query has arrived
        if (query.cluster_receiver_id != parent->cluster_id) {
          throw std::invalid_argument(
              "query " + std::to_string(query.query_id) +
              " has cluster_reciever_id " +
              std::to_string(query.cluster_receiver_id) +
              " but this node has cluster id " +
              std::to_string(parent->cluster_id));
        }
        std::string emb_key =
          parent->cluster_data_prefix + "_emb_" + std::to_string(query.node_id);
        bool stable = true;
        auto emb_get_result =
          typed_ctxt->get_service_client_ref().get(emb_key, stable);

        auto &emb_reply = emb_get_result.get().begin()->second.get();
        Blob emb_blob = std::move(const_cast<Blob &>(emb_reply.blob));
        const data_type *vec_emb =
          reinterpret_cast<const data_type *>(emb_blob.bytes);

        float distance = parent->dist_fn->compare(
            vec_emb, query_emb_ptr->get_embedding_ptr(),
						  query_emb_ptr->get_dim());
        diskann::Neighbor node = {query.node_id, distance};

        std::string nbr_key =
          parent->cluster_data_prefix + "_nbr_" + std::to_string(query.node_id);
        auto nbr_get_result =
          typed_ctxt->get_service_client_ref().get(nbr_key, stable);
        auto &nbr_reply = nbr_get_result.get().begin()->second.get();
        Blob nbr_blob = std::move(const_cast<Blob &>(nbr_reply.blob));
        nbr_blob.memory_mode = object_memory_mode_t::EMPLACED;
        std::shared_ptr<const uint32_t> nbr_ptr(
						reinterpret_cast<const uint32_t *>(nbr_blob.bytes), free_const);
        
	uint32_t num_nbrs = *nbr_ptr.get();
        

        if (distance >= query.min_distance) {
          compute_result_t res(parent->cluster_id, query.cluster_sender_id,
                               query.receiver_thread_id, node, query.query_id,
                               num_nbrs, nbr_ptr);
          
          parent->batch_thread->push_compute_res(res);
        }
      }
    }

  public:
    DistanceComputeThread(uint64_t thread_id, GlobalSearchOCDPO *parent)
    : thread_id(thread_id), parent(parent) {}

    void start(DefaultCascadeContextType *typed_ctxt) {
      running = true;
      real_thread =
        std::thread(&DistanceComputeThread::main_loop, this, typed_ctxt);
    }

    void join() {
      if (real_thread.joinable()) {
        real_thread.join();
      }
    }

    void signal_stop() {
      std::scoped_lock<std::mutex> l(compute_queue_mutex);
      running = false;

      // compute_queue.push()
    }
    void push_compute_query(compute_query_t compute_query) {
      std::scoped_lock<std::mutex> lock(compute_queue_mutex);
      compute_queue.emplace(std::move(compute_query));
      compute_queue_cv.notify_all();
    }

    void push_embedding_query(std::shared_ptr<EmbeddingQuery<data_type>> query) {
      std::scoped_lock<std::mutex> lock(query_map_mutex);
      if (query_map.count(query->get_query_id()) == 0){
	query_map[query->get_query_id()] = query;
      } else {
        throw std::runtime_error("query " + std::to_string(query->get_query_id()) +
                                 " already pushed to query_map for cluster " +
                                 std::to_string(parent->cluster_id));
      }
    }
  };

  /**
     batches stuff based on where the cluster is
  */
  class BatchingThread {
    GlobalSearchOCDPO<data_type> *parent;
    uint64_t thread_id;
    std::thread real_thread;
    
    template <typename K, typename V>
    bool is_empty(const std::unordered_map<K, std::unique_ptr<std::vector<V>>> &map) {
      bool empty = true;
      for (auto &item : map) {
        if (!item.second->empty()) {
          empty = false;
          break;
        }
      }
      return empty;
    }

    std::unordered_map<uint8_t,
                       std::unique_ptr<GlobalSearchMessageBatcher<data_type>>>
        cluster_messages;
    //key is client_id
    std::unordered_map<uint32_t,
                       std::unique_ptr<ANNSearchResultBatcher>>
        search_results;
    std::condition_variable_any messages_cv;
    std::mutex messages_mutex;

    bool running = false;

    // TODO: fix it to include search_results consideration.
    // the batching for sending results could probably be better. Should ask alicia 
    void main_loop(DefaultCascadeContextType *typed_ctxt) {
      std::unique_lock<std::mutex> lock(messages_mutex);
      std::unordered_map<uint8_t, std::chrono::steady_clock::time_point>
      wait_time;
      auto batch_time = std::chrono::microseconds(parent->batch_time_us);
      
      }

  public:
    BatchingThread(uint64_t thread_id, GlobalSearchOCDPO *parent) : thread_id(thread_id),parent(parent) {}

    void push_compute_query(uint32_t node_id, uint32_t query_id,
                            float min_distance, uint8_t cluster_receiver_id, uint32_t receiver_thread_id) {
      std::scoped_lock<std::mutex> lock(messages_mutex);
      compute_query_t query(node_id, query_id, min_distance, parent->cluster_id,
                            cluster_receiver_id, receiver_thread_id);

      if (cluster_messages.count(cluster_receiver_id) == 0) {
        cluster_messages[cluster_receiver_id] =
            std::make_unique<GlobalSearchMessageBatcher<data_type>>(
								    parent->dim);
      }
      cluster_messages[cluster_receiver_id]->push_compute_query(query);
    }

    void push_compute_res(compute_result_t res) {
      std::scoped_lock<std::mutex> lock(messages_mutex);      
      if (cluster_messages.count(res.cluster_receiver_id) == 0) {
        cluster_messages[res.cluster_receiver_id] =
            std::make_unique<GlobalSearchMessageBatcher<data_type>>(
								    parent->dim);
      }
      cluster_messages[res.cluster_receiver_id]->push_compute_res(res);
    }

    void push_ann_result(uint32_t query_id, uint32_t client_id, uint32_t K,
                         uint32_t L, std::shared_ptr<uint32_t[]> search_result,
                         uint8_t cluster_id) {
      std::scoped_lock<std::mutex> lock(messages_mutex);
      ann_search_result_t search_res = {.query_id = query_id,
                                        .client_id = client_id,
                                        .K = K,
                                        .L = L,
                                        .search_result = search_result,
                                        .cluster_id = cluster_id};
      if (search_results.count(client_id) == 0) {
        search_results[client_id] = std::make_unique<ANNSearchResultBatcher>(
									     parent->max_batch_size + 1);
      }
      search_results[client_id]->push(search_res);
    }

    void start(DefaultCascadeContextType *typed_ctxt) {
      running = true;
      this->real_thread =
        std::thread(&BatchingThread::main_loop, this, typed_ctxt);
      
    }
    void join() {
      if (real_thread.joinable()) {
	real_thread.join();
      }
    }
    void signal_stop() {
      std::scoped_lock<std::mutex> l(messages_mutex);
      running = false;
      messages_cv.notify_all();
    }
    
  };


  static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;


  bool initialized_index = false;
  std::unique_ptr<GlobalIndex> index;
  std::unique_ptr<BatchingThread> batch_thread;
  std::unique_ptr<DistanceComputeThread> distance_compute_thread;
  
  uint8_t cluster_id;

  uint32_t my_id;
  uint32_t dim;
  uint32_t num_search_threads;
  uint32_t next_search_thread;
  uint32_t num_compute_threads;
  uint32_t next_compute_thread;
  uint32_t min_batch_size;
  uint32_t max_batch_size;
  uint32_t batch_time_us;

  std::unique_ptr<diskann::Distance<data_type>> dist_fn;
  std::string cluster_data_prefix = UDL2_DATA_PREFIX;
  std::string cluster_search_prefix = UDL2_PATHNAME;

  std::string cluster_assignment_bin_file;
  std::vector<std::unique_ptr<SearchThread>> search_threads;


  void validate_search_query(
			     const std::shared_ptr<GreedySearchQuery<data_type>> &search_query) {
    if (search_query->get_dim() != this->dim) {
      throw std::runtime_error("Global UDL: dimension of query " +
                               std::to_string(search_query->get_query_id()) +
                               " different "
                               "from dimension specified in config");
    }

    if (search_query->get_cluster_id() != this->cluster_id) {
      throw std::runtime_error("Global UDL: cluster_id of query" +
                               std::to_string(search_query->get_query_id()) +
                               " different "
                               "from cluster id " +
                               std::to_string(this->cluster_id));
    }
  }

  void validate_emb_query(
			  const std::shared_ptr<EmbeddingQuery<data_type>> &emb_query) {
    if (emb_query->get_dim() != this->dim) {
      throw std::runtime_error("Global UDL: dimension of query " +
                               std::to_string(emb_query->get_query_id()) +
                               " different "
                               "from dimension specified in config");
    }
  }

  void validate_compute_query(const compute_query_t &compute_query) {
    if (compute_query.cluster_receiver_id != this->cluster_id) {
      throw std::runtime_error(
          "Global UDL: compute_query.cluster_receiver_id " +
          std::to_string(compute_query.cluster_receiver_id) +
          " different from this cluster id" + std::to_string(this->cluster_id));
    }
  }

  void validate_compute_result(const ComputeResult &compute_res) {
    if (compute_res.get_cluster_receiver_id() != this->cluster_id) {
      throw std::runtime_error(
          "Global UDL: " +
          std::to_string(compute_res.get_cluster_receiver_id()) +
          " different from this cluster id  " + std::to_string(cluster_id));
    }
  }
  
public:
  void ocdpo_handler(const node_id_t sender,
                     const std::string &object_pool_pathname,
                     const std::string &key_string,
                     const ObjectWithStringKey &object, const emit_func_t &emit,
                     DefaultCascadeContextType *typed_ctxt,
                     uint32_t worker_id) override {
    std::cout << "Global index called " << std::endl;
    if (!initialized_index) {
      cluster_id = get_cluster_id(key_string);
      cluster_data_prefix += "/cluster_" + std::to_string(cluster_id);
      cluster_search_prefix += "/cluster_" + std::to_string(cluster_id);
      dist_fn.reset(
          (diskann::Distance<data_type> *)
              diskann::get_distance_function<data_type>(diskann::Metric::L2));
      this->index =
        std::make_unique<GlobalIndex>(this, this->dim, this->cluster_id);
      initialized_index = true;
    }
    if (get_cluster_id(key_string) != cluster_id) {
      throw std::runtime_error(key_string + "doesn't belong to cluster " +
                               std::to_string(cluster_id));
    }

    GlobalSearchMessageBatchManager<data_type> manager(object.blob.bytes, object.blob.size, this->dim);
    for (const std::shared_ptr<GreedySearchQuery<data_type>> &search_query : manager.get_greedy_search_queries()) {
      validate_search_query(search_query);
      search_threads[next_search_thread]->push_search_query(search_query);
      next_search_thread = (next_search_thread + 1) % num_search_threads;
    }

    for (const std::shared_ptr<EmbeddingQuery<data_type>> &emb_query :
         manager.get_embedding_queries()) {
      validate_emb_query(emb_query);
      distance_compute_thread->push_embedding_query(emb_query);
    }

    for (const compute_query_t &query : manager.get_compute_queries()) {
      validate_compute_query(query);
      distance_compute_thread->push_compute_query(query);
    }

    for (const ComputeResult &result : manager.get_compute_results()) {
      validate_compute_result(result);
      search_threads[result.get_receiver_thread_id()]->push_compute_res(result);
    }
  }
  static void initialize() {
    if (!ocdpo_ptr) {
      ocdpo_ptr = std::make_shared<GlobalSearchOCDPO<data_type>>();
    }
    
  };
  static auto get() { return ocdpo_ptr; }
  void set_config(DefaultCascadeContextType *typed_ctxt,
                  const nlohmann::json &config) {
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    try {

      if (config.contains("cluster_assignment_bin_file")) {
        this->cluster_assignment_bin_file =
          config["cluster_assignment_bin_file"].get<std::string>();
      }
      if (config.contains("dim"))
        this->dim = config["dim"].get<uint32_t>();
      
      if (config.contains("num_search_threads")){
        this->num_search_threads = config["num_search_threads"].get<uint32_t>();
      }

      if (config.contains("min_batch_size")) {
        this->min_batch_size = config["min_batch_size"].get<uint32_t>();
      }

      if (config.contains("max_batch_size")) {
        this->max_batch_size = config["max_batch_size"].get<uint32_t>();
      }
      if (config.contains("batch_time_us")) {
        this->batch_time_us = config["batch_time_us"].get<uint32_t>();
      }
      
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert emb_dim or top_num_centroids from config" << std::endl;
        dbg_default_error("Failed to convert emb_dim or top_num_centroids from config, at centroids_search_udl.");
    }
    for (uint32_t thread_id = 0; thread_id < this->num_search_threads;
         thread_id++) {
      search_threads.emplace_back(new SearchThread(thread_id, this));
    }
    for (auto &search_thread : search_threads) {
      search_thread->start(typed_ctxt);
    }

    this->batch_thread = std::make_unique<BatchingThread>(this->my_id, this);
    this->batch_thread->start(typed_ctxt);

    this->distance_compute_thread = std::make_unique<DistanceComputeThread>(this->my_id, this);
    this->batch_thread->start(typed_ctxt);
    
  }

  void shutdown() {
    for (auto &search_thread : search_threads) {
      if (search_thread) {
        search_thread->signal_stop();
        search_thread->join();
      }
    }
    if (batch_thread) {
      batch_thread->signal_stop();
      batch_thread->join();
    }
    if (distance_compute_thread) {
      distance_compute_thread->signal_stop();
      distance_compute_thread->join();
    }
  }
  
};

template <typename data_type>
std::shared_ptr<OffCriticalDataPathObserver>
GlobalSearchOCDPO<data_type>::ocdpo_ptr;

  void initialize(ICascadeContext *ctxt);

  std::shared_ptr<OffCriticalDataPathObserver>
  get_observer(ICascadeContext *ctxt, const nlohmann::json &config);



} // namespace cascade
} // namespace derecho
