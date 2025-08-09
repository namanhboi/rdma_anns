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
#include <filesystem>


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
    GlobalIndex(GlobalSearchOCDPO *parent, uint32_t dim, uint8_t cluster_id) {
      this->parent = parent;
      this->cluster_id = cluster_id;
      // std::cout << "this cluster is " << cluster_id <<std::endl;
      std::string cluster_assignment_file =
        parent->cluster_assignment_bin_file;

      if (!std::filesystem::exists(cluster_assignment_file)) {
        throw std::runtime_error("cluster assignment file :" +
                                 cluster_assignment_file + " doesn't exist");
      }
      this->dim = dim;
      // std::cout << "global index dim " << this->dim <<std::endl;
      std::ifstream in(cluster_assignment_file, std::ios::binary);
      in.read((char *)&num_points, sizeof(num_points));
      // std::cout << "number of points is" << num_points << std::endl;
      cluster_assignment.resize(num_points);
      in.read((char *)&num_clusters, sizeof(num_clusters));
      in.read((char *)cluster_assignment.data(), sizeof(uint8_t) * num_points);
      // std::cout << "Done initializing global index" << std::endl;
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
      // std::cout << node_ids << std::endl;
      for (size_t i = 0; i < node_ids.size(); i++) {
	uint32_t node_id = node_ids[i];
        // std::cout << " starting retrieve process " << std::endl;
        if (!is_in_cluster(node_id)) {
          // std::cout << " not in cluster " << std::endl;
          throw std::runtime_error("Node id " + std::to_string(node_id) +
                                   " not in cluster " +
                                   std::to_string(this->cluster_id) + " " +
                                   __PRETTY_FUNCTION__ + " " + __FILE__);
        }

        std::string emb_key =
          parent->cluster_data_prefix + "_emb_" + std::to_string(node_id);
	// std::cout << "embedding key is " << emb_key << std::endl;
	auto emb = typed_ctxt->get_service_client_ref().get(emb_key,
                                                            CURRENT_VERSION, true);
	auto &emb_reply = emb.get().begin()->second.get();
	Blob emb_blob = std::move(const_cast<Blob &>(emb_reply.blob));
        emb_blob.memory_mode = object_memory_mode_t::EMPLACED;
	// std::cout  << "size of emb blob " << emb_blob.size << std::endl;
	// for (uint32_t j = 0; j < dim; j++) {
	  // std::cout << (reinterpret_cast<const data_type*>(emb_blob.bytes))[j] <<  std::endl;
	// }
	// std::cout << std::endl;

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

      // for (uint32_t i = 0; i < query->get_dim(); i++) {
	  // std::cout << query->get_embedding_ptr()[i] << " " ;
        // }
        // std::cout << std::endl;      
      // Lambda to batch compute query<-> node distnaces
      auto compute_dists =
          [this,
           query](const std::vector<std::shared_ptr<const data_type>> &embs,
                  std::vector<float> &dists_out) {
            for (size_t i = 0; i < embs.size(); i++) {
              // std::cout << "dim is " << this->dim << std::endl;
              // std::cout << "query emb ptr" << query->get_embedding_ptr() << std::endl;
              // std::cout << "embedding ptr" << embs[i].get() << " "
              // << *embs[i].get() << std::endl;
              float distance = (parent->dist_fn->compare(
							 query->get_embedding_ptr(), embs[i].get(), this->dim));
              // std::cout << "distance is " << distance << std::endl;
              dists_out.push_back(distance);
            }
          };

      std::vector<uint32_t> init_node_ids(query->get_candidate_queue_ptr(),
                                          query->get_candidate_queue_ptr() +
                                          query->get_candidate_queue_size());
      // std::cout << "starting to retrieve local embs" << std::endl;
      std::vector<std::shared_ptr<const data_type>> init_embs =
        retrieve_local_embeddings(init_node_ids, typed_ctxt);
      // std::cout << "done retrievnig local embs " << init_embs.size() << std::endl;
      std::vector<float> init_distances;

      compute_dists(init_embs, init_distances);
      // std::cout << "done calculating distance for init" << std::endl;
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

      GetRequestManager<data_type, derecho::cascade::ObjectWithStringKey>
      emb_get_request_manager;


      std::unordered_map<uint32_t,
                         std::shared_ptr<derecho::rpc::QueryResults<const ObjectWithStringKey>>>
      nbr_get_requests;
      // std::cout << "starting search" << std::endl;
      while (candidate_queue.has_unexpanded_node()) {
	diskann::Neighbor node =  candidate_queue.closest_unexpanded();
	uint32_t node_id = node.id;
	hops++;
	primary_node_ids.clear();
        primary_dist.clear();

        secondary_dist.clear();

        std::shared_ptr<const uint32_t> neighbors;
        if (!is_in_cluster(node_id)) {
          // if node_id not in cluster then the get request for its neighbors
          // has already been sent and the future obj resides in
          // nbr_get_requests
          // std::cout << "node id " << node_id << "not in current cluster " << parent->cluster_id << std::endl;
          if (nbr_get_requests[node_id] == nullptr) {
            throw std::runtime_error("request for neighbor not issued even "
                                     "though this is the candidate node");
          }
          auto &reply = nbr_get_requests[node_id]->get().begin()->second.get();
          Blob nbr_blob = std::move(const_cast<Blob &>(reply.blob));
          nbr_blob.memory_mode =
            derecho::cascade::object_memory_mode_t::EMPLACED;
          std::shared_ptr<const uint32_t> tmp(reinterpret_cast<const uint32_t *>(nbr_blob.bytes), free_const);
          neighbors = std::move(tmp);
        } else {
          // std::cout << "start getting neighbors " << std::endl;
          neighbors = get_neighbors(node_id, typed_ctxt);
	  // std::cout << "done getting neighbors" << std::endl;          
        }

        uint32_t num_nbrs = neighbors.get()[0];
	// std::cout << "number of neighbors " << std::endl;
        const uint32_t *neighbors_ptr = neighbors.get() + 1;
	for (size_t i = 0; i < num_nbrs; i++) {
          uint32_t nbr_node_id = neighbors_ptr[i];
	  uint8_t nbr_cluster_id = cluster_assignment[nbr_node_id];
          if (nbr_cluster_id == this->cluster_id) {
            if (is_not_visited(nbr_node_id)) {
              primary_node_ids.push_back(nbr_node_id);
              mark_visited(nbr_node_id);
            }
          } else {
            // std::cout << "node not in cluster " << nbr_node_id << std::endl;
            if (is_not_visited(nbr_node_id)) {
              const std::string &emb_key = parent->cluster_data_prefix +
                                           "_emb_" +
                                           std::to_string(nbr_node_id);
              emb_get_request_manager.submit_request(
                  nbr_node_id, typed_ctxt->get_service_client_ref().get(
									emb_key, CURRENT_VERSION, true));
              // there is probably a better way to do this without sending a
              // request for the neighbors of a a neighboring node
              const std::string &nbr_key = parent->cluster_data_prefix +
                                           "_nbr_" +
                                           std::to_string(nbr_node_id);
              nbr_get_requests[nbr_node_id] = std::make_shared<
                  derecho::rpc::QueryResults<const ObjectWithStringKey>>(
                  std::move(typed_ctxt->get_service_client_ref().get(
								     nbr_key, CURRENT_VERSION, true)));
              mark_visited(nbr_node_id);
            }
          }
        }
	// std::cout << "retrieveing embeddings for nodes in this cluster " << std::endl;
        std::vector<std::shared_ptr<const data_type>> embs =
          retrieve_local_embeddings(primary_node_ids, typed_ctxt);

        // std::cout << primary_node_ids.size() << std::endl;
	// std::cout << "done retrieveing embeddings for nodes in this cluster " << std::endl;                
	primary_dist.reserve(embs.size());
	// std::cout << "start distance calc for nodes in this cluster " << std::endl;        
        compute_dists(embs, primary_dist);
        // std::cout << "done distance calc for nodes in this cluster " <<
        // std::endl;
        // for (float x : primary_dist) {
	  // std::cout << x << " " ;
        // }
	// std::cout << std::endl;
        cmps += primary_dist.size();
        for (size_t m = 0; m < primary_node_ids.size(); m++) {
	  candidate_queue.insert(diskann::Neighbor(primary_node_ids[m], primary_dist[m]));
        }

	// std::cout << "done with nodes in this cluster " << std::endl;

        // get embedding data of whichever requests has arrived.
        // const std::pair<std::vector<uint64_t>,
        //                 std::vector<std::shared_ptr<const data_type>>>
        //     &data_secondary_partition =
        //       emb_get_request_manager.get_available_requests();
        // const std::vector<uint64_t> &id = data_secondary_partition.first;
        // const std::vector<std::shared_ptr<const data_type>> &embeddings = data_secondary_partition.second;
	
        // secondary_dist.reserve(embeddings.size());
        // compute_dists(embeddings, secondary_dist);
        // cmps += secondary_dist.size();
        // for (size_t m = 0; m < embeddings.size(); m++) {
	//   candidate_queue.insert(diskann::Neighbor(id[m], secondary_dist[m]));
        // }
      }
      for (size_t i = 0; i < std::min(K, (uint32_t)candidate_queue.size()); i++) {
        indices[i] = candidate_queue[i].id;
        if (distances != nullptr)
          distances[i] = candidate_queue[i].distance;
      }
      // std::cout << "number of hops " << hops << std::endl;
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
      // std::cout << "number of hops " << hops << std::endl;
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
        // std::unique_lock<std::mutex> compute_res_lock(compute_res_queues_mtx);
	// ComputeResult empty_res;
        // compute_res_queues[query_id] =
        //   std::make_shared<diskann::ConcurrentQueue<ComputeResult>>(empty_res);
        // compute_res_lock.unlock();
        lock.unlock();
        // std::cout << query_id << " "  << query->get_dim() <<std::endl;
        // for (uint32_t i = 0; i < query->get_dim(); i++) {
	  // std::cout << query->get_embedding_ptr()[i] << " " ;
        // }
        // std::cout << std::endl;
        
        const uint32_t &K = query->get_K();
        const uint32_t &L = query->get_L();
        std::shared_ptr<uint32_t[]> result(new uint32_t[query->get_K()]);
#ifdef TASK_PUSH
        parent->index->search_task_push(
            typed_ctxt, query, compute_res_queues[query_id], this->my_thread_id,
					      K, L, result.get(), nullptr);
#elif GLOBAL_BASELINE
	// std::cout << "hello" << std::endl;
	parent->index->search_global_baseline(typed_ctxt, query, K, L, result.get(), nullptr);
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
      std::unique_lock<std::mutex> map_lock(query_map_mutex, std::defer_lock);
      
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
        std::thread(&GlobalSearchOCDPO::DistanceComputeThread::main_loop, this, typed_ctxt);
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

    enum class message_type : uint8_t { COMPUTE_RESULT, COMPUTE_QUERY };

    /**
       this struct is for inter-node communication and to simplify the process
       of batching and serializing these inter-node messages. I added this to
       make doing max_batch_size easier.
       In the future, it will also support candidate queue for cotra search modex.
    */
    struct message {
      message_type msg_type;
      std::variant<compute_result_t, compute_query_t> msg;
    };
    
    GlobalSearchOCDPO<data_type> *parent;
    uint64_t thread_id;
    std::thread real_thread;

    template <typename K, typename V>
    bool
    is_empty(const std::unordered_map<K, std::unique_ptr<std::vector<V>>> &map) {
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
                       std::unique_ptr<std::vector<message>>>
        cluster_messages;
    //key is client_id
    std::unordered_map<uint32_t,
                       std::unique_ptr<std::vector<ann_search_result_t>>>
        search_results;
    std::condition_variable_any messages_cv;
    std::mutex messages_mutex;

    bool running = false;

    void main_loop(DefaultCascadeContextType *typed_ctxt) {
      std::unique_lock<std::mutex> lock(messages_mutex, std::defer_lock);
      std::unordered_map<uint8_t, std::chrono::steady_clock::time_point>
      wait_time_messages;
      std::unordered_map<uint32_t, std::chrono::steady_clock::time_point>
      wait_time_results;
      auto batch_time = std::chrono::microseconds(parent->batch_time_us);
      GlobalSearchMessageBatcher<data_type> message_batcher(parent->dim);
      ANNSearchResultBatcher result_batcher;
      
      while (running) {
        lock.lock();
        while (is_empty(cluster_messages) && is_empty(search_results)) {
          messages_cv.wait_for(lock, batch_time);
        }
        if (!running) {
	  break;
        }

        std::unordered_map<uint8_t, std::unique_ptr<std::vector<message>>>
        messages_to_send;

        std::unordered_map<uint32_t,
                           std::unique_ptr<std::vector<ann_search_result_t>>>
        results_to_send;

	auto now = std::chrono::steady_clock::now();
        for (auto &[cluster_id, messages] : cluster_messages) {
          if (wait_time_messages.count(cluster_id) == 0) {
            wait_time_messages[cluster_id] = now;
          }
          if (messages->size() >= parent->min_batch_size ||
              ((now - wait_time_messages[cluster_id]) >= batch_time)) {
            messages_to_send[cluster_id] = std::move(messages);
            cluster_messages[cluster_id] =
              std::make_unique<std::vector<message>>();
            cluster_messages[cluster_id]->reserve(parent->max_batch_size);
          }
        }

        for (auto &[client_id, results] : search_results) {
          if (wait_time_results.count(client_id) == 0) {
            wait_time_results[client_id] = now;
          }
          if (results->size() >= parent->min_batch_size ||
              ((now - wait_time_results[client_id]) >= batch_time)) {
            results_to_send[client_id] = std::move(results);
            search_results[client_id] =
              std::make_unique<std::vector<ann_search_result_t>>();
          }
        }
        lock.unlock();

        for (auto &[cluster_id, messages] : messages_to_send) {
          uint64_t num_sent = 0;
          uint64_t total = messages->size();

          while (num_sent < total) {
            uint32_t left = total - num_sent;
            uint32_t batch_size = std::min(parent->max_batch_size, left);
            for (uint32_t i = num_sent; i < num_sent + batch_size; i++) {
              if (messages->at(i).msg_type == message_type::COMPUTE_RESULT) {

                message_batcher.push_compute_result(
						    std::move(std::get<compute_result_t>(messages->at(i).msg)));
              } else if (messages->at(i).msg_type ==
                         message_type::COMPUTE_QUERY) {

                message_batcher.push_compute_query(
						   std::move(std::get<compute_query_t>(messages->at(i).msg)));
              } 
            }
            message_batcher.serialize();
            ObjectWithStringKey obj;
            obj.blob = std::move(*message_batcher.get_blob());
            obj.previous_version = INVALID_VERSION;
            obj.previous_version_by_key = INVALID_VERSION;

            // send data to the correct cluster, send using pathname because
            // that is how you trigger the udl handler
            obj.key = UDL2_PATHNAME "/cluster_" + std::to_string(cluster_id);
            typed_ctxt->get_service_client_ref().trigger_put(obj);
            num_sent += batch_size;
            message_batcher.reset();
          }
        }
        for (auto &[client_id, results] : results_to_send) {
          uint64_t num_sent = 0;
          uint64_t total = results->size();

          while (num_sent < total) {
            uint32_t left = total - num_sent;
            uint32_t batch_size = std::min(parent->max_batch_size, left);
            for (uint32_t i = num_sent; i < num_sent + batch_size; i++) {
              result_batcher.push(std::move(results->at(i)));
            }
            
            result_batcher.serialize();
            // need to notify the client.

            std::string client_id_pool_path =
              RESULTS_OBJ_POOL_PREFIX "/" + std::to_string(client_id);
            typed_ctxt->get_service_client_ref().notify(
							*(result_batcher.get_blob()), client_id_pool_path, client_id);

            num_sent += batch_size;
            result_batcher.reset();
          }          
	}
      }
      
    }
    

  public:
    BatchingThread(uint64_t thread_id, GlobalSearchOCDPO *parent) : thread_id(thread_id),parent(parent) {}

    void push_compute_query(uint32_t node_id, uint32_t query_id,
                            float min_distance, uint8_t cluster_receiver_id, uint32_t receiver_thread_id) {
      std::scoped_lock<std::mutex> lock(messages_mutex);
      compute_query_t query(node_id, query_id, min_distance, parent->cluster_id,
                            cluster_receiver_id, receiver_thread_id);
      // message mes(std::move(query));
      message msg = {message_type::COMPUTE_QUERY, std::move(query)};

      if (cluster_messages.count(cluster_receiver_id) == 0) {
        cluster_messages[cluster_receiver_id] =
          std::make_unique<std::vector<message>>();
        cluster_messages[cluster_receiver_id]->reserve(parent->max_batch_size);
      }
      cluster_messages[cluster_receiver_id]->emplace_back(std::move(msg));
    }

    void push_compute_res(compute_result_t res) {
      std::scoped_lock<std::mutex> lock(messages_mutex);      
      if (cluster_messages.count(res.cluster_receiver_id) == 0) {
        cluster_messages[res.cluster_receiver_id] =
          std::make_unique<std::vector<message>>();
        cluster_messages[res.cluster_receiver_id]->reserve(
							   parent->max_batch_size);
      }
      message mes = {message_type::COMPUTE_RESULT, std::move(res)};
      cluster_messages[res.cluster_receiver_id]->emplace_back(std::move(mes));
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
        search_results[client_id] =
          std::make_unique<std::vector<ann_search_result_t>>();
        search_results[client_id]->reserve(parent->max_batch_size);
      }
      search_results[client_id]->emplace_back(std::move(search_res));
    }

    void start(DefaultCascadeContextType *typed_ctxt) {
      running = true;
      this->real_thread =
        std::thread(&GlobalSearchOCDPO::BatchingThread::main_loop, this, typed_ctxt);
    }
    void join() {
      if (real_thread.joinable()) {
	real_thread.join();
      }
    }
    void signal_stop() {
      std::scoped_lock<std::mutex> l(messages_mutex);
      running = false;
      search_results[0]->push_back({});
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
                               "(" + std::to_string(search_query->get_dim()) +
                               ")" +
                               " different "
                               "from dimension specified in config " +
                               std::to_string(this->dim));
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
      throw std::runtime_error(
          "Global UDL: dimension of query " +
          std::to_string(emb_query->get_query_id()) + "(" +
          std::to_string(emb_query->get_dim()) + ")" + " different " +
          "from dimension specified in config " + std::to_string(this->dim));
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
    // std::cout << "Global index called " <<  key_string<< std::endl;
    // can probably optimize this part by doing shared ptr for compute query and
    // compute result
    if (!initialized_index) {
      cluster_id = get_cluster_id(key_string);
      cluster_data_prefix += "/cluster_" + std::to_string(cluster_id);
      cluster_search_prefix += "/cluster_" + std::to_string(cluster_id);
      // std::cout << cluster_data_prefix << std::endl;
      // std::cout << cluster_search_prefix << std::endl;
      dist_fn.reset(
          (diskann::Distance<data_type> *)
              diskann::get_distance_function<data_type>(diskann::Metric::L2));
      // std::cout << "started making global idnex" << std::endl;
      this->index =
        std::make_unique<GlobalIndex>(this, this->dim, this->cluster_id);
      initialized_index = true;
    }
    if (get_cluster_id(key_string) != cluster_id) {
      throw std::runtime_error(key_string + "doesn't belong to cluster " +
                               std::to_string(cluster_id));
    }

    GlobalSearchMessageBatchManager<data_type> manager(
						       object.blob.bytes, object.blob.size, this->dim);
    // std::cout << "num search threads started " << search_threads.size() << std::endl;
    for (std::shared_ptr<GreedySearchQuery<data_type>> &search_query :
         manager.get_greedy_search_queries()) {
      validate_search_query(search_query);
      // std::cout << "query ok" << std::endl;
      search_threads[next_search_thread]->push_search_query(std::move(search_query));
      next_search_thread = (next_search_thread + 1) % num_search_threads;
      // std::cout << next_search_thread << std::endl;
    }

    // for (std::shared_ptr<EmbeddingQuery<data_type>> &emb_query :
         // manager.get_embedding_queries()) {
      // validate_emb_query(emb_query);
      // distance_compute_thread->push_embedding_query(std::move(emb_query));
    // }

    // for (compute_query_t &query : manager.get_compute_queries()) {
      // validate_compute_query(query);
      // distance_compute_thread->push_compute_query(std::move(query));
    // }

    for (ComputeResult &result : manager.get_compute_results()) {
      validate_compute_result(result);
      search_threads[result.get_receiver_thread_id()]->push_compute_res(
									std::move(result));
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
      std::cout << "dimension is " << dim << std::endl;
      
      if (config.contains("num_search_threads")){
        this->num_search_threads = config["num_search_threads"].get<uint32_t>();
      }
      std::cout << "num_search_threads " <<  num_search_threads << std::endl;

      if (config.contains("min_batch_size")) {
        this->min_batch_size = config["min_batch_size"].get<uint32_t>();
      }

      if (config.contains("max_batch_size")) {
        this->max_batch_size = config["max_batch_size"].get<uint32_t>();
      }
      
      if (config.contains("batch_time_us")) {
        this->batch_time_us = config["batch_time_us"].get<uint32_t>();
      }

    } catch (const std::exception &e) {
      std::cout << "error while parsing config" << std::endl;
    }
    std::cout << "hello " << std::endl;
    
    for (uint32_t thread_id = 0; thread_id < this->num_search_threads;
         thread_id++) {
      search_threads.emplace_back(new SearchThread(thread_id, this));
    }
    std::cout << "hello 1" << std::endl;    
    for (auto &search_thread : search_threads) {
      search_thread->start(typed_ctxt);
      // std::cout << "started thread "  << search_thread.
    }

    std::cout << "hello 2" << std::endl;

    this->batch_thread = std::make_unique<BatchingThread>(this->my_id, this);
    this->batch_thread->start(typed_ctxt);
    std::cout << "hello 3" << std::endl;

    // this->distance_compute_thread = std::make_unique<DistanceComputeThread>(this->my_id, this);
    // this->distance_compute_thread->start(typed_ctxt);
    // std::cout << "hello 4" << std::endl;
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
    // if (distance_compute_thread) {
    //   distance_compute_thread->signal_stop();
    //   distance_compute_thread->join();
    // }
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
