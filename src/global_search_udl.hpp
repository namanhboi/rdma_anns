#include <cascade/object.hpp>
#include <chrono>
#include <concepts>
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
    std::unique_ptr<const uint8_t, decltype(&std::free)> node_id_cluster_mapping;
    DefaultCascadeContextType *typed_ctxt;

    uint32_t dim;
    uint8_t cluster_id;
    GlobalSearchOCDPO* parent;
  public:
    GlobalIndex(GlobalSearchOCDPO *parent,
                DefaultCascadeContextType *typed_ctxt, int num_points,
                uint32_t dim, uint8_t cluster_id) {
      this->parent = parent;
      this->num_points = num_points;
      this->dim = dim;
      this->cluster_id = cluster_id;

      // retrieve data like mappings, etc...
      // TODO
      std::string mapping_key = parent->cluster_data_prefix + "_mapping";
      auto result = typed_ctxt->get_service_client_ref().get(mapping_key, true);
      auto &reply = result.get().begin()->second.get();
      Blob blob = std::move(const_cast<Blob &>(reply.blob));
      blob.memory_mode = object_memory_mode_t::EMPLACED; // memory now owned by us
      std::unique_ptr<const uint8_t, decltype(&std::free)> tmp(
							 (blob.bytes), std::free);

      node_id_cluster_mapping = std::move(tmp);
    }

    bool is_in_cluster(uint32_t node_id) {
      return node_id_cluster_mapping.get()[node_id] == this->cluster_id;
    }

    // could involve less copying of data, could just be a pointer to neighbor
    // address and then the number of neighbors
    std::vector<uint32_t> get_neighbors(uint32_t node_id) {
      if (!is_in_cluster(node_id))
	throw std::runtime_error(
				 "Node id " + std::to_string(node_id) + " not in cluster " +
				 std::to_string(this->cluster_id) + " " + __PRETTY_FUNCTION__);
    
      std::string nbr_key = parent->cluster_data_prefix + "_nbr_" + std::to_string(node_id);
      auto nbr = typed_ctxt->get_service_client_ref().get(nbr_key,
                                                          CURRENT_VERSION, true);
      auto &nbr_reply = nbr.get().begin()->second.get();
      Blob nbr_blob = std::move(const_cast<Blob &>(nbr_reply.blob));
      // nbr_blob.memory_mode =
	// object_memory_mode_t::EMPLACED; 
      uint32_t num_nbrs = nbr_blob.size / sizeof(uint32_t);
      const uint32_t *start_nbr =
	reinterpret_cast<const uint32_t *>(nbr_blob.bytes);
      std::vector<uint32_t> nbrs(start_nbr, start_nbr + num_nbrs);
      return nbrs;
    }

    // no need to do memcpy
    std::shared_ptr<data_type[]>
    retrieve_embeddings(const std::vector<uint32_t> &node_ids) {
      std::shared_ptr<data_type[]> emb_data = std::make_shared<data_type[]>(node_ids.size() * this->dim);
      for (size_t i = 0; i < node_ids.size(); i++) {
	uint32_t node_id = node_ids[i];
	if (!is_in_cluster(node_id))
          throw std::runtime_error(
				   "Node id " + std::to_string(node_id) + " not in cluster " +
				   std::to_string(this->cluster_id) + " " + __PRETTY_FUNCTION__ + " " + __FILE__ + " "  +__LINE__);
            
	std::string emb_key = parent->cluster_data_prefix + "_emb_" + std::to_string(node_id);
	auto emb = typed_ctxt->get_service_client_ref().get(emb_key,
                                                            CURRENT_VERSION, true);
	auto &emb_reply = emb.get().begin()->second.get();
	Blob emb_blob = std::move(const_cast<Blob &>(emb_reply.blob));
	// emb_blob.memory_mode =
          // object_memory_mode_t::EMPLACED; // this transfer ownership of memory to
	// this function?
    
	std::memcpy(emb_data.get() + i * this->dim, emb_blob.bytes, this->dim * sizeof(data_type));
      }
      return emb_data;
    }
  
      

    /**
       does greedy search on the global graph index.
       the server that is doing the search is called the primary partition while
       the servers doing distance compute tasks are called secondary partitions.

     If the current node in the candidate queue is stored in the local server's
     storage then explore their neighbors as usual. If the current node is not local
     then issue a distance computation request to the server with that node, which in
     turn will send back node_id and distance pairs to be added to the candidate
     queue.

For this, we need to know which nodes are local and which are not. Also there
should be a queue where the 2nd partitions can send back results which we will
check upon each new iteration of the search loop.
     */
    std::pair<uint32_t, uint32_t>
    search(std::shared_ptr<GreedySearchQuery<data_type>> query,
           std::shared_ptr<diskann::ConcurrentQueue<diskann::Neighbor>>
               compute_res_q,
           const uint32_t L, const uint32_t K, uint32_t *indices,
           float *distances) {

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
      auto compute_dists = [this, query](std::shared_ptr<data_type[]> embs,
					 std::vector<float> &dists_out,
					 size_t num_embs) {
	for (size_t i = 0; i < num_embs; i++) {
          dists_out.push_back(parent->dist_fn->compare(
						     query->get_embedding_ptr(), embs.get() + i * this->dim, this->dim));
	}
      };
      std::vector<uint32_t> init_node_ids(query->get_candidate_queue_ptr(),
                                          query->get_candidate_queue_ptr() +
                                          query->get_candidate_queue_size());
    
      std::shared_ptr<data_type[]> init_embs = retrieve_embeddings(init_node_ids);
      std::vector<float> init_distances;

      compute_dists(init_embs, init_distances, init_node_ids.size());
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
          diskann::Neighbor n = compute_res_q->pop();
          if (n.id != 0 && n.distance != -1) { // not popping an empty queue
            candidate_queue.insert(n);
          }
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
				   __FILE__ + " " + __LINE__);
	}

        std::vector<uint32_t> neighbors = get_neighbors(node_id);
	for (size_t i = 0; i < neighbors.size(); i++) {
          uint32_t nbr_node_id = neighbors[i];
	  uint8_t nbr_cluster_id = node_id_cluster_mapping.get()[nbr_node_id];
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
        
	std::shared_ptr<data_type[]> embs = retrieve_embeddings(primary_node_ids);
	size_t num_embs = primary_node_ids.size();
	primary_dist.reserve(num_embs);
        compute_dists(embs, primary_dist, num_embs);
        cmps += primary_dist.size();
        for (size_t m = 0; m < primary_node_ids.size(); m++) {
	  candidate_queue.insert(diskann::Neighbor(primary_node_ids[m], primary_dist[m]));
        }
        float min_distance = candidate_queue[candidate_queue.size() - 1].distance;

        for (auto &[cluster_id, node_ids] : second_partition_nodes) {
          for (uint32_t node_id : node_ids) {
            parent->batch_thread->push_compute_query(
						     node_id, query->get_query_id(), min_distance, cluster_id);
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
    std::unordered_map<uint32_t, std::shared_ptr<diskann::ConcurrentQueue<compute_result_t>>> compute_res_queues;
    std::mutex compute_res_queues_mtx;

    void main_loop() {
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
	compute_result_t empty_res;
        compute_res_queues[query_id] =
          std::make_shared<diskann::ConcurrentQueue<compute_result_t>>(empty_res);
        compute_res_lock.unlock();
        lock.unlock();
	std::shared_ptr<uint32_t[]> result(new uint32_t[parent->K]);
        parent->index->search(query, compute_res_queues[query_id], parent->L,
                              parent->K, result.get(), nullptr);
        parent->batch_thread->push_ann_result(
            query->get_query_id(), query->get_client_node_id(), parent->K,
					      parent->L, std::move(result), parent->cluster_id);
        
      }


    }
    
  public:
    /**
       push the compute query result
       // TODO this should be fixed to cuse ComputeResult
    */
    void push_compute_res(compute_result_t result) {
      if (result.cluster_receiver_id != parent->cluster_id) {
        throw std::invalid_argument(
            "result computation between query id " +
            std::to_string(result.query_id) + " and node id " +
            std::to_string(result.node.id) +
            " has mismatch cluster id (this cluster vs intended receiver):" +
            std::to_string(parent->cluster_id) + " " +
            std::to_string(result.cluster_receiver_id));
      }
      uint32_t query_id = result.query_id;
      this->compute_res_queues[query_id]->push(result);
      // this is fine since concurrent queue has internal locks
    }
    void push_search_query(std::shared_ptr<GreedySearchQuery<data_type>> search_query) {
      std::scoped_lock<std::mutex> lock(query_queue_mutex);
      query_queue.emplace(std::move(search_query));
      query_queue_cv.notify_all();
    }

    void start() {}
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
        std::shared_ptr<const uint32_t> nbr_ptr(
						reinterpret_cast<const uint32_t *>(nbr_blob.bytes), std::free);
        
	uint32_t num_nbrs = *nbr_ptr.get();
        

        if (distance >= query.min_distance) {
          compute_result_t res = {
            .cluster_sender_id = parent->cluster_id,
            .cluster_receiver_id = query.cluster_sender_id,
            .node = node,
            .query_id = query.query_id,
            .num_neighbors = num_nbrs,
            .nbr_ptr = std::move(nbr_ptr)
          };
          parent->batch_thread->push_compute_res(res);
        }
      }
    }
  public:
    void push_compute_query(compute_query_t &compute_query) {
      std::scoped_lock<std::mutex> lock(compute_queue_mutex);
      compute_queue.emplace(compute_query);
      compute_queue_cv.notify_all();
    }

    void push_embedding_query(std::shared_ptr<EmbeddingQuery<data_type>> query) {
      std::scoped_lock<std::mutex> lock(query_map_mutex);
      if (query_map.count(query->get_query_id()) == 0){
	query_map[query->get_query_id()] = query;
      }
    }

  };


  /**
     batches stuff based on where the cluster is
  */
  class BatchingThread {
    GlobalSearchOCDPO<data_type> *parent;

    // key is cluster_id
    template <typename K, typename V>
    bool is_empty(
		      const std::unordered_map<K, std::unique_ptr<std::vector<V>>> &map) {
      bool empty = true;
      for (auto &item : map) {
        if (!item.second->empty()) {
          empty = false;
          break;
        }
      }
      return empty;
    }

    std::unordered_map<
        uint8_t, std::unique_ptr<std::vector<global_search_message<data_type>>>>
        cluster_messages;



    //key is client_id
    std::unordered_map<uint32_t,
                       std::unique_ptr<std::vector<ann_search_result_t>>>
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

      while (running) {
        lock.lock();
        if (is_empty(cluster_messages) && is_empty(search_results)) {
          messages_cv.wait_for(lock, batch_time);
        }
        if (!running)
          break;
        
	// BELOW HERE I NEED TO TAKE A LOOK AT, NEED FIXING

	//key is cluster_id
        std::unordered_map<
            uint8_t,
            std::unique_ptr<std::vector<global_search_message<data_type>>>>
        to_send;

        auto now = std::chrono::steady_clock::now();
        
	for (auto &[cluster_id, messages_ptr] : cluster_messages) {
          if (wait_time.count(cluster_id) == 0) {
	    wait_time[cluster_id] = now;
          }
          if (messages_ptr->size() >= parent->min_batch_size ||
              (now - wait_time[cluster_id] >= batch_time)) {
            to_send[cluster_id] = std::move(cluster_messages[cluster_id]);
            cluster_messages[cluster_id] = std::make_unique<
							    std::vector<global_search_message<data_type>>>();
            cluster_messages[cluster_id].reserve(parent->max_batch_size);
          }
        }
        lock.unlock();
//TODO
        
	// //key is client node id
        // std::unordered_map<
            // uint32_t,
            // std::unique_ptr<std::vector<global_search_message<data_type>>>>
        // results_by_client_id;


        // for (auto &search_res_msg : *to_send[search_result_bucket_id]) {
          // if (search_res_msg.msg_type != SEARCH_RES) {
            // throw std::runtime_error("expected search result message in " +
                                     // std::to_string(search_result_bucket_id));
          // }
          // uint32_t client_id = search_res_msg.search_res.client_id;
          // if (results_by_client_id.count(client_id) == 0) {
            // results_by_client_id[client_id] = std::make_unique<
							       // std::vector<global_search_message<data_type>>>();
            // results_by_client_id[client_id]->reserve(parent->max_batch_size);
          // }
          // results_by_client_id[client_id]->emplace_back(
							// std::move(search_res_msg));
        // }

        // done with all data prep, now time to send batches of messages. Need
        // to finish serializaiton part first.
        



        
      }

    }
    
  public:
    void push_message(uint8_t cluster_receiver_id,
                      global_search_message<data_type> &msg) {
      std::scoped_lock<std::mutex> lock(messages_mutex);
      if (cluster_messages.count(cluster_receiver_id) == 0) {
        cluster_messages[cluster_receiver_id] =
          std::make_unique<std::vector<global_search_message<data_type>>>();
        cluster_messages[cluster_receiver_id]->reserve(parent->max_batch_size);
      }
      cluster_messages[cluster_receiver_id]->push_back(msg);
    }
    void push_compute_query(uint32_t node_id, uint32_t query_id,
                            float min_distance, uint8_t cluster_receiver_id) {
      global_search_message<data_type> msg = {
          .msg_type = COMPUTE_QUERY,
          .compute_query = {.node_id = node_id,
                            .query_id = query_id,
                            .min_distance = min_distance,
                            .cluster_sender_id = parent->cluster_id,
                            .cluster_receiver_id = cluster_receiver_id}};
      push_message(cluster_receiver_id, msg);
    }
    void push_compute_res(compute_result_t res) {
      global_search_message<data_type> msg = {.msg_type = COMPUTE_RES,
                                              .msg = res};
      push_message(res.cluster_receiver_id, msg);
    }

    void push_ann_result(uint32_t query_id, uint32_t client_id, uint32_t K,
                         uint32_t L, std::shared_ptr<uint32_t[]> search_result,
                         uint8_t cluster_id) {
      ann_search_result_t search_res = {.query_id = query_id,
                                        .client_id = client_id,
                                        .K = K,
                                        .L = L,
                                        .search_result =
                                            std::move(search_result),
                                        .cluster_id = cluster_id};
      std::scoped_lock<std::mutex> lock(messages_mutex);
      search_results[client_id]->emplace_back(std::move(search_res));
    }      
  };


  static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;


  bool initialized_index = false;
  std::unique_ptr<GlobalIndex> index;
  std::unique_ptr<BatchingThread> batch_thread;
  uint8_t cluster_id;

  uint32_t num_local_points;
  uint32_t my_id;
  uint32_t dim;
  uint32_t num_search_threads;
  uint32_t min_batch_size;
  uint32_t max_batch_size;
  uint32_t batch_time_us;
  uint32_t K;
  uint32_t L;

  std::string cluster_data_prefix;
  std::string cluster_search_prefix;
  std::unique_ptr<diskann::Distance<data_type>> dist_fn;

  
  
  

public:
  void ocdpo_handler(const node_id_t sender,
                     const std::string &object_pool_pathname,
                     const std::string &key_string,
                     const ObjectWithStringKey &object, const emit_func_t &emit,
                     DefaultCascadeContextType *typed_ctxt,
                     uint32_t worker_id) override {
    // todo how to 
    std::cout << "Global index called " << std::endl;
    if (!initialized_index) {
      cluster_id = get_cluster_id(key_string);
      // cluster_prefix = GLOBAL_SEARCH_SEARCH_PREFIX "/cluster
      cluster_data_prefix = GLOBAL_SEARCH_DATA_PREFIX "/cluster_" + std::to_string(cluster_id);
      cluster_search_prefix = GLOBAL_SEARCH_SEARCH_PREFIX "/cluster_" + std::to_string(cluster_id);
      dist_fn.reset(
          (diskann::Distance<data_type> *)
              diskann::get_distance_function<data_type>(diskann::Metric::L2));
      this->index = std::make_unique<GlobalIndex>(
          this, typed_ctxt, this->num_local_points, this->dim,
						  this->cluster_id);
      initialized_index = true;
    }
  }
  static void initialize() {};
  static auto get() { return ocdpo_ptr; }
  void set_config(DefaultCascadeContextType *typed_ctxt,
                  const nlohmann::json &config) {
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    try{
      if (config.contains("num_local_points"))
        this->dim = config["num_local_points"].get<uint32_t>();
      
      if (config.contains("dim"))
        this->dim = config["dim"].get<uint32_t>();
      
      if (config.contains("num_search_threads"))
        this->num_search_threads = config["num_search_threads"].get<uint32_t>();

      if (config.contains("min_batch_size"))
        this->min_batch_size = config["min_batch_size"].get<uint32_t>();

      if (config.contains("max_batch_size"))
        this->max_batch_size = config["max_batch_size"].get<uint32_t>();

      if (config.contains("batch_time_us"))
        this->batch_time_us = config["batch_time_us"].get<uint32_t>();
      
      if (config.contains("batch_time_us"))
        this->batch_time_us = config["batch_time_us"].get<uint32_t>();
      
      if (config.contains("batch_time_us"))
        this->batch_time_us = config["batch_time_us"].get<uint32_t>();
      
      if (config.contains("K"))
        this->K = config["K"].get<uint32_t>();
      
      if (config.contains("L"))
        this->L = config["L"].get<uint32_t>();
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert emb_dim or top_num_centroids from config" << std::endl;
        dbg_default_error("Failed to convert emb_dim or top_num_centroids from config, at centroids_search_udl.");
    }    
  }
  void shutdown() {}
};

template <typename data_type>
std::shared_ptr<OffCriticalDataPathObserver>
GlobalSearchOCDPO<data_type>::ocdpo_ptr;

  void initialize(ICascadeContext *ctxt);

  std::shared_ptr<OffCriticalDataPathObserver>
  get_observer(ICascadeContext *ctxt, const nlohmann::json &config);



} // namespace cascade
} // namespace derecho
