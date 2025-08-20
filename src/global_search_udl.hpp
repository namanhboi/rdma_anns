#include <cascade/object.hpp>
#include <chrono>
#include <derecho/core/detail/p2p_connection.hpp>
#include <immintrin.h> // needed to include this to make sure that the code compiles since in DiskANN/include/utils.h it uses this library.
#include "serialize_utils.hpp"
#include <cascade/cascade_interface.hpp>
#include <cascade/service_types.hpp>
#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include "concurrent_queue.h"
#include "neighbor.h"
#include <stdexcept>
#include <boost/dynamic_bitset.hpp>
#include "udl_path_and_index.hpp"
#include "in_mem_search_index.hpp"
#include "ssd_search_index.hpp"

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
  class SearchThread {
    uint64_t my_thread_id;
    GlobalSearchOCDPO *parent;
    std::thread real_thread;
    bool running = false;

    std::condition_variable_any query_queue_cv;
    std::mutex query_queue_mutex;
    std::queue<std::shared_ptr<GreedySearchQuery<data_type>>> query_queue;


    std::shared_ptr<diskann::ConcurrentNeighborPriorityQueue> retset =
        std::make_shared<diskann::ConcurrentNeighborPriorityQueue>();

    

    bool initialized_thread_data = false;

    void main_loop(DefaultCascadeContextType *typed_ctxt) {
      // std::cout << "Search thread id is " << std::this_thread::get_id() << std::endl;
      std::unique_lock<std::mutex> query_queue_lock(query_queue_mutex, std::defer_lock);
      std::shared_lock<std::shared_mutex> index_lock(parent->index_mutex,
                                                     std::defer_lock);

      while (running) {
        query_queue_lock.lock();

        while (query_queue.empty()) {
	  query_queue_cv.wait(query_queue_lock);
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

        query_queue_lock.unlock();
        
        
        // std::cout << query_id << " "  << query->get_dim() <<std::endl;
        // for (uint32_t i = 0; i < query->get_dim(); i++) {
	  // std::cout << query->get_embedding_ptr()[i] << " " ;
        // }
        // std::cout << std::endl;
        
        uint32_t K = query->get_K();
        uint32_t L = query->get_L();
	// std::cout << query->get_candidate_queue() << std::endl;
        // std::cout << "query with id " << query->get_query_id() << " has K "
                  // << query->get_K() << " " << query->get_L() << " "
        // << query->get_candidate_queue() << std::endl;
        std::shared_ptr<uint64_t[]> result_64(new uint64_t[K]);
        std::shared_ptr<uint32_t[]> result_32(new uint32_t[K]);



        index_lock.lock();
#ifdef IN_MEM
	// std::cout << "hello searching in mem" << std::endl;
        parent->index->search_global_baseline(typed_ctxt, query, K, L,
                                              result_64.get(), nullptr);
#elif defined(DISK_FS_DISKANN_WRAPPER)
	parent->index->search(query->get_embedding_ptr(), K, L, result_64.get(), nullptr);
#elif defined(DISK_FS_DISTRIBUTED)
        if (!initialized_thread_data) {
	  // this is safe because every data structure invovled has internal locks
          parent->index->setup_search_thread_data(4096);
	  initialized_thread_data = true;
        }

	retset->start_new_query(query_id);
	
	parent->index->search(typed_ctxt, query->get_embedding_ptr(), query->get_query_id(), my_thread_id , K, L, result_64.get(), nullptr, query->get_candidate_queue(), retset);
	retset->clear();
#elif defined(DISK_KV)
	// std::cout << "disk kv search called " << std::endl;
	parent->index->search_pq_fs(typed_ctxt, query->get_embedding_ptr(), K, L, result_64.get(), nullptr, query->get_candidate_queue());
#endif
        index_lock.unlock();
        for (uint32_t i = 0; i < K; i++) {
	  result_32.get()[i] = result_64.get()[i];
        }
        parent->batch_thread->push_ann_result(
            query->get_query_id(), query->get_client_node_id(), K, L,
					      std::move(result_32), parent->cluster_id);
      }
    }

  public:
    SearchThread(uint64_t thread_id, GlobalSearchOCDPO<data_type> *parent)
    : my_thread_id(thread_id), parent(parent) {}

    void push_compute_result(
				std::shared_ptr<ComputeResult<data_type>> compute_result) {
      if (retset == nullptr) {
	throw std::runtime_error("concurrent neighbor pq retset is nullptr");
      }
      // std::cout << "compute result arrived " << compute_result->get_query_id()
      // << std::endl;
      retset->check_query_id_insert_nbrs(
          compute_result->get_query_id(), compute_result->get_num_neighbors(),
					 compute_result->get_nbr_ids(), compute_result->get_nbr_distances());
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

      std::shared_lock<std::shared_mutex> index_lock(parent->index_mutex,
                                                     std::defer_lock);
      bool initialized_thread_data = false;
      while (running) {
        // std::cout << "compute queue size " << compute_queue.size() << std::endl;
        // std::cout << "query map size " << query_map.size() << std::endl;
	std::shared_ptr<EmbeddingQuery<data_type>> query_emb_ptr;
        queue_lock.lock();
        while (compute_queue.empty()) {
	  compute_queue_cv.wait(queue_lock);
        }
        if (!running)
          break;
        
        compute_query_t compute_query = compute_queue.front();
        compute_queue.pop();
	queue_lock.unlock();

	map_lock.lock();
        if (query_map.count(compute_query.query_id) == 0) {
          // this should be an error because trigger put is an atomic multicast
          std::stringstream err;
          err << "[compute distance] " << "cluster "
              << static_cast<int>(parent->cluster_id)
              << " query embedding for compute query " << compute_query.query_id
          << " has not arrived" << std::endl;
          std::cout << err.str() << std::endl;
          // throw std::runtime_error(err.str());
          queue_lock.lock();
          compute_queue.emplace(std::move(compute_query));
          queue_lock.unlock();
          
        } else {
          query_emb_ptr = query_map[compute_query.query_id];
        }

        map_lock.unlock();
        // it has been re added to queue
        if (query_emb_ptr == nullptr)
          continue;
        index_lock.lock();
        if (!initialized_thread_data) {
	  // this is safe because every data structure invovled has internal locks
          parent->index->setup_compute_thread_data();
	  initialized_thread_data = true;
        }
        std::shared_ptr<compute_result_t<data_type>> compute_result =
            parent->index->execute_compute_query(typed_ctxt, compute_query,
                                                 query_emb_ptr);
        index_lock.unlock();
        
        parent->batch_thread->push_compute_res(compute_result);
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
      compute_query_t empty_q;
      compute_queue.push(empty_q);
      compute_queue_cv.notify_all();
    }
    void push_compute_query(compute_query_t compute_query) {
      std::scoped_lock<std::mutex> lock(compute_queue_mutex);
      compute_queue.emplace(std::move(compute_query));
      compute_queue_cv.notify_all();
    }

    void push_embedding_query(std::shared_ptr<EmbeddingQuery<data_type>> query) {
      std::scoped_lock<std::mutex> lock(query_map_mutex);
      // if (query_map.count(query->get_query_id()) == 0){
	query_map[query->get_query_id()] = query;
      // } else {
        // throw std::runtime_error("query " +
        // std::to_string(query->get_query_id()) + " already pushed to query_map
        // for cluster " + std::to_string(parent->cluster_id));
      // }
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
      std::variant<std::shared_ptr<compute_result_t<data_type>>,
                   compute_query_t>
          msg;
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

                message_batcher.push_compute_result(std::move(
                    std::get<std::shared_ptr<compute_result_t<data_type>>>(
									   messages->at(i).msg)));
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
            obj.key = UDL2_PATHNAME_CLUSTER + std::to_string(cluster_id);
            typed_ctxt->get_service_client_ref()
                .put_and_forget<UDL2_OBJ_POOL_TYPE>(
                    obj, UDL2_SUBGROUP_INDEX, static_cast<uint32_t>(cluster_id),
						    true);
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

    void push_compute_query(compute_query_t query) {
      std::scoped_lock<std::mutex> lock(messages_mutex);
      if (cluster_messages.count(query.cluster_receiver_id) == 0) {
        cluster_messages[query.cluster_receiver_id] =
          std::make_unique<std::vector<message>>();
        cluster_messages[query.cluster_receiver_id]->reserve(parent->max_batch_size);
      }
      message msg = {message_type::COMPUTE_QUERY, std::move(query)};
      cluster_messages[query.cluster_receiver_id]->emplace_back(std::move(msg));
    }

    void push_compute_res(std::shared_ptr<compute_result_t<data_type>> res) {
      
      std::scoped_lock<std::mutex> lock(messages_mutex);      
      uint8_t receiver_id = res->cluster_receiver_id;
      if (cluster_messages.count(res->cluster_receiver_id) == 0) {
        cluster_messages[receiver_id] =
          std::make_unique<std::vector<message>>();
        cluster_messages[receiver_id]->reserve(parent->max_batch_size);
      }
      message mes = {message_type::COMPUTE_RESULT, std::move(res)};
      cluster_messages[receiver_id]->emplace_back(std::move(mes));
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
      if (cluster_messages.count(0) == 0) {
        cluster_messages[0] = std::make_unique<std::vector<message>>();
      }
      message mes;
      cluster_messages[0]->push_back(mes);
      messages_cv.notify_all();
    }
  };


  static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

  mutable std::shared_mutex index_mutex;
  // search threads will use shared_lock, initialization will use unique lock
  std::atomic<bool> initialized_index{false};
#ifdef IN_MEM
  std::unique_ptr<GlobalIndex<data_type>> index;
#elif defined(DISK_FS_DISKANN_WRAPPER)
  std::unique_ptr<SSDIndexFileSystem<data_type>> index;
#elif defined(DISK_FS_DISTRIBUTED)
  std::unique_ptr<SSDIndex<data_type>> index;  
#elif defined(DISK_KV)
  std::unique_ptr<SSDIndexKV<data_type>> index;
#endif


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
  std::string cluster_data_prefix = UDL2_DATA_PREFIX_CLUSTER;
  std::string cluster_search_prefix = UDL2_PATHNAME_CLUSTER;

  std::string cluster_assignment_bin_file;
  std::string pq_table_bin;
  std::string pq_compressed_vectors;    
  std::string index_path_prefix;
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

  void validate_compute_result(
			       const std::shared_ptr<ComputeResult<data_type>> &compute_res) {
    if (compute_res->get_cluster_receiver_id() != this->cluster_id) {
      throw std::runtime_error(
          "Global UDL: " +
          std::to_string(compute_res->get_cluster_receiver_id()) +
          " different from this cluster id  " + std::to_string(cluster_id));
    }
    if (compute_res->get_receiver_thread_id() >= search_threads.size()) {
      std::stringstream err;
      err << "compute result receiver thread id too big "
          << compute_res->get_receiver_thread_id()
          << " compared to number of search threads " << search_threads.size()
      << std::endl;
      throw std::runtime_error(err.str());
    }
  }
  
public:
  void ocdpo_handler(const node_id_t sender,
                     const std::string &object_pool_pathname,
                     const std::string &key_string,
                     const ObjectWithStringKey &object, const emit_func_t &emit,
                     DefaultCascadeContextType *typed_ctxt,
                     uint32_t worker_id) override {
    // std::cout << "Global index called " << key_string << std::endl;
    // can probably optimize this part by doing shared ptr for compute query and
    // compute result


    // double check locking with atomic bool should be good according to
    // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rconc-double-pattern
    
    if (initialized_index == false) {
      std::unique_lock lock(index_mutex);
      if (initialized_index == false) {
	// std::cout << "index not initialized" << std::endl;
        cluster_id = get_cluster_id(key_string);
	cluster_data_prefix += std::to_string(cluster_id);
	cluster_search_prefix += std::to_string(cluster_id);
	// std::cout << cluster_data_prefix << std::endl;
        // std::cout << cluster_search_prefix << std::endl;
        // std::cout << "cluster index is " << static_cast<int>(cluster_id) << " " << key_string
        // << std::endl;
	dist_fn.reset(
		      (diskann::Distance<data_type> *)
              diskann::get_distance_function<data_type>(diskann::Metric::L2));
	// std::cout << "started making global idnex" << std::endl;
#ifdef IN_MEM
	// std::cout << "createing in mem index" << std::endl;
	this->index = std::make_unique<GlobalIndex<data_type>>(
							       this->dim, this->cluster_id, cluster_assignment_bin_file,
							       cluster_data_prefix);
#elif defined(DISK_FS_DISKANN_WRAPPER)
        // std::cout << "start creating SSDIndexFileSystem" << cluster_id << " "
        // << key_string << std::endl;
        // std::cout << "index path prefix " << this->index_path_prefix
        // << std::endl;
        // std::cout << "num search threads " << this->num_search_threads
        // << std::endl;        
	this->index = std::make_unique<SSDIndexFileSystem<data_type>>(
								      this->index_path_prefix, this->num_search_threads);
        // std::cout << "done creating SSDIndexFileSystem" << cluster_id << " "
        // << key_string << std::endl;
#elif defined(DISK_FS_DISTRIBUTED)

        std::string cluster_files_path_prefix =
          index_path_prefix + std::to_string(cluster_id);
        uint32_t num_compute_threads = 1;
        this->index = std::make_unique<SSDIndex<data_type>>(

            cluster_files_path_prefix, cluster_id, cluster_data_prefix,
            cluster_assignment_bin_file, pq_table_bin, num_search_threads, num_compute_threads,
							    [this](compute_query_t query) { this->batch_thread->push_compute_query(query); }, pq_compressed_vectors);
	// std::cout << " done createing diskfs " << std::endl;
#elif defined(DISK_KV)

	// std::cout << "start creating SSDINDEXKV" << std::endl;
        this->index = std::make_unique<SSDIndexKV<data_type>>(
            this->index_path_prefix, cluster_data_prefix,
							      this->num_search_threads);
        // std::cout << "done creating SSDINDEXKV" << std::endl;
#endif
        initialized_index = true;
      }
    }
    if (get_cluster_id(key_string) != cluster_id) {
      std::stringstream err;
      err << key_string << "doesn't belong to cluster "
      << static_cast<int>(cluster_id) << std::endl;
      std::cout << err.str() << std::endl;
      throw std::runtime_error(err.str());
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
    for (std::shared_ptr<EmbeddingQuery<data_type>> &emb_query :
         manager.get_embedding_queries()) {
      validate_emb_query(emb_query);
      distance_compute_thread->push_embedding_query(std::move(emb_query));
    }

    for (compute_query_t &query : manager.get_compute_queries()) {
      validate_compute_query(query);
      distance_compute_thread->push_compute_query(std::move(query));
    }

    for (std::shared_ptr<ComputeResult<data_type>> &result : manager.get_compute_results()) {
      validate_compute_result(result);
      search_threads[result->get_receiver_thread_id()]->push_compute_result(
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
    std::cout << "global search udl id is  " << my_id << std::endl;

    std::cout << "members "
              << typed_ctxt->get_service_client_ref().get_members()
    << std::endl;
    std::cout << "my shard "
    << typed_ctxt->get_service_client_ref().get_my_shard<VolatileCascadeStoreWithStringKey>(UDL2_SUBGROUP_INDEX)
    << std::endl;    
    try {
      if (config.contains("cluster_assignment_bin_file")) {
        this->cluster_assignment_bin_file =
          config["cluster_assignment_bin_file"].get<std::string>();
      }

      if (config.contains("pq_table_bin")) {
        this->pq_table_bin =
          config["pq_table_bin"].get<std::string>();
      }

      if (config.contains("pq_compressed_vectors")) {
        this->pq_compressed_vectors =
          config["pq_compressed_vectors"].get<std::string>();
      }
      

      if (config.contains("index_path_prefix")) {
        this->index_path_prefix =
          config["index_path_prefix"].get<std::string>();
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

    this->distance_compute_thread = std::make_unique<DistanceComputeThread>(this->my_id, this);
    this->distance_compute_thread->start(typed_ctxt);
    std::cout << "hello 4" << std::endl;
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
