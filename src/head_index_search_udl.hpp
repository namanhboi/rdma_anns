#include <algorithm>
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
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include "index_factory.h"
#include "udl_path_and_index.hpp"
#include "utils.h"
namespace derecho {
namespace cascade {

#define MY_UUID "69eb06e2-017c-481b-8534-2e5dac301949"
#define MY_DESC                                                                \
  "UDL for searching the head index to find good starting points in clusters " \
  "for greedy/beam search udl"
#define DATA_TYPE "float"

#define MAX_PTS 1'000'000
std::string get_uuid();

std::string get_description();


template<typename data_type>
class HeadIndexSearchOCDPO : public DefaultOffCriticalDataPathObserver {
  class HeadIndexSearchThread {
  private:
    uint64_t my_thread_id;
    HeadIndexSearchOCDPO *parent;
    std::thread real_thread;
    bool running = false;

    std::condition_variable_any query_queue_cv;
    std::mutex query_queue_mutex;
    std::queue<std::shared_ptr<EmbeddingQuery<data_type>>> query_queue;

    // if we are doing global baseline then we just put the starting points from
    // the cluster with the most starting points into the candidate queue
    // if we are doing cotra then we follow paper and put starting points of
    // multiple main partitions
    std::unordered_map<uint8_t, std::vector<uint32_t>>
    determine_candidate_queues(
			       const std::vector<uint32_t> &search_results) {
      std::unordered_map<uint8_t, std::vector<uint32_t>>
      candidate_queues_per_cluster;

      uint32_t max_elements = 0;
      uint8_t max_element_cluster_id = 0;
      for (const uint32_t &node_id : search_results) {
#ifdef TEST_UDL1
        const uint8_t &cluster_id = 0;
#else
        const uint8_t &cluster_id = parent->cluster_assignment[node_id];
#endif
        if (candidate_queues_per_cluster.count(cluster_id) == 0) {
          candidate_queues_per_cluster[cluster_id] = std::vector<uint32_t>();
        }
        candidate_queues_per_cluster[cluster_id].push_back(node_id);
        if (candidate_queues_per_cluster[cluster_id].size() > max_elements) {
	  max_elements = candidate_queues_per_cluster[cluster_id].size();
	  max_element_cluster_id = cluster_id;
        }
      }
      return {{max_element_cluster_id,
               candidate_queues_per_cluster[max_element_cluster_id]}};
    }

    void main_loop() {
      std::unique_lock<std::mutex> lock(query_queue_mutex, std::defer_lock);
      while (running) {
        lock.lock();

        while (query_queue.empty()) {
	  query_queue_cv.wait(lock);
        }
        if (!running)
          break;
	std::shared_ptr<EmbeddingQuery<data_type>> query = query_queue.front();
        query_queue.pop();
        if (query == nullptr) {
          throw std::runtime_error(
              "encounter nullptr query even thought running = true, meaning it "
              "didn't come from signal stop");
        }
        lock.unlock();

	std::vector<uint32_t> search_results;
        try {
          head_index_search(query, search_results);
        } catch (std::exception &e) {
          std::cout << "exception from head_index_search " << e.what()
          << std::endl;
          throw e;
        }
        parent->batch_thread->push(
				   determine_candidate_queues(std::move(search_results)), query);
      }
    }

    void head_index_search(std::shared_ptr<EmbeddingQuery<data_type>> &query, std::vector<uint32_t> &search_results) {
      search_results.resize(query->get_K());
      const data_type *emb = query->get_embedding_ptr();
      
      if (emb == nullptr) throw std::runtime_error("embedding nullptr");
      auto [hops, dist_cmps] = parent->head_index->search(
							  emb, query->get_K(), query->get_L(), search_results.data());
    }

  public:
    HeadIndexSearchThread(uint64_t thread_id, HeadIndexSearchOCDPO *parent)
    : my_thread_id(thread_id), parent(parent) {}
    void start() {
      running = true;
      real_thread = std::thread(&HeadIndexSearchThread::main_loop, this);
    }
    void join() {
      if (real_thread.joinable()) real_thread.join();
    }
    void signal_stop() {
      std::scoped_lock l(query_queue_mutex);
      running = false;
      query_queue.push(nullptr);
      query_queue_cv.notify_all();
    }

    void push(std::shared_ptr<EmbeddingQuery<data_type>> query) {
      std::scoped_lock l(query_queue_mutex);
      query_queue.push(query);
      query_queue_cv.notify_all();      
    }      
  };
  uint64_t num_search_threads = 1;
  uint64_t next_search_thread = 0;
  std::vector<std::unique_ptr<HeadIndexSearchThread>> search_threads;

  /**
     this thread batches the result from the search thread based on the
     cluster_id and sends them to the next udl.
   */
  class BatchingThread {
  private:
    uint64_t my_thread_id;
    HeadIndexSearchOCDPO *parent;
    std::thread real_thread;
    bool running = false;

    std::unordered_map<uint8_t,
                       std::unique_ptr<std::vector<
                           std::pair<std::shared_ptr<EmbeddingQuery<data_type>>,
                                     std::vector<uint32_t>>>>>
        cluster_queue; // for each cluster, there is the list of the starting
                       // candidates in the candidates queue and query to batch.
    // if the candidate list is not empty then form GreedySearchQuery, it it's
    // empty then form a EmbeddingQuery
    
    std::condition_variable_any cluster_queue_cv;
    std::mutex cluster_queue_mutex;

    void main_loop(DefaultCascadeContextType *typed_ctxt) {
      std::unique_lock<std::mutex> lock(cluster_queue_mutex, std::defer_lock);
      std::unordered_map<uint8_t, std::chrono::steady_clock::time_point>
      wait_time;
      auto batch_time = std::chrono::microseconds(parent->batch_time_us);
      while (running) {
        lock.lock();
        bool empty = true;
        for(auto& item : cluster_queue){
          if(!(item.second->empty())){
            empty = false;
            break;
          }
        }

        if (empty){
            cluster_queue_cv.wait_for(lock,batch_time);
        }

        if (!running)
          break;
        std::unordered_map<
            uint8_t, std::unique_ptr<std::vector<
                           std::pair<std::shared_ptr<EmbeddingQuery<data_type>>,
                                     std::vector<uint32_t>>>>>
        to_send;
        
        auto now = std::chrono::steady_clock::now();
        for (auto &[cluster_id, queries_and_cand_q] : cluster_queue) {
	  if (wait_time.count(cluster_id) == 0) 
            wait_time[cluster_id] = now;
	  if (queries_and_cand_q->size() >= parent->min_batch_size || ((now-wait_time[cluster_id]) >= batch_time)) {
            to_send[cluster_id] = std::move(queries_and_cand_q);
            cluster_queue[cluster_id] = std::make_unique<
              std::vector<std::pair<std::shared_ptr<EmbeddingQuery<data_type>>,
                                    std::vector<uint32_t>>>>();
            cluster_queue[cluster_id]->reserve(parent->max_batch_size);
          }
          
        }
        lock.unlock();

#ifndef TEST_UDL1
        // this is the actual data sending pipeline to the global search udl
        for (auto &[cluster_id, queries_and_cand_q] : to_send) {
          uint64_t num_sent = 0;
          uint64_t total = queries_and_cand_q->size();
          while (num_sent < total) {
            uint64_t left = total - num_sent;
            uint64_t batch_size = std::min(parent->max_batch_size, left);
            GlobalSearchMessageBatcher<data_type> batcher(
							  queries_and_cand_q->at(0).first->get_dim());

            for (uint64_t i = num_sent; i < num_sent + batch_size; i++) {
              if (queries_and_cand_q->at(i).second.size() == 0) {
                batcher.push_embedding_query(
					     std::move(queries_and_cand_q->at(i).first));
              } else {
                greedy_query_t<data_type> search_query(
                    cluster_id, std::move(queries_and_cand_q->at(i).second),
						       std::move(queries_and_cand_q->at(i).first));
                batcher.push_search_query(std::move(search_query));
              }
            }
            batcher.serialize();
            ObjectWithStringKey obj;
            obj.blob = std::move(*batcher.get_blob());
            obj.previous_version = INVALID_VERSION;
            obj.previous_version_by_key = INVALID_VERSION;
            obj.key = UDL2_PATHNAME "/cluster_" + std::to_string(static_cast<int>(cluster_id));
            typed_ctxt->get_service_client_ref().put_and_forget(
								obj, true); // trigger put
            num_sent += batch_size;
          }
        }
#else
	// this is to test if head index is working correctly
        std::unordered_map<std::uint32_t,
                           std::unique_ptr<std::vector<greedy_query_t<data_type>>>>
        queries_by_client_id;
        for (auto &[cluster_id, queries_and_cand_q] : to_send) {
          for (auto &[query, candidate_queue] : *queries_and_cand_q) {
            uint32_t client_node_id = query->get_client_node_id();
            if (queries_by_client_id.count(client_node_id) == 0) {
              queries_by_client_id[client_node_id] =
                std::make_unique<std::vector<greedy_query_t<data_type>>>();
              queries_by_client_id[client_node_id]->reserve(parent->max_batch_size);
            }
            queries_by_client_id[client_node_id]->emplace_back(
							       cluster_id, std::move(candidate_queue), std::move(query));
          }
        }
	
        for (auto &[client_node_id, queries] : queries_by_client_id) {
          uint64_t num_sent = 0;
          uint64_t total = queries->size();
          while (num_sent < total) {
            uint32_t batch_size =
              std::min(parent->max_batch_size, total - num_sent);
            GlobalSearchMessageBatcher<data_type> batcher(parent->dim);
            for (uint64_t i = num_sent; i < num_sent + batch_size; i++) {
	      batcher.push_search_query(std::move(queries->at(i)));
            }
            batcher.serialize();
            std::string client_id_pool_path =
              RESULTS_OBJ_POOL_PREFIX "/" + std::to_string(client_node_id);
            
            
            // std::cout << "notifying " << client_id_pool_path << std::endl;
            typed_ctxt->get_service_client_ref().notify(
							*(batcher.get_blob()), client_id_pool_path, client_node_id);
            num_sent += batch_size;
          }
        }
#endif        
      }
    }

  public:
    BatchingThread(uint64_t thread_id, HeadIndexSearchOCDPO *parent) : my_thread_id(thread_id), parent(parent), running(false) {}
    void start(DefaultCascadeContextType *typed_ctxt) {
      running = true;
      this->real_thread = std::thread(&BatchingThread::main_loop, this, typed_ctxt);
    }
    void join() {
      if (real_thread.joinable()) {
        real_thread.join();
      }
    }
    void signal_stop() {
      std::scoped_lock<std::mutex> l(cluster_queue_mutex);
      running = false;
      cluster_queue_cv.notify_all();
    }


    void
    push(std::unordered_map<uint8_t, std::vector<uint32_t>> candidate_queues,
         std::shared_ptr<EmbeddingQuery<data_type>> query) {
      std::unique_lock<std::mutex> lock(cluster_queue_mutex);

#ifdef TEST_UDL1
      if (cluster_queue.count(0) ==0 ) {
        cluster_queue[0] = std::make_unique<
						     std::vector<std::pair<std::shared_ptr<EmbeddingQuery<data_type>>,
									   std::vector<uint32_t>>>>();
        cluster_queue[0]->reserve(parent->max_batch_size);
      }
      cluster_queue[0]->emplace_back(query, std::move(candidate_queues[0]));
#else
      for (uint8_t cluster_id; cluster_id < parent->num_clusters; cluster_id++) {
        if (cluster_queue.count(cluster_id) == 0) {
          cluster_queue[cluster_id] = std::make_unique<
              std::vector<std::pair<std::shared_ptr<EmbeddingQuery<data_type>>,
                                    std::vector<uint32_t>>>>();
          cluster_queue[cluster_id]->reserve(parent->max_batch_size);
        }
        if (candidate_queues.count(cluster_id) != 0) {
          cluster_queue[cluster_id]->emplace_back(
						  query, std::move(candidate_queues[cluster_id]));
          // no longer need the cand q so we just move it.
        } else {
          // pass it an empty vector, then the main batching loop will see that
          // the vector is empty and then send an EmbeddingQuery (for the
          // secondary partitions to do distance compute with) instead of the
          // GreedySearchQueries
          cluster_queue[cluster_id]->emplace_back(query,
                                                  std::vector<uint32_t>());
        }
      }
#endif
      cluster_queue_cv.notify_all();
    }
  };
  size_t num_pts;
  uint8_t num_clusters;
  uint32_t max_deg;
  uint32_t dim;
  uint32_t aligned_dim;  

  std::unique_ptr<diskann::AbstractIndex> head_index;
  bool cached_head_index = false; // if head index is loaded into mem or not



  // id_mapping is the mapping from the vector id of the head index to the
  // ids of the graph
  std::vector<uint32_t> id_mapping;
  // each byte represents the cluster assignment of the corresponding graph
  // vector id from the id_mapping. 
  std::vector<uint8_t> cluster_assignment; //TODO

  // data here: num_pts (uint32_t), num_dim (uint32_t), data....
  // std::string data_store_key = "/anns/head_index/data_store";
  // std::string graph_store_key =
  // "/anns/head_index/graph_store";


  int my_id = -1; // id of this node, logging purpose.
  

  std::string head_index_prefix = "/anns/head_index";
  
  std::string emit_key_prefix = "/anns/graph"; // will need to look into this,
  
  uint64_t min_batch_size = 1;
  uint64_t max_batch_size = 10;
  uint64_t batch_time_us = 1000;

  std::string index_path = "";
  std::string id_mapping_path = "";
  std::string cluster_assignment_path = "";

  void retrieve_and_cache_head_index_fs(DefaultCascadeContextType *typed_ctxt) {
    if (index_path == "") {
      throw std::runtime_error("index path not specified");
    }
    if (cluster_assignment_path == "") {
      throw std::runtime_error("cluster assignment path not specified");
    }
    if (id_mapping_path == "") {
      throw std::runtime_error("id mapping path not specified");
    }
    
    diskann::Metric metric = diskann::Metric::L2;
    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);

    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(dim)
                      .with_max_points(0)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(diskann_type_to_name<data_type>())
                      .is_dynamic_index(false)
                      .is_enable_tags(false)
                      .is_concurrent_consolidate(false)
                      .is_pq_dist_build(false)
                      .is_use_opq(false)
                      .with_num_pq_chunks(0)
                      .with_num_frozen_pts(num_frozen_pts)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    head_index = index_factory.create_instance();
    head_index->load(index_path.c_str(), num_search_threads, 20);
    std::cout << "Index loaded from " << index_path << std::endl;
    std::ifstream cluster_assignment_in(cluster_assignment_path,
                                        std::ios::binary);
    uint32_t whole_graph_num_pts;
    cluster_assignment_in.read((char *)&whole_graph_num_pts,
                               sizeof(whole_graph_num_pts));
    cluster_assignment_in.read((char *)&num_clusters, sizeof(num_clusters));

    cluster_assignment = std::vector<uint8_t>(whole_graph_num_pts);
    cluster_assignment_in.read((char *)cluster_assignment.data(),
                               whole_graph_num_pts * sizeof(uint8_t));

    id_mapping = std::vector<uint32_t>(MAX_PTS);
    size_t num_dim_file;
    size_t num_aligned_dim_file;
    uint32_t *ptr = id_mapping.data();
    diskann::load_bin<uint32_t>(id_mapping_path, ptr, num_pts, num_dim_file);
    assert(num_dim_file == 1);
    id_mapping.resize(num_pts);

    cached_head_index = true;
  }

  void ocdpo_handler(const node_id_t sender,
                     const std::string &object_pool_pathname,
                     const std::string &key_string,
                     const ObjectWithStringKey &object, const emit_func_t &emit,
                     DefaultCascadeContextType *typed_ctxt,
                     uint32_t worker_id) override {
    // std::cout << "Head index called " << std::endl;
    if (cached_head_index == false) {
      retrieve_and_cache_head_index_fs(typed_ctxt);
    }
    std::unique_ptr<EmbeddingQueryBatchManager<data_type>> batch_manager =
        std::make_unique<EmbeddingQueryBatchManager<data_type>>(
								object.blob.bytes, object.blob.size);
    for (auto &query : batch_manager->get_queries()) {
      if (query->get_dim() != dim) {
        throw std::runtime_error(
            "dimension of query differ from dim specified in dfgs.json" +
            std::to_string(query->get_dim()) + " vs " + std::to_string(dim));
      }
      search_threads[next_search_thread]->push(query);
      next_search_thread = (next_search_thread + 1) % num_search_threads;
    }
  }
  static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

public:
  std::unique_ptr<BatchingThread> batch_thread;

  static void initialize() {
    if (!ocdpo_ptr) {
      ocdpo_ptr = std::make_shared<HeadIndexSearchOCDPO<data_type>>();
    }
  }
  static auto get() { return ocdpo_ptr; }
  void set_config(DefaultCascadeContextType *typed_ctxt,
                  const nlohmann::json &config) {
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    try{
      if (config.contains("dim"))
        this->dim = config["dim"].get<int>();
      if (config.contains("num_search_threads")) {
        this->num_search_threads = config["num_search_threads"].get<int>();
      }

      if (config.contains("min_batch_size")) {
        this->min_batch_size = config["min_batch_size"].get<int>();
      }

      if (config.contains("max_batch_size")) {
        this->max_batch_size = config["max_batch_size"].get<int>();
      }
      
      if (config.contains("batch_time_us")) {
        this->batch_time_us = config["batch_time_us"].get<int>();
      }
      
      if (config.contains("batch_time_us")) {
        this->batch_time_us = config["batch_time_us"].get<int>();
      }
      
      if (config.contains("batch_time_us")) {
        this->batch_time_us = config["batch_time_us"].get<int>();
      }

      if (config.contains("index_path")) {
        this->index_path =
          config["index_path"].get<std::string>();
      }

      if (config.contains("id_mapping_path")) {
        this->id_mapping_path =
          config["id_mapping_path"].get<std::string>();
      }
      if (config.contains("cluster_assignment_path")) {
        this->cluster_assignment_path =
          config["cluster_assignment_path"].get<std::string>();
      }

    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert emb_dim or top_num_centroids from config" << std::endl;
        dbg_default_error("Failed to convert emb_dim or top_num_centroids from config, at centroids_search_udl.");
    }

    // start search threads
    for(uint64_t thread_id = 0; thread_id < this->num_search_threads; thread_id++) {
      search_threads.emplace_back(new HeadIndexSearchThread(thread_id,this));
    }
    for(auto& search_thread : search_threads) {
        search_thread->start();
    }

    // start batching thread
    this->batch_thread = std::make_unique<BatchingThread>(this->my_id, this);
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
  }    
};
template <typename data_type>
std::shared_ptr<OffCriticalDataPathObserver>
    HeadIndexSearchOCDPO<data_type>::ocdpo_ptr;

void initialize(ICascadeContext *ctxt);

std::shared_ptr<OffCriticalDataPathObserver>
get_observer(ICascadeContext *ctxt, const nlohmann::json &config);  

} // namespace cascade
} // namespace derecho
