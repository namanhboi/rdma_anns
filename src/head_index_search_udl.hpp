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
namespace derecho {
namespace cascade {

#define MY_UUID "69eb06e2-017c-481b-8534-2e5dac301949"
#define MY_DESC                                                                \
  "UDL for searching the head index to find good starting points in clusters " \
  "for greedy/beam search udl"
#define DATA_TYPE "float"

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
    // if we are doing cotra then we follow paper.
    std::unordered_map<uint8_t, std::vector<uint32_t>>
    determine_candidate_queues(
			       std::unordered_map<uint8_t, std::vector<uint32_t>> &starting_points) {
      std::unordered_map<uint8_t, std::vector<uint32_t>>
      candidate_queues_per_cluster;
#ifdef GLOBAL_BASELINE
      // choose the cluster with the most candidate points
      int max_size = 0;
      uint8_t chosen_cluster;
      for (const auto &[cluster_id, starting_points_ptr] : starting_points) {
        if (starting_points_ptr.size() > max_size) {
          max_size = starting_points_ptr.size();
          chosen_cluster = cluster_id;
        }
      }
      candidate_queues_per_cluster[chosen_cluster] =
        starting_points[chosen_cluster];
#else
      throw std::runtime_error("GLOBAL_BASELINE not defined, but I haven't yet implemented the cotra navigation index search");
#endif
      return candidate_queues_per_cluster;
    }


    void main_loop() {
      std::unique_lock<std::mutex> lock(query_queue_mutex, std::defer_lock);
      while (running) {
        lock.lock();

        if (query_queue.empty()) {
	  query_queue_cv.wait(lock);
        }
        
        if (!running)
          break;
	std::shared_ptr<EmbeddingQuery<data_type>> query = query_queue.front();
        query_queue.pop();
        lock.unlock();
        // do search here
        std::unordered_map<uint8_t, std::vector<uint32_t>>
        candidate_queues;
        try {
          candidate_queues = head_index_search(query);
        } catch (std::exception &e) {
          std::cout << "exception from head_index_search " << e.what()
          << std::endl;
          throw e;
        }
	// std::cout << "done getting a candidate queue" << std::endl;
        std::unordered_map<uint8_t, std::vector<uint32_t>>
            final_candidate_queue =
              determine_candidate_queues(candidate_queues);
	// std::cout << "done determining candidate quuee" << std::endl;
        parent->batch_thread->push(final_candidate_queue, std::move(query));
	std::cout << "pushed a batch" << std::endl;
      }

    }

    // searches the head index and returns a the (cluster (where the node
    // belongs + node id), with this info, you can trigger computation on next
    // udl. K = 1 for search
    std::unordered_map<uint8_t, std::vector<uint32_t>>
    head_index_search(std::shared_ptr<EmbeddingQuery<data_type>> &query) {
      std::vector<uint32_t> search_id_results(parent->K);
      std::vector<float> search_dist_results(parent->K);
      // std::cout << "start search on query " << query->get_query_id()
      // << std::endl;
      const data_type *emb = query->get_embedding_ptr();
      
      if (emb == nullptr) throw std::runtime_error("embedding nullptr");
      auto [hops, dist_cmps] = parent->head_index->search(
          emb, parent->K, parent->L,
							  search_id_results.data());
      // std::cout << "done search on query " << query->get_query_id() << std::endl;

      std::unordered_map<uint8_t, std::vector<uint32_t>> res;
      uint8_t cluster_0 = 0;
      res[cluster_0] = std::vector<uint32_t>();
      for (int i = 0; i < parent->K; i++) {
	uint8_t cluster_id = (0);
        if (res.count(cluster_id) == 0) {
          res[cluster_id] = std::vector<uint32_t>();
        }
#ifndef HEAD_INDEX_TEST
        res[cluster_id].push_back(parent->id_mapping[search_id_results[i]]);
#else
        res[cluster_id].push_back(search_id_results[i]);
#endif
      }
      return res;
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
    
    std::condition_variable_any cluster_queue_cv;
    std::mutex cluster_queue_mutex;

    // for now, the main loop will just put the search results into emit_key_prefix_test = /anns/results
    void main_loop(DefaultCascadeContextType *typed_ctxt) {
      // TODO
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

        if(empty){
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

#ifndef HEAD_INDEX_TEST
        // need to fix
        for (auto &[cluster_id, queries_and_cand_q] : to_send) {
          uint64_t num_sent = 0;
          uint64_t total = queries_and_cand_q->size();
          while (num_sent < total) {
            uint64_t left = total - num_sent;
            uint64_t batch_size = std::min(parent->max_batch_size, left);
            GreedySearchQueryBatcher<data_type> batcher(
							queries_and_cand_q->at(0).first->get_dim());
            for (uint64_t i = num_sent; i < num_sent + batch_size; i++) {
	      batcher.add_query(cluster_id, std::move(queries_and_cand_q->at(i).second), queries_and_cand_q->at(i).first);
            }
            batcher.serialize();
            ObjectWithStringKey obj;
            obj.blob = std::move(*batcher.get_blob());
            obj.previous_version = INVALID_VERSION;
	    obj.previous_version_by_key = INVALID_VERSION;
            obj.key = parent->emit_key_prefix + "/cluster" +
                      std::to_string(static_cast<int>(cluster_id));
            typed_ctxt->get_service_client_ref().put_and_forget(obj);
            num_sent += batch_size;
          }
        }
#else
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
            std::tuple<uint8_t, std::shared_ptr<EmbeddingQuery<data_type>>,
                       std::vector<uint32_t>>
                query_tuple = std::make_tuple(cluster_id, std::move(query),
                                              candidate_queue);
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
            std::string client_id_pool_path = parent->notification_test_prefix +
                                              "/" +
                                              std::to_string(client_node_id);
            std::cout << "notifying " << client_id_pool_path << std::endl;
            typed_ctxt->get_service_client_ref().notify(
							*(batcher.get_blob()), client_id_pool_path, client_node_id);
            num_sent += batch_size;
            std::cout << " done notifying " << client_id_pool_path << std::endl;
          }
          std::cout << "done sending queries for this batch" << std::endl;
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


    void push(const std::unordered_map<uint8_t, std::vector<uint32_t>>
                  &candidate_queues,
              std::shared_ptr<EmbeddingQuery<data_type>> query) {
      std::unique_lock<std::mutex> lock(cluster_queue_mutex);
      for (auto &[cluster_id, candidate_q] : candidate_queues) {
        if (cluster_queue.count(cluster_id) == 0) {
          cluster_queue[cluster_id] = std::make_unique<
              std::vector<std::pair<std::shared_ptr<EmbeddingQuery<data_type>>,
                                    std::vector<uint32_t>>>>();
          cluster_queue[cluster_id]->reserve(parent->max_batch_size);
        }
        cluster_queue[cluster_id]->emplace_back(query, candidate_q);
      }
      cluster_queue_cv.notify_all();
    }
  };
  uint32_t L = 20;
  uint32_t K = 10;
  uint32_t num_pts;
  uint32_t max_deg;
  uint32_t dim;
  uint32_t aligned_dim;  
  uint32_t start_node = 353;

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


  std::string head_index_header_key = "/anns/head_index/header";
  std::string vector_id_mapping_key = "/anns/head_index/mapping";
  // mapping of the vector id in the head index to the ones in the actual graph.
  // also stores on 1 byte for each vector, its cluster assignment.

  int my_id = -1; // id of this node, logging purpose.
  

  std::string head_index_prefix = "/anns/head_index";
  
  std::string notification_test_prefix = "/anns/head_index_results";
  std::string emit_key_prefix = "/anns/graph"; // will need to look into this,
  
  uint64_t min_batch_size = 1;
  uint64_t max_batch_size = 10;
  uint64_t batch_time_us = 1000;

  std::string index_path;


  bool retrieve_and_cache_head_index_fs(DefaultCascadeContextType *typed_ctxt) {
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
    head_index->load(index_path.c_str(), num_search_threads, L);
    std::cout << "Index loaded" << std::endl;
  }


  /*this functions gets the data store data to create the datastore and also the
    neigbors of each elements to add iteratively to the graph store.
  */
  // bool retrieve_and_cache_head_index_kv(DefaultCascadeContextType *typed_ctxt) {
  //   std::cout << "cache index called " << std::endl;
  //   auto header = typed_ctxt->get_service_client_ref().get(
  // 							   head_index_header_key, CURRENT_VERSION, true);
  //   std::cout << "called to get header " << std::endl;
  //   auto &header_reply = header.get().begin()->second.get();
  //   Blob header_blob = std::move(const_cast<Blob &>(header_reply.blob));
  //   // header_blob.memory_mode = object_memory_mode_t::EMPLACED; // why need emplaced?


  //   uint32_t num_pts = *reinterpret_cast<const uint32_t *>(header_blob.bytes);
  //   uint32_t num_dim = *reinterpret_cast<const uint32_t *>(header_blob.bytes +
  //                                                          sizeof(uint32_t));
  //   uint32_t max_deg = *reinterpret_cast<const uint32_t *>(header_blob.bytes +
  //                                                          sizeof(uint32_t) * 2);
  //   std::cout << "num pts: " << num_pts << ", num_dim" << num_dim << ",max_deg "
  //   << max_deg << std::endl;
    
  //   if (num_dim != this->dim)
  //     throw std::runtime_error("Number of dim from data store is different "
  //                              "from the dim specified in config file");
  //   this->num_pts = num_pts;
  //   this->max_deg = max_deg;
  //   this->aligned_dim = ROUND_UP(dim, 8);

  //   std::unique_ptr<diskann::InMemGraphStore> graph_store =
  //     std::make_unique<diskann::InMemGraphStore>(num_pts, max_deg);


  //   data_type * emb_buf = new data_type[num_dim * num_pts];
  //   for (uint32_t i = 0; i < num_pts; i++) {
  //     std::cout << "loading point " << i << std::endl;
  //     std::string nbr_key = head_index_prefix + "/nbr_" + std::to_string(i);
  //     std::string emb_key = head_index_prefix + "/emb_" + std::to_string(i);   
  //     auto nbr_get = typed_ctxt->get_service_client_ref().get(nbr_key, CURRENT_VERSION, true);
  //     auto &nbr_reply = nbr_get.get().begin()->second.get();
  //     Blob nbr_blob = std::move(const_cast<Blob &>(nbr_reply.blob));
  //     // nbr_blob.memory_mode = object_memory_mode_t::EMPLACED; // why the need
  //     // for emplaced, will need to call free later
  //     uint32_t num_nbr = nbr_blob.size / sizeof(uint32_t);
  //     const uint32_t *start_nbr =
  //       reinterpret_cast<const uint32_t *>(nbr_blob.bytes);
  //     const uint32_t *end_nbr = start_nbr + num_nbr;
  //     std::vector<uint32_t> neighbors(start_nbr, end_nbr);
  //     std::cout << neighbors.size() << std::endl;
  //     if (neighbors.size() != num_nbr || neighbors.size()> max_deg)
  //       throw std::runtime_error("wrong nbr size");
  //     graph_store->set_neighbours(i, neighbors);

  //     auto emb_get = typed_ctxt->get_service_client_ref().get(emb_key, CURRENT_VERSION, true);
  //     auto &emb_reply = emb_get.get().begin()->second.get();
  //     Blob emb_blob = std::move(const_cast<Blob &>(emb_reply.blob));
  //     // emb_blob.memory_mode = object_memory_mode_t::EMPLACED; // why 
  //     if (emb_blob.size != sizeof(data_type) * dim)
  //       throw std::runtime_error("wrong size emb");
  //     std::memcpy(emb_buf + i * dim, emb_blob.bytes , emb_blob.size);
  //   }
  //   // auto get_data_store_results = typed_ctxt->get_service_client_ref().get(
  // 									   // data_store_key, CURRENT_VERSION, true);
  //   // auto &reply = get_data_store_results.get().begin()->second.get();
  //   // Blob data_blob = std::move(const_cast<Blob &>(reply.blob));
  //   // data_blob.memory_mode = object_memory_mode_t::EMPLACED; // what tf is emplaced?
  //   // uint32_t num_pts = *reinterpret_cast<const uint32_t *>(data_blob.bytes);
  //   // uint32_t num_dim =
  //     // *reinterpret_cast<const uint32_t *>(data_blob.bytes + sizeof(uint32_t));


  //   // std::cout << "Loaded in " << num_pts << " points with dim "  << num_dim << std::endl;
  //   // auto data_store_ptr =
  //       // const_cast<data_type *>(reinterpret_cast<const data_type *>(
  // 								    // data_blob.bytes + sizeof(uint32_t) * 2));

  //   std::unique_ptr<diskann::Distance<data_type>> dist;
  //   dist.reset((diskann::Distance<data_type> *)diskann::get_distance_function<data_type>(diskann::Metric::L2));
  //   std::shared_ptr<diskann::InMemDataStore<data_type>> data_store =
  //       std::make_shared<diskann::InMemDataStore<data_type>>(num_pts, num_dim,
  //                                                            std::move(dist));
  //   data_store->populate_data(emb_buf, num_pts);

  // uint32_t filter_list_size = 0;
  // auto index_build_params = diskann::IndexWriteParametersBuilder(50, 32)
  //                               .with_filter_list_size(0)
  //                               .with_alpha(1.2)
  //                               .with_saturate_graph(false)
  //                               .with_num_threads(16)
  //                               .build();
  // auto filter_params = diskann::IndexFilterParamsBuilder()
  //                          .with_universal_label("")
  //                          .with_label_file("")
  //                          .build();

  // //bruh this caused a whole ass wild goose chase
  // auto search_params = diskann::IndexSearchParams(20, 16);

  // auto config =
  //     diskann::IndexConfigBuilder()
  //         .with_metric(diskann::Metric::L2)
  //         .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
  //         .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
  //         .with_dimension(dim)
  //         .with_max_points(num_pts) 
  //         .is_dynamic_index(false)
  //         .is_enable_tags(false)
  //         .is_pq_dist_build(false)
  //         .is_use_opq(false)
  //         .is_filtered(false)
  //         .with_num_pq_chunks(0)
  //         .with_num_frozen_pts(0)
  //         .with_data_type(diskann_type_to_name<data_type>())
  //         .with_index_write_params(index_build_params)
  //         .with_index_search_params(search_params)
  //         .build();

  // std::shared_ptr<diskann::AbstractDataStore<data_type>> pq_data_store = data_store;
  // head_index = std::make_unique<diskann::Index<data_type>>(
  //     config, data_store, std::move(graph_store), pq_data_store,
  // 							   this->start_node);
    
  //   cached_head_index = true;
  //   std::cout << "Done loading head index data";
  //   delete[] emb_buf;
  //   return true;

  // }

  void ocdpo_handler(const node_id_t sender,
                     const std::string &object_pool_pathname,
                     const std::string &key_string,
                     const ObjectWithStringKey &object, const emit_func_t &emit,
                     DefaultCascadeContextType *typed_ctxt,
                     uint32_t worker_id) override {
    std::cout << "Head index called " << std::endl;
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
    
    // TODO
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
      if (config.contains("start_node")) this->start_node = config["start_node"].get<uint32_t>();

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
