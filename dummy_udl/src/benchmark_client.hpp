#pragma once

#include <cascade/service_client_api.hpp>
#include <cstdint>
#include <openssl/objects.h>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <iostream>
#include <atomic>
#include <unordered_map>
#include <shared_mutex>
#include "serialize_utils.hpp"
#include "udl_path_and_index.hpp"

using namespace derecho::cascade;
#define CLIENT_MAX_WAIT_TIME 1000

template<typename data_type>
class BenchmarkClient {
  class ClientThread {
  private:
    std::thread real_thread;
    ServiceClientAPI &capi = ServiceClientAPI::get_service_client();
    uint32_t node_id = capi.get_my_id();
    uint64_t batch_min_size = 0;
    uint64_t batch_max_size = 5;
    uint64_t batch_time_us = 500;
    uint32_t dim;
    bool running = false;

    std::mutex send_request_queue_mtx;
    std::condition_variable send_request_queue_cv;

    std::queue<send_request_t> send_request_queue;

    void main_loop() {
      if (!running)
        return;

      std::vector<uint64_t> id_list;
      id_list.reserve(batch_max_size);
      auto wait_start = std::chrono::steady_clock::now();
      auto batch_time = std::chrono::microseconds(batch_time_us);
      while (true) {
        std::unique_lock<std::mutex> lock(send_request_queue_mtx);
        if (send_request_queue.empty()) {
          send_request_queue_cv.wait_for(lock, batch_time);
        }
        if (!running)
          break;
        uint64_t send_count = 0;
        uint64_t queued_count = send_request_queue.size();

        auto now = std::chrono::steady_clock::now();
        uint8_t cluster_id = 0;
        if ((queued_count >= batch_min_size) ||
            (now - wait_start) >= batch_time) {
          send_count = std::min(batch_max_size, queued_count);
          wait_start = now;
          for (uint64_t i = 0; i < send_count; i++) {
            query_t<data_type> &query = _queue.front();
            id_list.push_back(query.get_query_id());
            batcher.add_query(std::move(query));
            query_queue.pop();
          }
        }
        lock.unlock();

        if (send_count > 0) {
          // std::cout << "hello" << std::endl;
          batcher.serialize();

          ObjectWithStringKey obj;
#ifdef TEST_UDL2
          obj.key = UDL2_PATHNAME_CLUSTER + std::to_string(cluster_id) +
                    "/" + std::to_string(node_id) + "_" +
                    std::to_string(batch_id);
#else
          obj.key = UDL1_PATHNAME "/" + std::to_string(node_id) + "_" +
                    std::to_string(batch_id);
#endif
          // std::cout << "query key " << obj.key << std::endl;
          obj.blob = std::move(*batcher.get_blob());
          capi.trigger_put(obj);
          // std::cout << "Done with obj" << std::endl;

          batch_size[batch_id] = send_count;
          batch_id++; // shouldn't this be in the critical section

          batcher.reset();
          id_list.clear();
        }
      }
      
    }
  public:
    ClientThread(uint64_t batch_min_size, uint64_t batch_max_size,
                 uint64_t batch_time_us, uint64_t dim) {
      this->batch_min_size = batch_min_size;
      this->batch_max_size = batch_max_size;
      this->batch_time_us = batch_time_us;
      this->dim = dim;
    }
#ifdef TEST_UDL2
    void push_query(greedy_query_t<data_type> query) {
      std::scoped_lock<std::mutex> l(query_queue_mtx);
      query_queue.push(std::move(query));
      query_queue_cv.notify_all();      
    }
#else
    void push_query(query_t<data_type> query) {
      std::scoped_lock<std::mutex> l(query_queue_mtx);
      query_queue.push(std::move(query));
      query_queue_cv.notify_all();
    }
#endif
    std::unordered_map<uint64_t, uint64_t> batch_size;
    void signal_stop() {
      std::scoped_lock l(query_queue_mtx);
      running = false;
      query_queue_cv.notify_all();
    }
    void start() {
      running = true;
      real_thread = std::thread(&ClientThread::main_loop, this);
    }
    uint64_t batch_id = 0;

    void join() { real_thread.join(); }
      
      
  };

  class NotificationThread {
    std::thread real_thread;
    bool running = false;
    std::mutex thread_mtx;
    std::condition_variable thread_signal;

    std::queue<std::pair<std::shared_ptr<uint8_t[]>, uint64_t>> to_process;
    BenchmarkClient<data_type> *parent;

    void main_loop() {
      if (!running)
        return;
      while (true) {
        std::unique_lock<std::mutex> lock(thread_mtx);

        if (to_process.empty()) thread_signal.wait(lock);

        if (!running)
          break;

        auto pending = to_process.front();
        to_process.pop();

	lock.unlock();
        if (pending.second == 0) {
          std::cerr << "Error: empty result blob" << std::endl;
          continue;
        }
#ifndef TEST_UDL1
        ANNSearchResultBatchManager manager(pending.first.get(),
                                            pending.second);
        for (const auto &result : manager.get_results()) {
          parent->result_received(result);
          
        }
#else 
        GlobalSearchMessageBatchManager<data_type> manager(
							   pending.first.get(), pending.second, parent->dim);
	for (auto &greedy_query : manager.get_greedy_search_queries()) {
          parent->result_received(greedy_query);
        }
#endif
      }
    }
  public:
    NotificationThread(BenchmarkClient<data_type> *parent) {
      this->parent = parent;
    }

    void push_result(const Blob &result) {
      std::shared_ptr<uint8_t[]> res(new uint8_t[result.size]);
      std::memcpy(res.get(), result.bytes, result.size);

      std::scoped_lock<std::mutex> l(thread_mtx);
      to_process.emplace(res, result.size);
      thread_signal.notify_all();
    }

    void signal_stop() {
      std::scoped_lock<std::mutex> l(thread_mtx);
      running = false;
      thread_signal.notify_all();
    }

    void start() {
      running = true;
      real_thread = std::thread(&NotificationThread::main_loop, this);
    }

    void join() { real_thread.join(); }
    
  };

  uint32_t skip;
  uint32_t dim = 128;

  uint32_t num_result_threads = 1;
  uint32_t next_thread = 0;
  ServiceClientAPI &capi = ServiceClientAPI::get_service_client();
  uint32_t my_id = capi.get_my_id();
  ClientThread *client_thread;
  std::deque<NotificationThread> notification_threads;

  std::unordered_map<uint32_t, std::chrono::steady_clock::time_point> query_send_time;
  std::unordered_map<uint32_t, std::chrono::steady_clock::time_point>
      query_result_time;

  std::mutex query_id_mtx;
  uint32_t query_count = 0;
  uint32_t next_query_id() {
    std::unique_lock<std::mutex> lock(query_id_mtx);
    uint32_t id = query_count;
    query_count++;
    return id;
  }

  std::atomic<uint64_t> result_count = 0;
  std::shared_mutex result_mutex;


#ifdef TEST_UDL1
  std::unordered_map<uint32_t, std::shared_ptr<GreedySearchQuery<data_type>>>
      results;
  void result_received(std::shared_ptr<GreedySearchQuery<data_type>> result) {
    uint32_t query_id = result->get_query_id();
    std::unique_lock<std::shared_mutex> lock(result_mutex);
    results[query_id] = result;
    result_count++;
    query_result_time[query_id] = std::chrono::steady_clock::now();
  }
#else
  std::unordered_map<uint32_t, ANNSearchResult>
      results;  
  void result_received(ANNSearchResult result) {
    uint32_t query_id = result.get_query_id();
    std::unique_lock<std::shared_mutex> lock(result_mutex);
    results[query_id] = result;
    result_count++;
    query_result_time[query_id] = std::chrono::steady_clock::now();
  }
#endif  
  
public:
  BenchmarkClient(uint32_t skip = 0) {
    this->skip = skip;
  }
  ~BenchmarkClient() {
    client_thread->signal_stop();
    client_thread->join();

    for(auto &t : notification_threads){
        t.signal_stop();
    }
    
    for(auto &t : notification_threads){
        t.join();
    }

    std::cout << "Finished joining the threads" << std::endl;

    // print e2e performance statistics (discarding the first 'skip' queries)
    std::vector<uint32_t> queries;
    if (query_send_time.size() == 0) {
      std::cout << "No queries" << std::endl;
      return;
      
    }
    for (const auto& [query_id, send_time] : query_send_time){
        if(query_result_time.count(query_id) == 0) continue;
        queries.push_back(query_id);
    }
    std::sort(queries.begin(),queries.end());

    std::cout << "Done sorting queries" << std::endl;
    uint64_t num_queries = queries.size();
    std::vector<double> latencies;
    std::chrono::steady_clock::time_point first = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last;
    double sum = 0.0;
    for(uint64_t i=skip; i<num_queries; i++){
        auto query_id = queries[i];
        auto& sent = query_send_time[query_id];
        auto& received = query_result_time[query_id];
        std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(received - sent);

        double lat = static_cast<double>(elapsed.count()) / 1000.0;
        latencies.push_back(lat);
        sum += lat;

        first = std::min(first,sent);
        last = std::max(last,received);
    }

    std::chrono::microseconds total_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(last - first);
    double total_time = static_cast<double>(total_elapsed.count()) / 1000000.0;
    double throughput = (num_queries-skip) / total_time;
    std::sort(latencies.begin(),latencies.end());
    double avg = sum / latencies.size();
    double min = latencies.front();
    double max = latencies.back();
    auto median = latencies[latencies.size()/2];
    auto p95 = latencies[(uint64_t)(latencies.size()*0.95)];

    std::cout << "Throughput: " << throughput << " queries/s" << " (" << num_queries-skip << " queries in " << total_time << " seconds)" << std::endl;
    std::cout << "E2E latency:" << std::endl;
    std::cout << "  avg: " << avg << std::endl;
    std::cout << "  median: " << median << std::endl;
    std::cout << "  min: " << min << std::endl;
    std::cout << "  max: " << max << std::endl;
    std::cout << "  p95: " << p95 << std::endl;

    // print batching statistics (discarding the first 10%)
    std::cout << "batching statistics:" << std::endl;
    std::vector<uint64_t> values;
    values.reserve(client_thread->batch_size.size());
    uint64_t start = (uint64_t)(client_thread->batch_id * 0.1);
    sum = 0.0;
    for(const auto& [batch_id, sz] : client_thread->batch_size){
        if(batch_id < start) continue;
        values.push_back(sz);
        sum += sz;
    }

    avg = sum / values.size();
    std::sort(values.begin(),values.end());
    min = values.front();
    max = values.back();
    median = values[values.size()/2];
    p95 = values[(uint64_t)(values.size()*0.95)];

    std::cout << "  avg: " << avg << std::endl;
    std::cout << "  median: " << median << std::endl;
    std::cout << "  min: " << min << std::endl;
    std::cout << "  max: " << max << std::endl;
    std::cout << "  p95: " << p95 << std::endl;    
  }

  void setup(uint64_t batch_min_size, uint64_t batch_max_size,
             uint64_t batch_time_us uint32_t num_result_threads) {
    this->num_result_threads = num_result_threads;

    std::string result_pool_name =
      RESULTS_OBJ_POOL_PREFIX "/" + std::to_string(my_id);
    std::cout << " creatig object pool to recieve results " << result_pool_name
    << std::endl;

    auto res =
        capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(
									    result_pool_name, RESULTS_OBJ_POOL_SUBGROUP_INDEX, HASH, {});

    for (auto &reply_future : res.get())
      reply_future.second.get();

    std::cout << " registering notification handler " << result_pool_name
    << std::endl;
    bool ret = capi.register_notification_handler(
        [&](const Blob &result) {
          notification_threads[next_thread].push_result(result);
          next_thread = (next_thread + 1) % this->num_result_threads;
          return true;
        },
						  result_pool_name);
    auto shards = capi.get_subgroup_members<VolatileCascadeStoreWithStringKey>(
									       RESULTS_OBJ_POOL_SUBGROUP_INDEX);

    std::cout << " pre-establishing connections in with all nodes in "
    << shards.size() << " shards  " << std::endl;

    ObjectWithStringKey obj;
    obj.key = "establish_connection";
    std::string val = "establish";
    obj.blob = Blob(reinterpret_cast<const uint8_t *>(val.c_str()), val.size());
    int i = 0;
    for (auto &shard : shards) {
      for (int j = 0; j < shard.size(); j++) {
        auto res = capi.template put<VolatileCascadeStoreWithStringKey>(
									obj, RESULTS_OBJ_POOL_SUBGROUP_INDEX, i, true);
        for (auto &reply : res.get()) {
	  reply.second.get();
        }
      }
      i++;
    }

    for (int i = 0; i < num_result_threads; i++) {
      notification_threads.emplace_back(this);
    }
    for (auto &t : notification_threads)
      t.start();
    

    client_thread =
      new ClientThread(batch_min_size, batch_max_size, batch_time_us, dim);

    client_thread->start();

    std::cout << "finished setting up" << std::endl;
  }

#ifdef TEST_UDL2
  /**
     precondition is that start node has to be in the cluster.
  */
  uint32_t query(const data_type *query_emb, uint32_t K, uint32_t L, uint8_t cluster_id, uint32_t start_node) {
    uint32_t query_id = next_query_id();
    query_send_time[query_id] = std::chrono::steady_clock::now();
    std::shared_ptr<EmbeddingQuery<data_type>> emb_query =
        std::make_shared<EmbeddingQuery<data_type>>(
            reinterpret_cast<const uint8_t *>(query_emb),
						    sizeof(data_type) * dim, query_id, my_id, K, L, dim);
    // std::cout << emb_query->get_query_id() << std::endl;

    std::vector<uint32_t> candidate_queue = {start_node};
    greedy_query_t<data_type> query(cluster_id, std::move(candidate_queue),
                                    emb_query);
    
    client_thread->push_query(query);
    return query_id;
  }
#else
  uint32_t query(const data_type *query_emb, uint32_t K, uint32_t L) {
    uint32_t query_id = next_query_id();
    query_send_time[query_id] = std::chrono::steady_clock::now();
    query_t<data_type> query(query_id, my_id, query_emb, dim, K, L);
    client_thread->push_query(query);
    // std::cout << "done pushing query" << std::endl;
    return query_id;
  }
#endif
  void wait_results() {
    std::cout << "  received " << result_count << std::endl;
    uint64_t wait_time = 0;
    while(result_count < query_count){
        if(wait_time >= CLIENT_MAX_WAIT_TIME){
            std::cout << "  waited more than " << CLIENT_MAX_WAIT_TIME << " seconds, stopping ..." << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "  received " << result_count << std::endl;
        wait_time += 2;
    }

  }
#ifdef TEST_UDL1
  std::shared_ptr<GreedySearchQuery<data_type>> get_result(uint32_t query_id) {
    std::shared_lock<std::shared_mutex> lock(result_mutex);
    return results[query_id];
  }
#else
  ANNSearchResult get_result(uint32_t query_id) {
    std::shared_lock<std::shared_mutex> lock(result_mutex);
    return results[query_id];
  }
#endif

  void dump_timestamp() {
    TimestampLogger::flush("client" + std::to_string(my_id) + ".dat");
    auto shards = capi.get_subgroup_members<VolatileCascadeStoreWithStringKey>(
									       RESULTS_OBJ_POOL_SUBGROUP_INDEX);
    uint32_t shard_id =0;
    for (auto &shard : shards) {
      ObjectWithStringKey obj;
      obj.key = UDL1_PATHNAME "/flush_logs";
      for (int j = 0; j < shard.size(); j++) {
        auto res = capi.trigger_put<VolatileCascadeStoreWithStringKey>(
								       obj, RESULTS_OBJ_POOL_SUBGROUP_INDEX, shard_id);
      }
      shard_id++;
    }
  }    
};
  
