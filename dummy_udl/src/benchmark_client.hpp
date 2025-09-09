#pragma once

#include <cascade/service_client_api.hpp>
#include <cstdint>
#include <limits>
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
#include <random>

using namespace derecho::cascade;
#define CLIENT_MAX_WAIT_TIME 1000

class BenchmarkClient {
  class ClientThread {
  private:
    std::thread real_thread;
    ServiceClientAPI &capi = ServiceClientAPI::get_service_client();
    uint32_t node_id = capi.get_my_id();
    uint64_t min_batch_size = 1;
    uint64_t max_batch_size = 5;
    uint64_t batch_time_us = 500;
    bool running = false;

    std::mutex send_request_queue_mtx;
    std::condition_variable send_request_queue_cv;

    std::unordered_map<
		       uint8_t, std::unique_ptr<std::vector<std::shared_ptr<send_request_t>>>> send_request_queue;

    void main_loop() {
      if (!running)
        return;

      std::unordered_map<uint8_t, std::chrono::steady_clock::time_point>
      wait_time;
      auto batch_time = std::chrono::microseconds(batch_time_us);
      std::unique_lock<std::mutex> lock(send_request_queue_mtx,
                                        std::defer_lock);
      std::random_device rd;
      std::mt19937 gen(rd()); // seed the generator
      std::uniform_int_distribution<uint64_t> uint64_t_gen(
							   0, std::numeric_limits<uint64_t>::max());
      
      while (true) {
        lock.lock();
        while (send_request_queue.empty()) {
          send_request_queue_cv.wait_for(lock, batch_time);
        }
        if (!running)
          break;
        std::unordered_map<
            uint8_t,
            std::unique_ptr<std::vector<std::shared_ptr<send_request_t>>>>
        request_to_send;
        auto now = std::chrono::steady_clock::now();

        for (auto &[cluster_id, send_requests] : send_request_queue) {
          if (wait_time.count(cluster_id) == 0) {
	    wait_time[cluster_id] = now;
          }
          if (send_requests->size() >= min_batch_size ||
              ((now - wait_time[cluster_id]) >= batch_time)) {
            request_to_send[cluster_id] = std::move(send_requests);
            send_request_queue[cluster_id] = std::make_unique<
							      std::vector<std::shared_ptr<send_request_t>>>();
            send_request_queue[cluster_id]->reserve(max_batch_size);
          }
        }
        lock.unlock();
        for (auto &[cluster_id, send_requests] : request_to_send) {
          size_t total = send_requests->size();
          size_t num_sent = 0;
          while (num_sent < total) {
            size_t batch_size = std::min(total - num_sent, max_batch_size);
            uint64_t batch_id = uint64_t_gen(gen);
            std::vector<std::shared_ptr<send_request_t>> batch_send_requests;
            for (size_t i = num_sent; i < num_sent + batch_size; i++) {
              // std::cout << (int)send_requests->at(i)->cluster_sender_id << " " << (int)send_requests->at(i)->cluster_receiver_id << std::endl;
              batch_send_requests.emplace_back(std::move(send_requests->at(i)));
            }
            std::shared_ptr<Blob> blob =
              send_request_t::get_send_requests_blob(batch_send_requests);
            ObjectWithStringKey obj;
            obj.key = UDL_PATHNAME_CLUSTER + std::to_string(cluster_id) +
                      "_" + std::to_string(batch_id)+"_req";
            obj.blob = std::move(*blob);
            capi.trigger_put<UDL_OBJ_POOL_TYPE>(
						obj, UDL_SUBGROUP_INDEX, static_cast<uint32_t>(cluster_id));
            num_sent += batch_size;
          }
        }
      }
    }
  public:
    ClientThread(uint64_t min_batch_size, uint64_t max_batch_size,
                 uint64_t batch_time_us)
        : min_batch_size(min_batch_size), max_batch_size(max_batch_size),
        batch_time_us(batch_time_us) {}
    void push_send_request(std::shared_ptr<send_request_t> send_req) {
      std::unique_lock l(send_request_queue_mtx);
      if (send_request_queue.count(send_req->cluster_sender_id) == 0) {
        send_request_queue[send_req->cluster_sender_id] =
          std::make_unique<std::vector<std::shared_ptr<send_request_t>>>();
        send_request_queue[send_req->cluster_sender_id]->reserve(max_batch_size);
      }
      send_request_queue[send_req->cluster_sender_id]->push_back(send_req);
      send_request_queue_cv.notify_all();
    }

    void signal_stop() {
      std::scoped_lock l(send_request_queue_mtx);
      running = false;
      send_request_queue_cv.notify_all();
    }
    void start() {
      running = true;
      real_thread = std::thread(&ClientThread::main_loop, this);
    }

    void join() { real_thread.join(); }
  };

  class NotificationThread {
    std::thread real_thread;
    bool running = false;
    std::mutex thread_mtx;
    std::condition_variable thread_signal;

    std::queue<std::pair<std::shared_ptr<uint8_t[]>, uint64_t>> to_process;
    BenchmarkClient *parent;

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
        std::vector<std::shared_ptr<ack_t>> acks =
          ack_t::deserialize_acks(pending.first.get());
        for (const auto &ack : acks) {
	  parent->receive_ack(ack);
        }
      }
    }
  public:
    NotificationThread(BenchmarkClient *parent) {
      this->parent = parent;
    }

    void push_result(const Blob &result) {
      std::scoped_lock<std::mutex> l(thread_mtx);
      std::shared_ptr<uint8_t[]> res(new uint8_t[result.size]);
      std::memcpy(res.get(), result.bytes, result.size);

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

  uint32_t num_result_threads = 1;
  std::atomic<uint32_t> next_thread{0};
  ServiceClientAPI &capi = ServiceClientAPI::get_service_client();
  uint32_t my_id = capi.get_my_id();
  ClientThread *client_thread;
  std::deque<NotificationThread> notification_threads;


  std::atomic<uint64_t> send_request_count = 0;
  std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> send_request_time;
  std::unordered_map<uint64_t, std::chrono::steady_clock::time_point>
      ack_time;

  std::atomic<uint64_t> ack_received_count = 0;
  std::shared_mutex ack_received_mutex;
  std::unordered_map<uint64_t, std::shared_ptr<ack_t>>
      ack_received;  
  void receive_ack(std::shared_ptr<ack_t> ack) {
    std::unique_lock<std::shared_mutex> lock(ack_received_mutex);
    uint64_t message_id = ack->message_id;
    ack_received[message_id] = ack;
    ack_received_count++;
    ack_time[message_id] = std::chrono::steady_clock::now();
  }
  
public:
  ~BenchmarkClient() {
    client_thread->signal_stop();
    client_thread->join();
    delete client_thread;

    for(auto &t : notification_threads){
        t.signal_stop();
    }
    
    for(auto &t : notification_threads){
        t.join();
    }

    std::cout << "Finished joining the threads" << std::endl;

    // print e2e performance statistics (discarding the first 'skip' queries)
    std::vector<uint64_t> send_request_ids;
    if (send_request_time.size() == 0) {
      std::cout << "No send requests" << std::endl;
      return;
    }
    for (const auto& [message_id, send_time] : send_request_time){
        if(ack_time.count(message_id) == 0) continue;
        send_request_ids.push_back(message_id);
    }
    std::sort(send_request_ids.begin(), send_request_ids.end());

    std::cout << "Done sorting requests" << std::endl;
    uint64_t num_send_requests = send_request_ids.size();
    std::vector<double> latencies;
    std::chrono::steady_clock::time_point first = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last;
    double sum = 0.0;
    for(uint64_t i=0; i<num_send_requests; i++){
      auto message_id = send_request_ids[i];
      auto& sent = send_request_time[message_id];
      auto& received = ack_time[message_id];
      std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(received - sent);

      double lat = static_cast<double>(elapsed.count()) / 1000.0;
      latencies.push_back(lat);
      sum += lat;

      first = std::min(first,sent);
      last = std::max(last,received);
    }

    std::chrono::microseconds total_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(last - first);
    double total_time = static_cast<double>(total_elapsed.count()) / 1000000.0;
    double throughput = (num_send_requests) / total_time;
    std::sort(latencies.begin(),latencies.end());
    double avg = sum / latencies.size();
    double min = latencies.front();
    double max = latencies.back();
    auto median = latencies[latencies.size()/2];
    auto p95 = latencies[(uint64_t)(latencies.size()*0.95)];

    std::cout << "Throughput: " << throughput << " requests/s" << " ("
              << num_send_requests << " queries in " << total_time
    << " seconds)" << std::endl;
    std::cout << "E2E latency:" << std::endl;
    std::cout << "  avg: " << avg << std::endl;
    std::cout << "  median: " << median << std::endl;
    std::cout << "  min: " << min << std::endl;
    std::cout << "  max: " << max << std::endl;
    std::cout << "  p95: " << p95 << std::endl;
  }

  void setup(uint64_t min_batch_size, uint64_t max_batch_size,
             uint64_t batch_time_us, uint32_t num_result_threads) {
    this->num_result_threads = num_result_threads;

    auto res_udl = capi.template create_object_pool<UDL_OBJ_POOL_TYPE>(
								       UDL_OBJ_POOL, UDL_SUBGROUP_INDEX, HASH, {});

    std::cout << "finished createing udl object pool " UDL_OBJ_POOL  << std::endl;
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
      new ClientThread(min_batch_size, max_batch_size, batch_time_us);

    client_thread->start();

    std::cout << "finished setting up" << std::endl;
  }
  uint64_t get_message_id() {
    std::random_device rd;
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<uint64_t> uint64_t_gen(
								0, std::numeric_limits<uint64_t>::max());
    return uint64_t_gen(gen);
    
  }

  uint64_t issue_send_request(uint64_t num_bytes, uint8_t cluster_sender_id,
                             uint8_t cluster_receiver_id) {

    std::shared_ptr<send_request_t> req = std::make_shared<send_request_t>();
    req->message_id = get_message_id();
    req->num_bytes = num_bytes;
    req->client_node_id = my_id;
    req->cluster_sender_id = cluster_sender_id;
    req->cluster_receiver_id = cluster_receiver_id;
    send_request_time[req->message_id] = std::chrono::steady_clock::now();
    client_thread->push_send_request(req);
    send_request_count.fetch_add(1);
    return req->message_id;
  }

  void wait_acks() {
    std::cout << "  received " << ack_received_count << std::endl;
    uint64_t wait_time = 0;
    while(ack_received_count < send_request_count){
        if(wait_time >= CLIENT_MAX_WAIT_TIME){
            std::cout << "  waited more than " << CLIENT_MAX_WAIT_TIME << " seconds, stopping ..." << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "  received " << ack_received_count << std::endl;
        wait_time += 2;
    }
  }

  void dump_timestamp() {
    TimestampLogger::flush("client" + std::to_string(my_id) + ".dat");
    auto shards = capi.get_subgroup_members<VolatileCascadeStoreWithStringKey>(
									       UDL_SUBGROUP_INDEX);
    uint32_t shard_id =0;
    for (auto &shard : shards) {
      ObjectWithStringKey obj;
      obj.key = UDL_PATHNAME "/flush_logs";
      for (int j = 0; j < shard.size(); j++) {
        auto res = capi.trigger_put<VolatileCascadeStoreWithStringKey>(
								       obj, UDL_SUBGROUP_INDEX, shard_id);
      }
      shard_id++;
    }
  }    
};
  
