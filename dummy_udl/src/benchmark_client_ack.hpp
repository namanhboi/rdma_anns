#pragma once
// Different from benchmark client, used to send data to a
// dummy udl as external client and receive back the same data as an ack.
// Data sent will be send object with cluster sendr and receiver being the same

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
    bool running = false;

    std::mutex send_object_queue_mtx;
    std::condition_variable send_object_queue_cv;

    std::queue<std::shared_ptr<send_object_t>> send_object_queue;

    void main_loop() {
      if (!running)
        return;
      std::unique_lock<std::mutex> lock(send_object_queue_mtx,
                                        std::defer_lock);
      std::random_device rd;
      while (true) {
        lock.lock();
        while (send_object_queue.empty()) {
          send_object_queue_cv.wait(lock);
        }

        if (!running)
          break;

        std::shared_ptr<send_object_t> send_obj = send_object_queue.front();
        send_object_queue.pop();
        lock.unlock();
        uint64_t message_id = send_obj->message_id;
        uint8_t cluster_id = send_obj->cluster_sender_id;
        std::vector<std::shared_ptr<send_object_t>> send_objects = {send_obj};

        std::shared_ptr<Blob> blob =
          send_object_t::get_send_objects_blob(send_objects);
        ObjectWithStringKey obj;
        obj.key = UDL_PATHNAME_CLUSTER + std::to_string(cluster_id) + "_" +
                  std::to_string(message_id) + "_" + std::to_string(node_id);
        obj.blob = std::move(*blob);
        parent->send_object_time[message_id] = std::chrono::steady_clock::now();
        capi.trigger_put<UDL_OBJ_POOL_TYPE>(obj, UDL_SUBGROUP_INDEX,
                                            static_cast<uint32_t>(cluster_id));
      }
    }
    BenchmarkClient *parent;
  public:
    ClientThread(BenchmarkClient *parent) : parent(parent) {}
    void push_send_object(std::shared_ptr<send_object_t> send_obj) {
      std::unique_lock l(send_object_queue_mtx);
      send_object_queue.push(send_obj);
      send_object_queue_cv.notify_all();
    }

    void signal_stop() {
      std::scoped_lock l(send_object_queue_mtx);
      running = false;
      send_object_queue.push(nullptr);
      send_object_queue_cv.notify_all();
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
      std::unique_lock<std::mutex> lock(thread_mtx, std::defer_lock);
      while (true) {
        lock.lock();

        while (to_process.empty() && (running == true)) {
          thread_signal.wait(lock);
        }
        if (!running)
          break;

        auto pending = to_process.front();
        to_process.pop();

	lock.unlock();
        if (pending.second == 0) {
          std::cerr << "Error: empty result blob" << std::endl;
          continue;
        }
        std::vector<std::shared_ptr<send_object_t>> acks =
          send_object_t::deserialize_send_objects(pending.first.get());
        for (const auto &ack : acks) {
	  parent->receive_ack(ack);
        } 
      }
    }
  public:
    NotificationThread(BenchmarkClient *parent) {
      this->parent = parent;
    }


    // receive ack blob, which contains the exact data sent from udl
    void push_ack_blob(const Blob &ack_blob) {
      std::scoped_lock<std::mutex> l(thread_mtx);
      std::shared_ptr<uint8_t[]> res(new uint8_t[ack_blob.size]);
      std::memcpy(res.get(), ack_blob.bytes, ack_blob.size);

      to_process.emplace(res, ack_blob.size);
      thread_signal.notify_all();
    }

    void signal_stop() {
      std::scoped_lock<std::mutex> l(thread_mtx);
      running = false;
      to_process.emplace(nullptr, 0);
      std::cout << "stop signalled to notify thread" << std::endl; 
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


  std::atomic<uint64_t> send_object_count = 0;
  std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> send_object_time;
  std::unordered_map<uint64_t, std::chrono::steady_clock::time_point>
      ack_time;

  std::atomic<uint64_t> ack_received_count = 0;
  std::shared_mutex ack_received_mutex;
  std::unordered_map<uint64_t, std::shared_ptr<send_object_t>>
      ack_received;  

  void receive_ack(std::shared_ptr<send_object_t> ack) {
    std::unique_lock<std::shared_mutex> lock(ack_received_mutex);
    uint64_t message_id = ack->message_id;
    ack_received[message_id] = ack;
    ack_received_count++;
    ack_time[message_id] = std::chrono::steady_clock::now();
  }
  
public:
  ~BenchmarkClient() {
    std::cout << "destructor called " << std::endl;
    client_thread->signal_stop();
    client_thread->join();
    delete client_thread;

    for (auto &t : notification_threads) {
      std::cout << "trying to call signal stop" << std::endl;
      t.signal_stop();
    }
    
    for(auto &t : notification_threads){
        t.join();
    }

    std::cout << "Finished joining the threads" << std::endl;

    // print e2e performance statistics (discarding the first 'skip' queries)
    std::vector<uint64_t> send_object_ids;
    if (send_object_time.size() == 0) {
      std::cout << "No send objects" << std::endl;
      return;
    }
    for (const auto& [message_id, send_time] : send_object_time){
        if(ack_time.count(message_id) == 0) continue;
        send_object_ids.push_back(message_id);
    }
    std::sort(send_object_ids.begin(), send_object_ids.end());

    std::cout << "Done sorting requests" << std::endl;
    uint64_t num_send_requests = send_object_ids.size();
    std::vector<double> latencies;
    std::chrono::steady_clock::time_point first = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last;
    double sum = 0.0;
    for(uint64_t i=0; i<num_send_requests; i++){
      auto message_id = send_object_ids[i];
      auto& sent = send_object_time[message_id];
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

  void setup(uint32_t num_result_threads) {
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
        [&](const Blob &ack_blob) {
          notification_threads[next_thread].push_ack_blob(ack_blob);
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
      new ClientThread(this);

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

  uint64_t issue_send_object(uint64_t num_bytes, uint8_t cluster_sender_id) {
    std::shared_ptr<send_object_t> obj = std::make_shared<send_object_t>();
    obj->message_id = get_message_id();
    obj->num_bytes = num_bytes;
    std::shared_ptr<uint8_t[]> tmp(new uint8_t[num_bytes]);
    obj->bytes = std::move(tmp);
    obj->client_node_id = my_id;
    obj->cluster_sender_id = cluster_sender_id;
    obj->cluster_receiver_id = cluster_sender_id;
    client_thread->push_send_object(obj);
    send_object_count.fetch_add(1);
    return obj->message_id;
  }

  void wait_acks() {
    std::cout << "  received " << ack_received_count << std::endl;
    uint64_t wait_time = 0;
    while(ack_received_count < send_object_count){
        if(wait_time >= CLIENT_MAX_WAIT_TIME){
            std::cout << "  waited more than " << CLIENT_MAX_WAIT_TIME << " seconds, stopping ..." << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "  received " << ack_received_count << "sent "
        << send_object_count << std::endl;
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
  
