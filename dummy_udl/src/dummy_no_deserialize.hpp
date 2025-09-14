#include <chrono>
#include <random>
#include <cascade/cascade_interface.hpp>
#include <cascade/service_types.hpp>
#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <cascade/object.hpp>
#include <cascade/service_types.hpp>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include "../../src/blockingconcurrentqueue.h"
#include "udl_path_and_index.hpp"
#include "serialize_utils.hpp"

namespace derecho {
  namespace cascade {

#define MY_UUID "8fe78d79-355e-479f-8909-e3b88a491831"
#define MY_DESC                                                              \
    "udl that just sends stuff to other udls to test baseline to test sending latency with no serialization. Handler is basically a no op"
          
    std::string get_uuid();

    std::string get_description();
    class DummyNoDeserializeOCDPO : public DefaultOffCriticalDataPathObserver {
      class SendingThread {
        uint64_t thread_id;
        std::thread real_thread;
        std::atomic<bool> running = false;
        DummyNoDeserializeOCDPO *parent;
        moodycamel::BlockingConcurrentQueue<std::shared_ptr<send_query_t>> concurrent_send_query_queue;

        void main_loop(DefaultCascadeContextType *typed_ctxt) {
	  std::random_device rd;
	  std::mt19937 gen(rd()); // seed the generator
          std::uniform_int_distribution<uint64_t> uint64_t_gen(
							       0, std::numeric_limits<uint64_t>::max());
          while (running) {
            std::shared_ptr<send_query_t> query;
            concurrent_send_query_queue.wait_dequeue(query);
            if (!running)
              break;
            if (query == nullptr) break;
            if (query->cluster_sender_id != parent->cluster_id) {
              throw std::invalid_argument(
                  "request has cluster id " +
                  std::to_string(static_cast<int>(query->cluster_sender_id)) +
                  ", but this is cluster " +
                  std::to_string(static_cast<int>(parent->cluster_id)));
            }
            int num_msg = query->num_msg;
            for (auto i = 0; i < num_msg; i++) {
              for (uint8_t cluster_receiver_id = 0;
                   cluster_receiver_id < query->num_clusters;
                   cluster_receiver_id++) {
                if (cluster_receiver_id == parent->cluster_id) continue;
		std::shared_ptr<send_object_t> obj =
		  std::make_shared<send_object_t>();
		obj->message_id = uint64_t_gen(gen);
		obj->cluster_receiver_id = cluster_receiver_id;
		obj->cluster_sender_id = parent->cluster_id;
                obj->num_bytes = query->num_bytes;
		obj->client_node_id = query->client_node_id;
		std::shared_ptr<uint8_t[]> tmp_bytes(new uint8_t[query->num_bytes]);
		obj->bytes = tmp_bytes;
                parent->batch_thread->enqueue_obj_to_batch(obj);
              }
            }

          }
        }
      public:
        SendingThread(uint64_t thread_id, DummyNoDeserializeOCDPO *parent)
        : thread_id(thread_id), parent(parent) {}

        void push_send_query(std::shared_ptr<send_query_t> send_query) {
          concurrent_send_query_queue.enqueue(std::move(send_query));
        }
        void start(DefaultCascadeContextType *typed_ctxt) {
          running = true;
          real_thread =
              std::thread(&DummyNoDeserializeOCDPO::SendingThread::main_loop,
                          this, typed_ctxt);
        }
        void join() {
          if (real_thread.joinable()) real_thread.join();
        }
        void signal_stop() {
          running = false;
          concurrent_send_query_queue.enqueue(nullptr);
	}
      };

      class BatchingThread {
        DummyNoDeserializeOCDPO *parent;
        uint64_t thread_id;
        std::thread real_thread;
        std::unordered_map<
            uint8_t,
            std::unique_ptr<std::vector<std::shared_ptr<send_object_t>>>>
            obj_queue;
        
        std::mutex queue_mutex;
        std::condition_variable_any queue_cv;

	std::atomic<bool> running = false;
        
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

        void main_loop(DefaultCascadeContextType *typed_ctxt) {
          std::unique_lock queue_lock(queue_mutex, std::defer_lock);
          auto batch_time = std::chrono::microseconds(parent->batch_time_us);
          std::unordered_map<uint8_t, std::chrono::steady_clock::time_point>
          wait_time_obj;

	  std::random_device rd; // obtain a random number from hardware
	  std::mt19937 gen(rd()); // seed the generator
          std::uniform_int_distribution<uint64_t> uint64_t_gen(
							       0, std::numeric_limits<uint64_t>::max());
          while (running) {
            queue_lock.lock();
            while (is_empty(obj_queue)) {
              queue_cv.wait_for(queue_lock, batch_time);
            }
            if (!running) {
	      break;
            }
            std::unordered_map<
			       uint8_t,
			       std::unique_ptr<std::vector<std::shared_ptr<send_object_t>>>>
            obj_to_send;
            auto now = std::chrono::steady_clock::now();
            for (auto &[cluster_id, objects] : obj_queue) {
              if (wait_time_obj.count(cluster_id) == 0) {
                wait_time_obj[cluster_id] = now;
              }
              if (objects->size() >= parent->min_batch_size ||
                  ((now - wait_time_obj[cluster_id]) >= batch_time)) {
                obj_to_send[cluster_id] = std::move(objects);
                obj_queue[cluster_id] = std::make_unique<
							  std::vector<std::shared_ptr<send_object_t>>>();
                obj_queue[cluster_id]->reserve(parent->max_batch_size);
              }
            }

            queue_lock.unlock();
            for (auto &[cluster_id, objects] : obj_to_send) {
              uint64_t num_sent = 0;
              uint64_t total = objects->size();
              while (num_sent < total) {
                uint32_t left = total - num_sent;
                uint32_t batch_size = std::min(parent->max_batch_size, left);
                uint64_t batch_id = uint64_t_gen(gen);
		std::vector<std::shared_ptr<send_object_t>> batch_send_objects;
                for (auto i = num_sent; i < num_sent + batch_size; i++) {
                  std::shared_ptr<send_object_t> obj = objects->at(i);
                  batch_send_objects.push_back(obj);
                }
                std::shared_ptr<Blob> blob =
                    send_object_t::get_send_objects_blob(
							 std::move(batch_send_objects));
                ObjectWithStringKey obj;
		obj.blob = std::move(*blob);
		obj.previous_version = INVALID_VERSION;
		obj.previous_version_by_key = INVALID_VERSION;
                obj.key = UDL_PATHNAME_CLUSTER + std::to_string(cluster_id) +
                          "_" + std::to_string(batch_id) + "_obj";
                TimestampLogger::log(LOG_DUMMY_BATCH_SEND_START,
                                     std::numeric_limits<uint64_t>::max(),
                                     batch_id, 0ull);
                typed_ctxt->get_service_client_ref()
                    .put_and_forget<UDL_OBJ_POOL_TYPE>(
                        obj, UDL_SUBGROUP_INDEX,
						       static_cast<uint32_t>(cluster_id), true);
                TimestampLogger::log(LOG_DUMMY_BATCH_SEND_END,
                                     std::numeric_limits<uint64_t>::max(),
                                     batch_id, 0ull);                
                num_sent += batch_size;
              }
            }
	    std::this_thread::sleep_for(std::chrono::microseconds(parent->sleep_interval_us));
          }
        }
      public:
        BatchingThread(uint64_t thread_id, DummyNoDeserializeOCDPO *parent)
        : thread_id(thread_id), parent(parent) {}

        void enqueue_obj_to_batch(std::shared_ptr<send_object_t> obj) {
          std::unique_lock l(queue_mutex);
          if (obj_queue.count(obj->cluster_receiver_id) == 0) {
            obj_queue[obj->cluster_receiver_id] =
              std::make_unique<std::vector<std::shared_ptr<send_object_t>>>();
          }
          obj_queue[obj->cluster_receiver_id]->push_back(obj);
          queue_cv.notify_all();
        }
        void start(DefaultCascadeContextType *typed_ctxt) {
          running = true;
          this->real_thread =
              std::thread(&DummyNoDeserializeOCDPO::BatchingThread::main_loop,
                          this, typed_ctxt);
        }
        void join() {
          if (real_thread.joinable()) {
	    real_thread.join();
          }
        }
        void signal_stop() {
          std::unique_lock l(queue_mutex);
          // has to do this to escape while loop
          if (obj_queue.count(static_cast<uint8_t>(0)) == 0) {
            obj_queue[static_cast<uint8_t>(0)] =
              std::make_unique<std::vector<std::shared_ptr<send_object_t>>>();
          }
          obj_queue[static_cast<uint8_t>(0)]->emplace_back(nullptr);
          running = false;
          queue_cv.notify_all();
        }
      };
      static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

      std::unique_ptr<BatchingThread> batch_thread;
      uint8_t cluster_id;
      std::once_flag initialized_cluster_id;

      uint32_t batch_time_us = 1000, min_batch_size = 1, max_batch_size = 100;
      
      std::pair<uint8_t, uint64_t>
      parse_cluster_and_batch_id(const std::string &key_string) {
	std::string cluster_prefix = "cluster";
	uint8_t cluster_id = static_cast<uint8_t>(
						  std::stoll(key_string.substr(cluster_prefix.size())));
	std::string cluster_id_str = std::to_string(cluster_id);

	size_t pos = key_string.find("_");
	uint64_t batch_id;
	if (pos != std::string::npos) {
	  batch_id = std::stoull(key_string.substr(pos + 1));
	} else {
	  batch_id = std::numeric_limits<uint64_t>::max();
	}
	return std::make_pair(cluster_id, batch_id); 
      }

      uint32_t num_send_threads = 1;
      std::vector<std::unique_ptr<SendingThread>> send_threads;
      std::unique_ptr<BatchingThread> batching_thread;
      uint64_t my_id, shard_id;
      std::atomic<int> current_send_thread = 0;
      std::atomic<int> current_ack_thread = 0;
      void initialize_cluster_id(uint8_t cluster_id) {
	this->cluster_id = cluster_id;
      }

      std::atomic<int> send_objects_received = 0;
      std::atomic<int> send_objects_milestone = 0;
      std::atomic<int> send_objects_milestone_space = 20'000;

      uint64_t sleep_interval_us = 0;
    public:
      void ocdpo_handler(const node_id_t sender,
                         const std::string &object_pool_pathname,
                         const std::string &key_string,
                         const ObjectWithStringKey &object,
                         const emit_func_t &emit,
                         DefaultCascadeContextType *typed_ctxt,
                         uint32_t worker_id) override {
        
        auto [key_cluster_id, key_batch_id] =
          parse_cluster_and_batch_id(key_string);
        TimestampLogger::log(LOG_DUMMY_HANDLER_START,
                             std::numeric_limits<uint64_t>::max(), key_batch_id,
                             object.blob.size);
        std::call_once(initialized_cluster_id,
                       &DummyNoDeserializeOCDPO::initialize_cluster_id, this,
                       key_cluster_id);
        if (key_string.find("obj") != std::string::npos) {
          // no_op this is supposed to test no serialization at all
          uint64_t num_objs =
            *reinterpret_cast<const uint64_t *>(object.blob.bytes);
          send_objects_received += (num_objs);
          int calculated_milestone =
              send_objects_milestone_space *
              (send_objects_received / send_objects_milestone_space);
          if (calculated_milestone != send_objects_milestone) {
            send_objects_milestone = calculated_milestone;
            std::string log_file_name = "node" + std::to_string(my_id) + "_udls_timestamp.dat";
            TimestampLogger::flush(log_file_name);
            std::cout << "Flushed logs to " << log_file_name << "."
            << std::endl;
          }
        } else if (key_string.find("query") != std::string::npos) {
          std::shared_ptr<send_query_t> send_query =
            send_query_t::deserialize(object.blob.bytes);
          // send_objects_milestone_space = send_query->num_msg * (send_query->num_clusters - 1);
          send_threads[current_send_thread]->push_send_query(std::move(send_query));
          current_send_thread = (current_send_thread + 1) % num_send_threads;
        } else {
          throw std::invalid_argument("weird keystring value in udl: " +
                                      key_string);
        }
        TimestampLogger::log(LOG_DUMMY_HANDLER_END,
                             std::numeric_limits<uint64_t>::max(), key_batch_id,
                             object.blob.size);
      }
      static void initialize() {
	if (!ocdpo_ptr) {
	  ocdpo_ptr = std::make_shared<DummyNoDeserializeOCDPO>();
	}
      }
      
      static auto get() { return ocdpo_ptr; }
      void set_config(DefaultCascadeContextType *typed_ctxt,
                      const nlohmann::json &config) {
	this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
	std::cout << "dummy udl id is  " << my_id << std::endl;
	std::cout << "members "
	<< typed_ctxt->get_service_client_ref().get_members()
	<< std::endl;

	shard_id = typed_ctxt->get_service_client_ref()
                   .get_my_shard<VolatileCascadeStoreWithStringKey>(
								    UDL_SUBGROUP_INDEX);
	std::cout << "my shard " << shard_id << std::endl;
	try {
	  if (config.contains("num_send_threads")){
            this->num_send_threads = config["num_send_threads"].get<uint32_t>();
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

	  if (config.contains("sleep_interval_us")) {
            this->sleep_interval_us =
              config["sleep_interval_us"].get<uint64_t>();
          }
          if (config.contains("send_objects_milestone_space")) {
            this->send_objects_milestone_space = config["send_objects_milestone_space"].get<uint64_t>();
          }

	} catch (const std::exception &e) {
	  std::cout << "error while parsing config" << std::endl;
	}
    
	for (uint32_t thread_id = 0; thread_id < this->num_send_threads;
             thread_id++) {
	  send_threads.emplace_back(new SendingThread(thread_id, this));
	}
	for (auto &send_thread : send_threads) {
	  send_thread->start(typed_ctxt);
        }
	this->batch_thread = std::make_unique<BatchingThread>(this->my_id, this);
	this->batch_thread->start(typed_ctxt);
      }

      void shutdown() {
	for (auto &send_thread : send_threads) {
	  if (send_thread) {
            send_thread->signal_stop();
            send_thread->join();
	  }
        }
	if (batch_thread) {
	  batch_thread->signal_stop();
	  batch_thread->join();
	}
      }
      
    };
    std::shared_ptr<OffCriticalDataPathObserver> DummyNoDeserializeOCDPO::ocdpo_ptr;
    void initialize(ICascadeContext *ctxt);

    std::shared_ptr<OffCriticalDataPathObserver>
    get_observer(ICascadeContext *ctxt, const nlohmann::json &config);
  } // namespace cascade
} // namespace derecho

