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

#define MY_UUID "15dbf1f9-e43d-4496-bd36-f3e43b505bf3"
#define MY_DESC                                                              \
    "udl that just sends stuff to other udls to test baseline batch sending "  \
    "latency"
          
    std::string get_uuid();

    std::string get_description();
    class DummyOCDPO : public DefaultOffCriticalDataPathObserver {
      class AckThread {
        uint64_t thread_id;
        std::thread real_thread;
        std::atomic<bool> running = false;
        DummyOCDPO *parent;
        moodycamel::BlockingConcurrentQueue<std::shared_ptr<send_object_t>>
            concurrent_send_obj_queue;
        
        void main_loop(DefaultCascadeContextType *typed_ctxt) {
          while (running) {
            std::shared_ptr<send_object_t> obj;
            concurrent_send_obj_queue.wait_dequeue(obj);
            if (!running)
              break;
            if (obj == nullptr)
              break;
            if (obj->cluster_receiver_id != parent->cluster_id) {
              throw std::invalid_argument(
					  "object has cluster receiver id " +
                  std::to_string(static_cast<int>(obj->cluster_sender_id)) +
                  ", but this is cluster " +
                  std::to_string(static_cast<int>(parent->cluster_id)));
            }
            std::shared_ptr<ack_t> ack = std::make_shared<ack_t>();
            ack->message_id = obj->message_id;
            ack->num_bytes = obj->num_bytes;
            ack->client_node_id = obj->client_node_id;
            parent->batch_thread->enqueue_ack_to_batch(ack);
          }
        }
      public:
        AckThread(uint64_t thread_id, DummyOCDPO *parent)
        : thread_id(thread_id), parent(parent) {}
        void push_send_objects(std::vector<std::shared_ptr<send_object_t>> send_objects) {
          concurrent_send_obj_queue.enqueue_bulk(
              std::make_move_iterator(send_objects.begin()),
						 send_objects.size());
	}
        void start(DefaultCascadeContextType *typed_ctxt) {
          running = true;
          real_thread =
            std::thread(&DummyOCDPO::AckThread::main_loop, this, typed_ctxt);
        }
        
        void join() {
          if (real_thread.joinable()) {
	    real_thread.join();
          }
        }
        void signal_stop() {
          running = false;
          concurrent_send_obj_queue.enqueue(nullptr);
	}          
      };

      class SendingThread {
        uint64_t thread_id;
        std::thread real_thread;
        std::atomic<bool> running = false;
        DummyOCDPO *parent;
        moodycamel::BlockingConcurrentQueue<std::shared_ptr<send_request_t>> concurrent_send_req_queue;

        void main_loop(DefaultCascadeContextType *typed_ctxt) {
          while (running) {
            std::shared_ptr<send_request_t> req;
            concurrent_send_req_queue.wait_dequeue(req);
            if (!running)
              break;
            if (req == nullptr) break;
            if (req->cluster_sender_id != parent->cluster_id) {
              throw std::invalid_argument(
                  "request has cluster id " +
                  std::to_string(static_cast<int>(req->cluster_sender_id)) +
                  ", but this is cluster " +
                  std::to_string(static_cast<int>(parent->cluster_id)));
            }
            if (req->cluster_receiver_id == parent->cluster_id) {
              throw std::invalid_argument(
                  "request has cluster id " +
                  std::to_string(static_cast<int>(req->cluster_receiver_id)) +
                  ", but this is cluster " +
                  std::to_string(static_cast<int>(parent->cluster_id)));
            }
            std::shared_ptr<send_object_t> obj =
              std::make_shared<send_object_t>();
            obj->message_id = req->message_id;
            obj->cluster_receiver_id = req->cluster_receiver_id;
            obj->cluster_sender_id = req->cluster_sender_id;
            obj->num_bytes = req->num_bytes;
            obj->client_node_id = req->client_node_id;
            std::shared_ptr<uint8_t[]> tmp_bytes(new uint8_t[req->num_bytes]);
            obj->bytes = tmp_bytes;
            parent->batch_thread->enqueue_obj_to_batch(obj);
          }
        }
      public:
        SendingThread(uint64_t thread_id, DummyOCDPO *parent)
        : thread_id(thread_id), parent(parent) {}

        void push_send_requests(std::vector<std::shared_ptr<send_request_t>> send_requests) {
          concurrent_send_req_queue.enqueue_bulk(
              std::make_move_iterator(send_requests.begin()),
						 send_requests.size());
        }
        void start(DefaultCascadeContextType *typed_ctxt) {
          running = true;
          real_thread = std::thread(&DummyOCDPO::SendingThread::main_loop, this,
                                    typed_ctxt);
          
        }
        void join() {
          if (real_thread.joinable()) real_thread.join();
        }
        void signal_stop() {
          running = false;
          concurrent_send_req_queue.enqueue(nullptr);
	}
      };

      class BatchingThread {
        DummyOCDPO *parent;
        uint64_t thread_id;
        std::thread real_thread;
        std::unordered_map<
            uint8_t,
            std::unique_ptr<std::vector<std::shared_ptr<send_object_t>>>>
            obj_queue;

        std::unordered_map<
            uint64_t,
            std::unique_ptr<std::vector<std::shared_ptr<ack_t>>>>
            ack_queue;
        
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

          std::unordered_map<uint64_t, std::chrono::steady_clock::time_point>
          wait_time_ack;

	  std::random_device rd; // obtain a random number from hardware
	  std::mt19937 gen(rd()); // seed the generator
          std::uniform_int_distribution<uint64_t> uint64_t_gen(
							       0, std::numeric_limits<uint64_t>::max());
          while (running) {
            queue_lock.lock();
            while (is_empty(obj_queue) && is_empty(ack_queue)) {
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

            std::unordered_map<
                uint64_t,
                std::unique_ptr<std::vector<std::shared_ptr<ack_t>>>>
            ack_to_send;
            now = std::chrono::steady_clock::now();
            for (auto &[client_node_id, acks] : ack_queue) {
              if (wait_time_ack.count(client_node_id) == 0) {
                wait_time_ack[client_node_id] = now;
              }
              if (acks->size() >= parent->min_batch_size ||
                  ((now - wait_time_ack[client_node_id]) >= batch_time)) {
                ack_to_send[client_node_id] = std::move(acks);
                ack_queue[client_node_id] = std::make_unique<
							  std::vector<std::shared_ptr<ack_t>>>();
                ack_queue[client_node_id]->reserve(parent->max_batch_size);
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
            for (auto &[client_node_id, acks] : ack_to_send) {
              uint64_t num_sent = 0;
              uint64_t total = acks->size();
              while (num_sent < total) {
                uint32_t left = total - num_sent;
                uint32_t batch_size = std::min(parent->max_batch_size, left);
                uint64_t batch_id = uint64_t_gen(gen);
		std::vector<std::shared_ptr<ack_t>> batch_acks;
                for (auto i = num_sent; i < num_sent + batch_size; i++) {
                  std::shared_ptr<ack_t> ack = acks->at(i);
                  batch_acks.push_back(ack);
                }
                std::shared_ptr<Blob> blob =
                  ack_t::get_acks_blob(std::move(batch_acks));
                std::string client_id_pool_path =
                    RESULTS_OBJ_POOL_PREFIX "/" +
                    std::to_string(client_node_id);
                typed_ctxt->get_service_client_ref().notify(
                    *blob, client_id_pool_path,
							    client_node_id);
                num_sent += batch_size;
              }
            }            
          }
        }
      public:
        BatchingThread(uint64_t thread_id, DummyOCDPO *parent)
        : thread_id(thread_id), parent(parent) {}

        void enqueue_ack_to_batch(std::shared_ptr<ack_t> ack) {
          std::unique_lock l(queue_mutex);
          if (ack_queue.count(ack->client_node_id) == 0) {
            ack_queue[ack->client_node_id] =
              std::make_unique<std::vector<std::shared_ptr<ack_t>>>();
          }
          ack_queue[ack->client_node_id]->push_back(ack);
          queue_cv.notify_all();
        }

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
          this->real_thread = std::thread(
					  &DummyOCDPO::BatchingThread::main_loop, this, typed_ctxt);
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
      uint32_t num_ack_threads = 1;
      std::vector<std::unique_ptr<AckThread>> ack_threads;      
      std::unique_ptr<BatchingThread> batching_thread;
      uint64_t my_id, shard_id;
      std::atomic<int> current_send_thread = 0;
      std::atomic<int> current_ack_thread = 0;
      void initialize_cluster_id(uint8_t cluster_id) {
	this->cluster_id = cluster_id;
      }
    public:
      void ocdpo_handler(const node_id_t sender,
                         const std::string &object_pool_pathname,
                         const std::string &key_string,
                         const ObjectWithStringKey &object,
                         const emit_func_t &emit,
                         DefaultCascadeContextType *typed_ctxt,
                         uint32_t worker_id) override {
	if (key_string == "flush_logs") {
          std::string log_file_name = "node" + std::to_string(my_id) + "_udls_timestamp.dat";
          TimestampLogger::flush(log_file_name);
          std::cout << "Flushed logs to " << log_file_name <<"."<< std::endl;
          return;
	}
        auto [key_cluster_id, key_batch_id] =
          parse_cluster_and_batch_id(key_string);
        TimestampLogger::log(LOG_DUMMY_HANDLER_START,
                             std::numeric_limits<uint64_t>::max(), key_batch_id,
                             object.blob.size);
        std::call_once(initialized_cluster_id,
                       &DummyOCDPO::initialize_cluster_id, this, key_cluster_id);
        if (key_string.find("obj") != std::string::npos) {
          std::vector<std::shared_ptr<send_object_t>> send_objects =
            send_object_t::deserialize_send_objects(object.blob.bytes);
          ack_threads[current_ack_thread]->push_send_objects(send_objects);
          current_ack_thread = (current_ack_thread + 1) % num_ack_threads;
        } else if (key_string.find("req") != std::string::npos) {
          std::vector<std::shared_ptr<send_request_t>> send_requests =
            send_request_t::deserialize_send_requests(object.blob.bytes);
          send_threads[current_send_thread]->push_send_requests(send_requests);
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
	  ocdpo_ptr = std::make_shared<DummyOCDPO>();
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
	  if (config.contains("num_ack_threads")){
            this->num_ack_threads = config["num_ack_threads"].get<uint32_t>();
	  }          
	  std::cout << "num_send_threads " <<  num_send_threads << std::endl;

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
    
	for (uint32_t thread_id = 0; thread_id < this->num_send_threads;
             thread_id++) {
	  send_threads.emplace_back(new SendingThread(thread_id, this));
	}
	for (auto &send_thread : send_threads) {
	  send_thread->start(typed_ctxt);
        }
	for (uint32_t thread_id = 0; thread_id < this->num_ack_threads;
             thread_id++) {
	  ack_threads.emplace_back(new AckThread(thread_id, this));
	}
	for (auto &ack_thread : ack_threads) {
	  ack_thread->start(typed_ctxt);
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
	for (auto &ack_thread : ack_threads) {
	  if (ack_thread) {
            ack_thread->signal_stop();
            ack_thread->join();
	  }
	}        
	if (batch_thread) {
	  batch_thread->signal_stop();
	  batch_thread->join();
	}
      }
      
    };
    std::shared_ptr<OffCriticalDataPathObserver> DummyOCDPO::ocdpo_ptr;
    void initialize(ICascadeContext *ctxt);

    std::shared_ptr<OffCriticalDataPathObserver>
    get_observer(ICascadeContext *ctxt, const nlohmann::json &config);
  } // namespace cascade
} // namespace derecho

