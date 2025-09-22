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
#include "udl_path_and_index.hpp"
#include "serialize_utils.hpp"

namespace derecho {
  namespace cascade {

#define MY_UUID "d62deb7a-c30b-41bd-8761-92b6ad42e94e"
#define MY_DESC                                                              \
    "test roundtrip send latency. client send an object, udl send back the same object as ack"
    std::string get_uuid();

    std::string get_description();
    class DummyAckOCDPO : public DefaultOffCriticalDataPathObserver {
      class AckThread {
        DummyAckOCDPO *parent;
        uint64_t thread_id;
        std::thread real_thread;
        std::queue<std::shared_ptr<send_object_t>> obj_queue;
        
        std::mutex queue_mutex;
        std::condition_variable_any queue_cv;

	std::atomic<bool> running = false;

        void main_loop(DefaultCascadeContextType *typed_ctxt) {
          std::unique_lock queue_lock(queue_mutex, std::defer_lock);
          while (running) {
            queue_lock.lock();
            while (obj_queue.empty()) {
              queue_cv.wait(queue_lock);
            }
            if (!running) {
	      break;
            }
            std::shared_ptr<send_object_t> send_obj = obj_queue.front();
            obj_queue.pop();
            queue_lock.unlock();

            std::vector<std::shared_ptr<send_object_t>> send_objects = {
              send_obj};

            uint64_t message_id = send_obj->message_id;
            uint64_t client_node_id = send_obj->client_node_id;
            
            std::string client_id_pool_path =
              RESULTS_OBJ_POOL_PREFIX "/" + std::to_string(client_node_id);

            std::shared_ptr<Blob> blob =
              send_object_t::get_send_objects_blob(send_objects);
            // std::cout << "blob size " << blob->size;
            TimestampLogger::log(LOG_DUMMY_BATCH_SEND_START,
                                 client_node_id,
                                 message_id, 0ull);
            typed_ctxt->get_service_client_ref().notify(
							*blob, client_id_pool_path, client_node_id);
            TimestampLogger::log(LOG_DUMMY_BATCH_SEND_END, client_node_id,
                                 message_id, 0ull);
	  }
        }
      public:
        AckThread(uint64_t thread_id, DummyAckOCDPO *parent)
        : thread_id(thread_id), parent(parent) {}

        void enqueue_obj(std::shared_ptr<send_object_t> obj) {
          std::unique_lock l(queue_mutex);
          obj_queue.push(obj);
          queue_cv.notify_all();
        }
        void start(DefaultCascadeContextType *typed_ctxt) {
          running = true;
          this->real_thread = std::thread(&DummyAckOCDPO::AckThread::main_loop,
                                          this, typed_ctxt);
        }
        void join() {
          if (real_thread.joinable()) {
	    real_thread.join();
          }
        }
        void signal_stop() {
          std::unique_lock l(queue_mutex);
          obj_queue.push(nullptr);
          running = false;
          queue_cv.notify_all();
        }
      };
      static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

      std::unique_ptr<AckThread> ack_thread;
      uint8_t cluster_id;
      std::once_flag initialized_cluster_id;

      std::tuple<uint8_t, uint64_t, uint64_t>
      parse_cluster_message_client_node_id(const std::string &key_string) {
	std::string cluster_prefix = "cluster";
	uint8_t cluster_id = static_cast<uint8_t>(
						  std::stoll(key_string.substr(cluster_prefix.size())));
	std::string cluster_id_str = std::to_string(cluster_id);

	size_t pos = key_string.find("_");
	uint64_t message_id;
	if (pos != std::string::npos) {
	  message_id = std::stoull(key_string.substr(pos + 1));
	} else {
	  message_id = std::numeric_limits<uint64_t>::max();
        }

        size_t pos_1 = key_string.find("_", pos + 1);
        uint64_t client_node_id;
	if (pos_1 != std::string::npos) {
	  client_node_id = std::stoull(key_string.substr(pos_1 + 1));
	} else {
	  client_node_id = std::numeric_limits<uint64_t>::max();
        }        
	return {cluster_id, message_id, client_node_id}; 
      }

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
        auto [key_cluster_id, key_message_id, client_node_id] =
          parse_cluster_message_client_node_id(key_string);
        TimestampLogger::log(LOG_DUMMY_HANDLER_START, client_node_id,
                             key_message_id, object.blob.size);
        std::call_once(initialized_cluster_id,
                       &DummyAckOCDPO::initialize_cluster_id, this,
                       key_cluster_id);
        std::vector<std::shared_ptr<send_object_t>> objs =
          send_object_t::deserialize_send_objects(object.blob.bytes);
        for (const auto &obj : objs) {
	  ack_thread->enqueue_obj(obj);
        }
        TimestampLogger::log(LOG_DUMMY_HANDLER_START, client_node_id,
                             key_message_id, object.blob.size);
      }
      static void initialize() {
	if (!ocdpo_ptr) {
	  ocdpo_ptr = std::make_shared<DummyAckOCDPO>();
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

	} catch (const std::exception &e) {
	  std::cout << "error while parsing config" << std::endl;
	}
    
	this->ack_thread = std::make_unique<AckThread>(this->my_id, this);
	this->ack_thread->start(typed_ctxt);
      }

      void shutdown() {
	if (ack_thread) {
	  ack_thread->signal_stop();
	  ack_thread->join();
	}
      }
    };
    std::shared_ptr<OffCriticalDataPathObserver> DummyAckOCDPO::ocdpo_ptr;
    void initialize(ICascadeContext *ctxt);

    std::shared_ptr<OffCriticalDataPathObserver>
    get_observer(ICascadeContext *ctxt, const nlohmann::json &config);
  } // namespace cascade
} // namespace derecho

