#include <cascade/user_defined_logic_interface.hpp>
#include <iostream>
#include <stdexcept>

namespace derecho{
namespace cascade{

#define MY_UUID     "48e60f7c-8500-11eb-8755-0242ac110002"
#define MY_DESC     "Dummy UDL that just logs stuff."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class DummyNoOpOCDPO: public DefaultOffCriticalDataPathObserver {
  uint64_t my_id;

  std::pair<uint64_t, uint64_t>
  parse_client_batch_id(const std::string& key_string) {
    uint64_t client_id = std::stoul(key_string);
    auto pos = key_string.find("_");
    if (pos == std::string::npos) {
      throw std::invalid_argument("key string weird value, no _ : " +
                                  key_string);
    }
    uint64_t batch_id = std::stoul(key_string.substr(pos + 1));
    return {client_id, batch_id};
  }
  
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
      auto [client_id , batch_id] = parse_client_batch_id(key_string);
      TimestampLogger::log(LOG_DUMMY_HANDLER_START,
                           client_id, batch_id,
                           object.blob.size);
    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:
    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<DummyNoOpOCDPO>();
        }
    }
  static auto get() { return ocdpo_ptr; }

  void set_config(DefaultCascadeContextType *typed_ctxt,
                  const nlohmann::json &config) {
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    
  }    
};

std::shared_ptr<OffCriticalDataPathObserver> DummyNoOpOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    DummyNoOpOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext*,const nlohmann::json&) {
    return DummyNoOpOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho
