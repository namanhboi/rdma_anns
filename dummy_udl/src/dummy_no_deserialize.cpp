#include "dummy_no_deserialize.hpp"
#include <cascade/detail/_user_defined_logic_interface.hpp>

namespace derecho {
  namespace cascade {
    std::string get_uuid() { return MY_UUID; }
    std::string get_description() { return MY_DESC; }

    void initialize(ICascadeContext *ctxt) {
      DummyNoDeserializeOCDPO::initialize();
    }

    std::shared_ptr<OffCriticalDataPathObserver>
    get_observer(ICascadeContext *ctxt, const nlohmann::json &config) {
      auto typed_ctxt = dynamic_cast<DefaultCascadeContextType *>(ctxt);
      std::static_pointer_cast<DummyNoDeserializeOCDPO>(DummyNoDeserializeOCDPO::get())
        ->set_config(typed_ctxt, config);
      return DummyNoDeserializeOCDPO::get();
    }
  } // namespace cascade
} // namespace derecho
