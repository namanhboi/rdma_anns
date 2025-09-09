#include "dummy_udl.hpp"
#include <cascade/detail/_user_defined_logic_interface.hpp>


namespace derecho {
  namespace cascade {
    std::string get_uuid() { return MY_UUID; }
    std::string get_description() { return MY_DESC; }

    void initialize(ICascadeContext *ctxt) {
      DummyOCDPO::initialize();
    }

    std::shared_ptr<OffCriticalDataPathObserver>
    get_observer(ICascadeContext *ctxt, const nlohmann::json &config) {
      auto typed_ctxt = dynamic_cast<DefaultCascadeContextType *>(ctxt);
      std::static_pointer_cast<DummyOCDPO>(DummyOCDPO::get())
        ->set_config(typed_ctxt, config);
      return DummyOCDPO::get();

    }
  } // namespace cascade
} // namespace derecho


