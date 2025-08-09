#include "global_search_udl.hpp"
#include <cascade/detail/_user_defined_logic_interface.hpp>


namespace derecho {
namespace cascade {
  std::string get_uuid() { return MY_UUID; }
  std::string get_description() { return MY_DESC; }

  void initialize(ICascadeContext *ctxt) {
    if (std::string(DATA_TYPE) == "float") {
      GlobalSearchOCDPO<float>::initialize();

    } else if (std::string(DATA_TYPE) == "uint8_t") {
      GlobalSearchOCDPO<uint8_t>::initialize();
    } else if (std::string(DATA_TYPE) == "int8_t") {
      GlobalSearchOCDPO<int8_t>::initialize();
    } else {
      throw std::runtime_error("DATA_TYPE macro in global_search_udl.hpp has weird value: " DATA_TYPE);
    }
    std::cout << "done initializing global search udl" << std::endl;
  }
  
  


  std::shared_ptr<OffCriticalDataPathObserver>
get_observer(ICascadeContext *ctxt, const nlohmann::json &config) {
  auto typed_ctxt = dynamic_cast<DefaultCascadeContextType *>(ctxt);
  if (std::string(DATA_TYPE) == "float") {
    std::static_pointer_cast<GlobalSearchOCDPO<float>>(
						       GlobalSearchOCDPO<float>::get())
        ->set_config(typed_ctxt, config);
    return GlobalSearchOCDPO<float>::get();
  } else if (std::string(DATA_TYPE) == "uint8_t") {
    std::static_pointer_cast<GlobalSearchOCDPO<uint8_t>>(
							 GlobalSearchOCDPO<uint8_t>::get())
        ->set_config(typed_ctxt, config);
    return GlobalSearchOCDPO<uint8_t>::get();    
  } else if (std::string(DATA_TYPE) == "int8_t") {
    std::static_pointer_cast<GlobalSearchOCDPO<int8_t>>(
        GlobalSearchOCDPO<int8_t>::get())
        ->set_config(typed_ctxt, config);
    return GlobalSearchOCDPO<int8_t>::get();    
  } else {
    throw std::runtime_error("DATA_TYPE macro in global_index_search_udl.hpp "
                             "has weird value: " DATA_TYPE);
  }
}
} // namespace cascade
} // namespace derecho


