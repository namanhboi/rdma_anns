#include "head_index_search_udl.hpp"
#include <cascade/detail/_user_defined_logic_interface.hpp>


namespace derecho {
namespace cascade {
std::string get_uuid() { return MY_UUID; }
std::string get_description() { return MY_DESC; }

void initialize(ICascadeContext *ctxt) {
  if (std::string(DATA_TYPE) == "float") {
    HeadIndexSearchOCDPO<float>::initialize();
    std::cout << "done initializing head index udl" << std::endl;
  } else if (std::string(DATA_TYPE) == "uint8_t") {
    HeadIndexSearchOCDPO<uint8_t>::initialize();
  } else if (std::string(DATA_TYPE) == "int8_t") {
    HeadIndexSearchOCDPO<int8_t>::initialize();
  } else {
    throw std::runtime_error("DATA_TYPE macro in head_index_search_udl.hpp has weird value: " DATA_TYPE);

  }
}
 
std::shared_ptr<OffCriticalDataPathObserver>
get_observer(ICascadeContext *ctxt, const nlohmann::json &config) {
  auto typed_ctxt = dynamic_cast<DefaultCascadeContextType *>(ctxt);
  if (std::string(DATA_TYPE) == "float") {
    std::static_pointer_cast<HeadIndexSearchOCDPO<float>>(
        HeadIndexSearchOCDPO<float>::get())
        ->set_config(typed_ctxt, config);
    return HeadIndexSearchOCDPO<float>::get();
  } else if (std::string(DATA_TYPE) == "uint8_t") {
    std::static_pointer_cast<HeadIndexSearchOCDPO<uint8_t>>(
        HeadIndexSearchOCDPO<uint8_t>::get())
        ->set_config(typed_ctxt, config);
    return HeadIndexSearchOCDPO<uint8_t>::get();    
  } else if (std::string(DATA_TYPE) == "int8_t") {
    std::static_pointer_cast<HeadIndexSearchOCDPO<int8_t>>(
        HeadIndexSearchOCDPO<int8_t>::get())
        ->set_config(typed_ctxt, config);
    return HeadIndexSearchOCDPO<int8_t>::get();    
  } else {
    throw std::runtime_error("DATA_TYPE macro in head_index_search_udl.hpp has weird value: " DATA_TYPE);
  }
}

}}

  
