/*
  This file is used to load in all the data necessary to run a benchmark. This includes all graphs for all shards in KV store.
*/
#include <cascade/service_client_api.hpp>
#include <ParlayANN/



using namespace derecho::cascade;


#define PROC_NAME "setup"


int main(int argc, char** argv) {
    std::cout << "1) Load configuration and connecting to cascade service..." << std::endl;
    ServiceClientAPI& capi = ServiceClientAPI::get_service_client();
    std::cout << " -- connected" << std::endl;
  
  
  

}
