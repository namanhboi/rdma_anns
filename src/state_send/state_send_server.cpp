#include "communicator.h"
#include "ssd_partition_index.h"
#include "types.h"
#include <chrono>
#include <concepts>
#include <memory>
#include <nlohmann/json.hpp>
#include <ratio>
#include <stdexcept>

#include <csignal>
#include <thread>
std::atomic<bool> should_kill_server = false;


template <typename T> class StateSendServer {
private:
  std::shared_ptr<AlignedFileReader> reader;
  std::unique_ptr<P2PCommunicator> communicator;
  std::unique_ptr<SSDPartitionIndex<T>> ssd_partition_index;

public:
  // partition id is also the id for communicator, so server 1 will be in charge
  // of partition 1
  // need to set is_local = false since this is doing some communication via
  // tcp/rdma
  StateSendServer(const std::string &communicator_json,
                  const std::string &index_prefix,
                  const std::string &cluster_assignment_file, pipeann::Metric m,
                  uint8_t my_partition_id,
                  uint32_t num_search_threads, bool use_mem_index, DistributedSearchMode dist_search_mode,bool tags, uint64_t batch_size, bool enable_locs, bool use_batching, uint64_t max_batch_size) {
    communicator = std::make_unique<ZMQP2PCommunicator>(
        static_cast<uint64_t>(my_partition_id), communicator_json);
    reader = std::make_shared<LinuxAlignedFileReader>();
    const char* cluster_file_ptr;
    if (cluster_assignment_file != "") {
      cluster_file_ptr = cluster_assignment_file.c_str();
    }
    ssd_partition_index = std::make_unique<SSDPartitionIndex<T>>(
        m, my_partition_id, num_search_threads, reader,
								 communicator, dist_search_mode, tags, nullptr, batch_size, enable_locs, use_batching, max_batch_size);
    int res =
      ssd_partition_index->load(index_prefix.c_str(), true, cluster_file_ptr);
    if (res != 0) {
      std::runtime_error("error loading index");
    }

    if (use_mem_index) {
      auto mem_index_path = index_prefix + "_mem.index";
      LOG(INFO) << "Load memory index " << mem_index_path;
      ssd_partition_index->load_mem_index(
					  m, ssd_partition_index->get_data_dim(), mem_index_path);
    }
    communicator->register_receive_handler(
        [index_ptr = (ssd_partition_index.get())](const char *buffer,
                                                  size_t size) {
          index_ptr->receive_handler(buffer, size);
        });
    std::cout << "done with constructor" << std::endl;
  }
  void start() {
    communicator->start_recv_thread();
    ssd_partition_index->start();
  }

  void signal_stop() {
    ssd_partition_index->shutdown();
    communicator->stop_recv_thread();
  }
};


template <typename T>
void run_server(std::unique_ptr<StateSendServer<T>> server) {
  server->start();
  std::cout <<"started server" <<std::endl;
  
  while (should_kill_server == false) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  server->signal_stop();
}



void sigint_handler(int signal) {
  if (signal == SIGINT)
    should_kill_server = true;
}


/**
   server id: both the partition id for index prefix and also the peer id in communicator json
   index json contains the parameters for the index, note that index_prefix is just the partial prefix, and the final index prefix will be: index_prefix + server_id
   commmunicator json contains the addresses of all p2p nodes
*/
int main(int argc, char **argv) {
  std::signal(SIGINT, sigint_handler);
  
  uint64_t server_id = std::stoull(argv[1]);
  std::string index_json(argv[2]);
  std::string communicator_json(argv[3]);

  std::ifstream f(index_json);
  json data = json::parse(f);
  std::string type = data["type"].get<std::string>();
  std::string index_prefix = data["index_prefix"].get<std::string>();
  index_prefix += std::to_string(server_id);
  std::string cluster_assignment_file =
      data["cluster_assignment_file"].get<std::string>();
  uint32_t num_search_threads = data["num_search_threads"].get<uint32_t>();
  bool use_tags = data["use_tags"].get<bool>();
  bool enable_locs = data["enable_locs"].get<bool>();
  bool use_mem_index = data["use_mem_index"].get<bool>();
  std::string metric = data["metric"].get<std::string>();

  uint64_t num_queries_balance = data["num_queries_balance"].get<uint64_t>();
  // uint64_t max_batch_size = data["max_batch_size"].get<uint64_t>();
  std::string dist_search_mode_str =
    data["dist_search_mode"].get<std::string>();
  // bool use_batching = data["use_batching"].get<bool>();
  // uint64_t max_batch_size = data["max_batch_size"].get<bool>();
  bool use_batching = data.value<bool>("use_batching", false);
  uint64_t max_batch_size = data.value<uint64_t>("max_batch_size", 0);

  DistributedSearchMode dist_search_mode;


  if (type != "uint8" && type != "int8" && type != "float") {
    throw std::invalid_argument("data type doesn't make sense");
  }
  if (dist_search_mode_str == "STATE_SEND") {
    dist_search_mode = DistributedSearchMode::STATE_SEND;
  } else if (dist_search_mode_str == "SCATTER_GATHER") {
    dist_search_mode = DistributedSearchMode::SCATTER_GATHER;
  } else {
    throw std::invalid_argument("Dist search mode has weird value " +
                                dist_search_mode_str);
  }

  if (dist_search_mode == DistributedSearchMode::SCATTER_GATHER) {
    if (use_tags == false) {
      throw std::invalid_argument("use_tags must be true if we are doing scatter gather");
    }

  }
  
  pipeann::Metric m =
      metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
  if (metric != "l2" && m == pipeann::Metric::L2) {
    std::cout << "Unknown distance metric: " << metric
              << ". Using default(L2) instead." << std::endl;
  }

  if (type == "uint8") {
    auto server = std::make_unique<StateSendServer<uint8_t>>(
        communicator_json, index_prefix, cluster_assignment_file, m,
        server_id, num_search_threads, use_mem_index,
							     dist_search_mode, use_tags, num_queries_balance, enable_locs, use_batching, max_batch_size);
    run_server(std::move(server));
  } else if (type == "int8") {
    auto server = std::make_unique<StateSendServer<int8_t>>(
        communicator_json, index_prefix, cluster_assignment_file, m,
        server_id, num_search_threads, use_mem_index,
							    dist_search_mode, use_tags, num_queries_balance, enable_locs, use_batching, max_batch_size);
    run_server(std::move(server));
  } else if (type == "float") {
    auto server = std::make_unique<StateSendServer<float>>(
        communicator_json, index_prefix, cluster_assignment_file, m,
        server_id, num_search_threads, use_mem_index,
							   dist_search_mode, use_tags, num_queries_balance, enable_locs, use_batching, max_batch_size);
    run_server(std::move(server));
  }

  return 0;
}
