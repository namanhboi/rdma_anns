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

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>

std::atomic<bool> should_kill_server = false;
namespace po = boost::program_options;

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
  StateSendServer(const std::vector<std::string> &address_list,
                  const std::string &index_prefix, pipeann::Metric m,
                  uint8_t my_partition_id, uint32_t num_search_threads,
                  bool use_mem_index, DistributedSearchMode dist_search_mode,
                  bool tags, uint64_t batch_size, bool enable_locs,
                  bool use_batching, uint64_t max_batch_size,
                  bool use_counter_thread, std::string counter_csv,
                  uint64_t counter_sleep_ms, std::string log_file) {
    communicator = std::make_unique<ZMQP2PCommunicator>(
        static_cast<uint64_t>(my_partition_id), address_list);
    reader = std::make_shared<LinuxAlignedFileReader>();

    ssd_partition_index = std::make_unique<SSDPartitionIndex<T>>(
        m, my_partition_id, num_search_threads, reader, communicator,
        dist_search_mode, tags, nullptr, batch_size, enable_locs, use_batching,
        max_batch_size, use_counter_thread, counter_csv, counter_sleep_ms,
								 log_file);
    int res = ssd_partition_index->load(index_prefix.c_str(), true);
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
  std::cout << "started server" << std::endl;

  while (should_kill_server == false) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  server->signal_stop();
}

void sigint_handler(int signal) {
  if (signal == SIGINT)
    should_kill_server = true;
}

void parse_args(int argc, char **argv) {}

/**
   server id: both the partition id for index prefix and also the peer id in
   communicator json index json contains the parameters for the index, note that
   index_prefix is just the partial prefix, and the final index prefix will be:
   index_prefix + server_id commmunicator json contains the addresses of all p2p
   nodes
*/
int main(int argc, char **argv) {
  std::signal(SIGINT, sigint_handler);
  po::options_description desc("Options here mate");
  uint64_t server_peer_id;
  std::vector<std::string> address_list;

  std::string data_type;
  std::string index_path_prefix;

  uint32_t num_search_threads;
  bool use_tags;
  bool enable_locs;
  bool use_mem_index;
  std::string metric;

  uint64_t num_queries_balance;
  // uint64_t max_batch_size = data["max_batch_size"].get<uint64_t>();
  std::string dist_search_mode_str;
  // bool use_batching = data["use_batching"].get<bool>();
  // uint64_t max_batch_size = data["max_batch_size"].get<bool>();
  bool use_batching;
  uint64_t max_batch_size;

  bool use_counter_thread;
  std::string counter_csv;
  uint64_t counter_sleep_ms;
  std::string log_file;

  desc.add_options()("help,h", "show help message")(
      "server_peer_id", po::value<uint64_t>(&server_peer_id)->required(),
      "Server peer ID")("address_list",
                        po::value<std::vector<std::string>>(&address_list)
                            ->multitoken()
                            ->required(),
                        "Address list")(
      "data_type", po::value<std::string>(&data_type)->required(), "Data type")(
      "index_path_prefix",
      po::value<std::string>(&index_path_prefix)->required(),
      "Index path prefix")("num_search_threads",
                           po::value<uint32_t>(&num_search_threads)->required(),
                           "Number of search threads")(
      "use_tags", po::value<bool>(&use_tags)->required(), "Use tags flag")(
      "enable_locs", po::value<bool>(&enable_locs)->required(),
      "Enable locations flag")("use_mem_index",
                               po::value<bool>(&use_mem_index)->required(),
                               "Use memory index flag")(
      "metric", po::value<std::string>(&metric)->required(),
      "Metric")("num_queries_balance",
                po::value<uint64_t>(&num_queries_balance)->required(),
                "Number of queries balance")(
      "dist_search_mode",
      po::value<std::string>(&dist_search_mode_str)->required(),
      "Distance search mode")("use_batching",
                              po::value<bool>(&use_batching)->required(),
                              "Use batching flag")(
      "max_batch_size", po::value<uint64_t>(&max_batch_size)->required(),
      "Maximum batch size")("use_counter_thread",
                            po::value<bool>(&use_counter_thread)->required(),
                            "whether to use teh counter thread to collect data "
                            "about the internal state of ssdpartitionindex")(
      "counter_csv", po::value<std::string>(&counter_csv)->required(),
      "path to save result of counter thread")(
      "counter_sleep_ms", po::value<uint64_t>(&counter_sleep_ms)->required(),
      "how long should counter thraed sleep before waking up to collect data")(
      "log_file", po::value<std::string>(&log_file)->required(),
									       "where to log to, if set to empty str then won't log");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);
  std::cout << "======== ARGUMENTS ========" << std::endl;
  std::cout << "server_peer_id: " << server_peer_id << std::endl;
  std::cout << "address_list: ";
  for (const auto &addr : address_list)
    std::cout << addr << " ";
  std::cout << std::endl;
  std::cout << "data_type: " << data_type << std::endl;
  std::cout << "index_path_prefix: " << index_path_prefix << std::endl;
  std::cout << "num_search_threads: " << num_search_threads << std::endl;
  std::cout << "use_tags: " << use_tags << std::endl;
  std::cout << "enable_locs: " << enable_locs << std::endl;
  std::cout << "use_mem_index: " << use_mem_index << std::endl;
  std::cout << "metric: " << metric << std::endl;
  std::cout << "num_queries_balance: " << num_queries_balance << std::endl;
  std::cout << "dist_search_mode: " << dist_search_mode_str << std::endl;
  std::cout << "use_batching: " << use_batching << std::endl;
  std::cout << "max_batch_size: " << max_batch_size << std::endl;
  std::cout << "===========================" << std::endl;

  index_path_prefix += std::to_string(server_peer_id);
  DistributedSearchMode dist_search_mode;

  if (data_type != "uint8" && data_type != "int8" && data_type != "float") {
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
      throw std::invalid_argument(
          "use_tags must be true if we are doing scatter gather");
    }
  }

  pipeann::Metric m =
      metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
  if (metric != "l2" && m == pipeann::Metric::L2) {
    std::cout << "Unknown distance metric: " << metric
              << ". Using default(L2) instead." << std::endl;
  }

  if (data_type == "uint8") {
    auto server = std::make_unique<StateSendServer<uint8_t>>(
        address_list, index_path_prefix, m, server_peer_id, num_search_threads,
        use_mem_index, dist_search_mode, use_tags, num_queries_balance,
        enable_locs, use_batching, max_batch_size, use_counter_thread,
							     counter_csv, counter_sleep_ms,log_file);
    run_server(std::move(server));
  } else if (data_type == "int8") {
    auto server = std::make_unique<StateSendServer<int8_t>>(
        address_list, index_path_prefix, m, server_peer_id, num_search_threads,
        use_mem_index, dist_search_mode, use_tags, num_queries_balance,
        enable_locs, use_batching, max_batch_size, use_counter_thread,
							    counter_csv, counter_sleep_ms,log_file);
    run_server(std::move(server));
  } else if (data_type == "float") {
    auto server = std::make_unique<StateSendServer<float>>(
        address_list, index_path_prefix, m, server_peer_id, num_search_threads,
        use_mem_index, dist_search_mode, use_tags, num_queries_balance,
        enable_locs, use_batching, max_batch_size, use_counter_thread,
							   counter_csv, counter_sleep_ms, log_file);
    run_server(std::move(server));
  }

  return 0;
}
