#include "benchmark_client.hpp"
#include "serialize_utils.hpp"
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#define DEFAULT_BATCH_MIN_SIZE 0
#define DEFAULT_BATCH_MAX_SIZE 5
#define DEFAULT_BATCH_TIME_US 500
#define DEFAULT_DIMENSIONS 1024
#define DEFAULT_NUM_QUERIES 10
#define DEFAULT_NUM_RESULT_THREADS 1
#define DEFAULT_WARMUP 0
#define MAX_NUM_BYTES 1024

std::pair<uint8_t, uint8_t> get_distinct_cluster_ids(uint8_t num_clusters) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> cluster_id_gen(0, num_clusters - 1);
  uint8_t first = cluster_id_gen(gen);
  uint8_t second = cluster_id_gen(gen);
  while (second == first) {
    second = cluster_id_gen(gen);
  }
  return {first, second};
}

uint64_t get_num_bytes(uint64_t max_num_bytes) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> num_bytes_gen(1, max_num_bytes);
  return num_bytes_gen(gen);
}

namespace po = boost::program_options;

void benchmark(uint64_t num_send_requests, uint64_t max_num_bytes,
               uint8_t num_clusters, uint64_t min_batch_size,
               uint64_t max_batch_size, uint64_t batch_time_us,
               uint32_t num_result_threads) {
  BenchmarkClient client;
  client.setup(min_batch_size, max_batch_size, batch_time_us,
               num_result_threads);
  for (auto i = 0; i < num_send_requests; i++) {
    auto [cluster_sender_id, cluster_receiver_id] = get_distinct_cluster_ids(num_clusters);
    client.issue_send_request(get_num_bytes(max_num_bytes), cluster_sender_id,
                              cluster_receiver_id);
    
  }
  client.wait_acks();
  client.dump_timestamp();
  std::this_thread::sleep_for(std::chrono::seconds(2));
  
}

int main(int argc, char **argv) {
  po::options_description desc("Options here mate");
  uint64_t min_batch_size;
  uint64_t max_batch_size;
  uint64_t batch_time_us;
  uint32_t num_result_threads;
  uint64_t num_send_requests;
  uint64_t max_num_bytes;
  uint32_t num_clusters;

  desc.add_options()("help,h", "show help message")(
      "min_batch_size,B",
      po::value<uint64_t>(&min_batch_size)
          ->default_value(DEFAULT_BATCH_MIN_SIZE),
      "Min size of batch")("max_batch_size,X",
                           po::value<uint64_t>(&max_batch_size)
                               ->default_value(DEFAULT_BATCH_MAX_SIZE),
                           "max size of batch")(
      "batch_time_us,U",
      po::value<uint64_t>(&batch_time_us)->default_value(DEFAULT_BATCH_TIME_US),
      "maximum time to wait for the batch minimum size, in microseconds")(
      "num_result_threads,N",
      po::value<uint32_t>(&num_result_threads)
          ->default_value(DEFAULT_NUM_RESULT_THREADS),
      "number of threads for processing results")(
      "num_send_requests", po::value<uint64_t>(&num_send_requests)->required(),
      "num send_requests")("max_num_bytes",
                           po::value<uint64_t>(&max_num_bytes)->required(),
                           "max bytes in send request")(
      "num_clusters", po::value<uint32_t>(&num_clusters)->required(),
      "max bytes in send request");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);

  if (min_batch_size > max_batch_size)
    throw std::invalid_argument(
				"min batch size can't be bigger than max_batch size");

  if (num_clusters <= 1) {
    throw std::invalid_argument("num clusters must be bigger than 1");
  }
  std::cout << "num clusters as args is " << num_clusters << std::endl;
  benchmark(num_send_requests, max_num_bytes, static_cast<uint8_t>(num_clusters), min_batch_size,
            max_batch_size, batch_time_us, num_result_threads);
  
}

