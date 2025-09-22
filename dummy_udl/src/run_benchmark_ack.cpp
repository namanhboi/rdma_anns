#include "benchmark_client_ack.hpp"
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

uint8_t get_random_cluster_id(int num_clusters) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> cluster_id_gen(0, num_clusters - 1);
  return cluster_id_gen(gen);
}

namespace po = boost::program_options;

void benchmark(uint64_t num_msg, uint64_t num_bytes, int num_clusters,
               int mili_sleep) {
  BenchmarkClient client;
  client.setup(1);
  for (auto i = 0; i < num_msg; i++) {
    uint8_t cluster_id = get_random_cluster_id(num_clusters);
    client.issue_send_object(num_bytes, cluster_id);
    std::this_thread::sleep_for(std::chrono::milliseconds(mili_sleep));
  }
  client.wait_acks();
  client.dump_timestamp();
}

int main(int argc, char **argv) {
  po::options_description desc("Options here mate");
  uint64_t num_bytes, num_msg;

  int mili_sleep, num_clusters;

  desc.add_options()("help,h", "show help message")(
      "num_msg", po::value<uint64_t>(&num_msg)->required(),
      "num send_requests")("num_bytes",
                           po::value<uint64_t>(&num_bytes)->required(),
                           "max bytes in send request")(
      "mili_sleep", po::value<int>(&mili_sleep)->default_value(1),
      "number of miliseconds to sleep for ")(
      "num_clusters", po::value<int>(&num_clusters)->default_value(1),
      "number of clusters");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);

  std::cout << "num clusters as args is " << num_clusters << std::endl;
  benchmark(num_msg, num_bytes, num_clusters, mili_sleep);
}
