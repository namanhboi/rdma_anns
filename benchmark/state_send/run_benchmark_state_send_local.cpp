#include "aux_utils.h"
#include "benchmark_client.hpp"
#include "communicator.h"
#include "linux_aligned_file_reader.h"
#include "percentile_stats.h"
#include "ssd_partition_index.h"
#include <chrono>
#include <iomanip>
#include <memory_resource>
#include <thread>

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
  std::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    std::cout << std::setw(8) << percentiles[s] << "%";
  }
  std::cout << std::endl;
  std::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    std::cout << std::setw(9) << results[s];
  }
  std::cout << std::endl;
}

template <typename T> int search_disk_index(int argc, char **argv) {
  // load query bin
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num, query_dim, gt_num, gt_dim;
  std::vector<uint64_t> Lvec;

  bool tags_flag = true;

  int index = 2;
  std::string index_prefix_path(argv[index++]);
  uint32_t num_threads = std::atoi(argv[index++]);
  uint32_t beamwidth = std::atoi(argv[index++]);
  std::string query_bin(argv[index++]);
  std::string truthset_bin(argv[index++]);
  uint64_t recall_at = std::atoi(argv[index++]);
  std::string dist_metric(argv[index++]);
  uint32_t search_mode = std::atoi(argv[index++]);
  uint32_t mem_L = std::atoi(argv[index++]);

  pipeann::Metric m =
      dist_metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
  if (dist_metric != "l2" && m == pipeann::Metric::L2) {
    std::cout << "Unknown distance metric: " << dist_metric
              << ". Using default(L2) instead." << std::endl;
  }

  std::string disk_index_tag_file = index_prefix_path + "_disk.index.tags";

  bool calc_recall_flag = false;

  for (int ctr = index; ctr < argc; ctr++) {
    uint64_t curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at"
              << std::endl;
    return -1;
  }

  std::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    std::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    std::cout << " beamwidth: " << beamwidth << std::endl;

  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);
  // std::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
  // query_aligned_dim);

  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                           &tags);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data"
                << std::endl;
    }
    calc_recall_flag = true;
  }

  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());

  uint32_t num_partitions = 1;
  std::unique_ptr<P2PCommunicator> null_com;
  std::unique_ptr<SSDPartitionIndex<T>> _pFlashIndex(new SSDPartitionIndex<T>(
									      m, 0, num_partitions, num_threads, reader, null_com, tags_flag));

  int res = _pFlashIndex->load(index_prefix_path.c_str(), true);

  if (res != 0) {
    return res;
  }

  if (mem_L != 0) {
    auto mem_index_path = index_prefix_path + "_mem.index";
    LOG(INFO) << "Load memory index " << mem_index_path << " " << query_dim;
    _pFlashIndex->load_mem_index(m, query_dim, mem_index_path);
  }
  _pFlashIndex->start();
  std::this_thread::sleep_for(std::chrono::seconds(1));

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  auto run_tests = [&](uint32_t test_id, bool output) {
    pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
    uint64_t L = Lvec[test_id];

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);
    auto s = std::chrono::high_resolution_clock::now();
    
    std::shared_ptr<std::atomic<uint64_t>> completion_count =
      std::make_shared<std::atomic<uint64_t>>(0);
    for (int i = 0; i < (int64_t)query_num; i += 1) {
      _pFlashIndex->search_ssd_index_local(
					   query + (i * query_dim), i, (uint64_t)recall_at, mem_L, (uint64_t)L,
          query_result_tags_32.data() + (i * recall_at),
          query_result_dists[test_id].data() + (i * recall_at),
					   (uint64_t)beamwidth, completion_count);
      // std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    while (*completion_count != query_num) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps =
        (float)((1.0 * (double)query_num) / (1.0 * (double)diff.count()));

    pipeann::convert_types<uint32_t, uint32_t>(
        query_result_tags_32.data(), query_result_tags[test_id].data(),
        (size_t)query_num, (size_t)recall_at);

    float mean_latency = (float)pipeann::get_mean_stats(
        stats, query_num,
        [](const pipeann::QueryStats &stats) { return stats.total_us; });

    float latency_999 = (float)pipeann::get_percentile_stats(
        stats, query_num, 0.999f,
        [](const pipeann::QueryStats &stats) { return stats.total_us; });

    float mean_hops = (float)pipeann::get_mean_stats(
        stats, query_num,
        [](const pipeann::QueryStats &stats) { return stats.n_hops; });

    float mean_ios = (float)pipeann::get_mean_stats(
        stats, query_num,
        [](const pipeann::QueryStats &stats) { return stats.n_ios; });

    delete[] stats;

    if (output) {
      float recall = 0;
      if (calc_recall_flag) {
        /* Attention: in SPACEV, there may be multiple vectors with the same
          distance, which may cause lower than expected recall@1 (?) */
        recall = (float)pipeann::calculate_recall(
            (uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
            query_result_tags[test_id].data(), (uint32_t)recall_at,
            (uint32_t)recall_at);
      }

      std::cout << std::setw(6) << L << std::setw(12) << beamwidth
                << std::setw(12) << qps << std::setw(12) << mean_latency
                << std::setw(12) << latency_999 << std::setw(12) << mean_hops
                << std::setw(12) << mean_ios;
      if (calc_recall_flag) {
        std::cout << std::setw(12) << recall << std::endl;
      }
    }
  };
  LOG(INFO) << "Use two ANNS for warming up...";
  uint32_t prev_L = Lvec[0];
  Lvec[0] = 200;
  run_tests(0, false);
  run_tests(0, false);
  Lvec[0] = prev_L;
  LOG(INFO) << "Warming up finished.";

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(6) << "L" << std::setw(12) << "I/O Width" << std::setw(12) << "QPS" << std::setw(12)
            << "AvgLat(us)" << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops" << std::setw(12) << "Mean IOs"
            << std::setw(12);
  if (calc_recall_flag) {
    std::cout << std::setw(12) << recall_string << std::endl;
  } else
    std::cout << std::endl;
  std::cout << "=============================================="
               "==========================================="
            << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    run_tests(test_id, true);
  }
  _pFlashIndex->shutdown();
  return 0;
}

/**
   export INDEX_PREFIX=/home/nam/big-ann-benchmarks/data/bigann/pipeann_10M
   ./build/benchmark/state_send/run_benchmark_state_send uint8 ${INDEX_PREFIX}
16 32  /home/nam/big-ann-benchmarks/data/bigann/query.public.10K.u8bin
/home/nam/big-ann-benchmarks/data/bigann/bigann-10M 10 l2 0 10 10 20 30 40

./build/benchmark/state_send/run_benchmark_state_send_local uint8 ${INDEX_PREFIX} 1 1 /home/nam/big-ann-benchmarks/data/bigann/query.public.10K.u8bin /home/nam/big-ann-benchmarks/data/bigann/bigann-10M 10 l2 0 0 10 20 30 40
*/

int main(int argc, char **argv) {
  if (argc < 12) {
    // tags == 1!
    std::cout << "Usage: " << argv[0]
              << " <index_type (float/int8/uint8)>  <index_prefix_path>"
                 " <num_threads>  <pipeline width> "
                 " <query_file.bin>  <truthset.bin (use \"null\" for none)> "
                 " <K> <similarity (cosine/l2)> "
                 " <search_mode(0 for beam search / 1 for page search / 2 for "
                 "pipe search)> <mem_L (0 means not "
                 "using mem index)> <L1> [L2] etc."
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8"
              << std::endl;
}
