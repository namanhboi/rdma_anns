#include "aux_utils.h"
#include "percentile_stats.h"
#include "state_send_client.h"
#include "types.h"
#include "utils.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

template <typename T>
int search_disk_index(const std::string &query_json,
                      const std::string &communicator_json) {
  std::ifstream query_ifstream(query_json);
  json query_data = json::parse(query_ifstream);

  uint64_t client_peer_id = query_data["client_peer_id"].get<uint64_t>();
  int num_client_thread = query_data["num_client_thread"].get<int>();
  uint64_t dim = query_data["dim"].get<uint64_t>();
  std::string query_bin(query_data["query_bin"].get<std::string>());
  std::string truthset_bin(query_data["truthset_bin"].get<std::string>());
  std::vector<uint64_t> Lvec(query_data["Lvec"].get<std::vector<uint64_t>>());
  uint64_t beam_width = query_data["beam_width"].get<uint64_t>();
  uint64_t K = query_data["K"].get<uint64_t>();
  uint64_t mem_L = query_data["mem_L"].get<uint64_t>();
  bool record_stats = query_data["record_stats"].get<bool>();

  if (beam_width != 1) {
    throw std::invalid_argument("beam_width should be 1, other sizes not yet impl");
  }  
  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());


  StateSendClient<T> client(client_peer_id, communicator_json,
                            num_client_thread, dim);
  client.start_result_thread();
  client.start_client_threads();
  
  // std::ifstream communincator_ifstream(communicator_json);
  // json communicator_data = json::parse(communincator_ifstream);
  
  
  
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num, query_dim, gt_num, gt_dim;


  bool calc_recall_flag = false;


  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);
  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                           &tags);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data"
                << std::endl;
    }
    calc_recall_flag = true;
  }


  auto run_tests = [&](uint32_t test_id, bool output) {
    uint64_t L = Lvec[test_id];

    query_result_ids[test_id].resize(K * query_num);
    query_result_dists[test_id].resize(K * query_num);
    query_result_tags[test_id].resize(K * query_num);

    std::vector<uint64_t> query_result_tags_64(K * query_num);
    std::vector<uint32_t> query_result_tags_32(K * query_num);


    std::vector<uint64_t> query_ids;
    for (int i = 0; i < (int64_t)query_num; i += 1) {
      uint64_t query_id =
        client.search(query + (i * query_dim), K, mem_L, L, beam_width, record_stats);
      query_ids.push_back(query_id);
      // std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    client.wait_results(query_num);

    std::vector<std::shared_ptr<search_result_t>> results;
    std::vector<double> e2e_latencies;
    std::vector<double> query_completion_time;
    double sum_e2e_latencies = 0;
    std::chrono::steady_clock::time_point first = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last;

    size_t i = 0;
    std::vector<std::shared_ptr<QueryStats>> query_stats;
    for (const auto &query_id : query_ids) {
      auto result = client.get_result(query_id);
      // results.push_back(client.get_result(query_id));
      results.push_back(result);
      query_stats.push_back(result->stats);
      // query_completion_time.push_back(result->stats);
      auto sent = client.get_query_send_time(query_id);
      auto received = client.get_query_result_time(query_id);

      std::chrono::microseconds elapsed =
          std::chrono::duration_cast<std::chrono::microseconds>(received -
                                                                sent);
      double lat = static_cast<double>(elapsed.count());
      e2e_latencies.push_back(lat);
      sum_e2e_latencies += lat;
      // sum_query_completion_time += result->query_time;

      first = std::min(first,sent);
      last = std::max(last, received);
      // std::cout << client.get_query_latency_milli(query_id) << std::endl;

      std::memcpy(query_result_tags_32.data() + i * K, result->node_id,
                  sizeof(uint32_t) * K);
      std::memcpy(query_result_dists[test_id].data() + i * K, result->distance,
                  sizeof(float) * K);
      i++;
    }
    std::sort(e2e_latencies.begin(),e2e_latencies.end());
    std::chrono::duration<double> total_elapsed = last - first;
    // std::cout << "total time is " << (double) total_elapsed.count() << std::endl;
    float qps =
        (float)((1.0 * (double)query_num) / (1.0 * (double)total_elapsed.count()));

    pipeann::convert_types<uint32_t, uint32_t>(
        query_result_tags_32.data(), query_result_tags[test_id].data(),
					       (size_t)query_num, (size_t)K);

    float mean_latency = (float)get_mean_stats(
        query_stats, query_num, [](const std::shared_ptr<QueryStats> &stats) {
          return stats ? stats->total_us : 0;
    });

    float latency_999 = (float)get_percentile_stats(
        query_stats, query_num, 0.999f,
        [](const std::shared_ptr<QueryStats> &stats) {
          return stats ? stats->total_us : 0;
        });

    float mean_hops = (float)get_mean_stats(
        query_stats, query_num, [](const std::shared_ptr<QueryStats> &stats) {
          return stats ? stats->n_hops : 0;
    });
    

    float mean_ios = (float)get_mean_stats(
        query_stats, query_num, [](const std::shared_ptr<QueryStats> &stats) {
          return stats ? stats->n_ios : 0;
    });

    // double mean_query_completion_time =
      // sum_query_completion_time / query_completion_time.size();
    double mean_e2e_latency = sum_e2e_latencies / e2e_latencies.size();
    // auto latency_999 = latencies[(uint64_t)(latencies.size() * 0.999)];
    // float mean_ios = 0, mean_hops = 0;

    if (output) {
      float recall = 0;
      if (calc_recall_flag) {
        /* Attention: in SPACEV, there may be  multiple vectors with the same
          distance, which may cause lower than expected recall@1 (?) */
        recall = (float)pipeann::calculate_recall(
            (uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
						  query_result_tags[test_id].data(), (uint32_t)K, (uint32_t)K);
      }

      std::cout << std::setw(6) << L << std::setw(12) << beam_width
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
  Lvec[0] = 50;
  run_tests(0, false);
  Lvec[0] = prev_L;
  LOG(INFO) << "Warming up finished.";

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(K);
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
  
  return 0;
}

int main(int argc, char **argv) {
  std::string query_json(argv[1]);
  std::string communicator_json(argv[2]);


  std::ifstream query_ifstream(query_json);
  json query_data = json::parse(query_ifstream);
  std::string data_type = query_data["data_type"].get<std::string>();
  if (data_type == "uint8") {
    search_disk_index<uint8_t>(query_json, communicator_json);
  } else if (data_type == "int8") {
    search_disk_index<int8_t>(query_json, communicator_json);
  } else if (data_type == "float") {
    search_disk_index<float>(query_json, communicator_json);
  } else {
    throw std::invalid_argument(
				"data type in json file is not uint8, int8, float " + data_type);
  }
  
}
