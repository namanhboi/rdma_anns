#include "aux_utils.h"
#include "percentile_stats.h"
#include "state_send_client.h"
#include "types.h"
#include "utils.h"
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>


void write_results_csv(
    const std::vector<std::shared_ptr<search_result_t>> &results,
    const std::vector<double> &send_timestamp,
    const std::vector<double> &receive_timestamp,
		       const std::string &output_file) {
  std::ofstream output(output_file);
  output << std::setprecision (15);
  auto partition_history_to_str = [](std::shared_ptr<search_result_t> result) {
    std::stringstream str;
    str << "[";
    for (const auto &id : result->partition_history) {
      str << (int)id << ",";
    }
    str << "],";
    str << "[";
    for (const auto &node_id : result->partition_history_hop_idx) {
      str << node_id << ",";
    }
    str << "]";
    return str.str();
  };

  output << "query_id"
         << "," << "send_timestamp_ns"
         << "," << "receive_timestamp_ns"
         << "," << "completion_time_us" << "," << "io_us" << ","
         << "n_hops" << "," << "n_ios"
         << "," << "n_cmps" << ","
         << "partition_history"
         << ",partition_history_hop_idx"
            "\n";
  for (auto i = 0; i < results.size(); i++) {
    output << results[i]->query_id << "," << send_timestamp[i] << ","
    << receive_timestamp[i] << ",";
    if (results[i]->stats) {
      output << results[i]->stats->total_us << "," << results[i]->stats->io_us
             << "," << results[i]->stats->n_hops << ","
             << results[i]->stats->n_ios << "," << results[i]->stats->n_cmps
      << ",";
    } else {
      output << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ",";
    }
    
    output << partition_history_to_str(results[i]) << "\n";
  }
}

namespace po = boost::program_options;

template <typename T>
int search_disk_index(uint64_t num_client_thread, uint64_t dim,
                      std::string query_bin, std::string truthset_bin,
                      std::vector<uint64_t> &Lvec, uint64_t beam_width,
                      uint64_t K, uint64_t mem_L, bool record_stats, bool write_query_csv, 
                      std::string dist_search_mode_str, uint64_t client_peer_id,
                      uint64_t send_rate_per_second,
                      const std::vector<std::string> &address_list,
                      const std::string &result_output_folder,
                      const std::string& partition_assignment_file) {

  uint64_t microsecond_sleep_time = 0;
  if (send_rate_per_second != 0) {
    microsecond_sleep_time = 1 * 1000 * 1000 / send_rate_per_second;
  }

  // uint64_t num_queries_to_send =
  //   query_data["num_queries_to_send"].<uint64_t>();

  DistributedSearchMode dist_search_mode;


  if (dist_search_mode_str == "STATE_SEND") {
    dist_search_mode = DistributedSearchMode::STATE_SEND;
  } else if (dist_search_mode_str == "SCATTER_GATHER") {
    dist_search_mode = DistributedSearchMode::SCATTER_GATHER;
  }else if (dist_search_mode_str == "SINGLE_SERVER") {
    dist_search_mode = DistributedSearchMode::SINGLE_SERVER;
  } else if (dist_search_mode_str == "DISTRIBUTED_ANN") {
    dist_search_mode = DistributedSearchMode::DISTRIBUTED_ANN;
  } else {
    throw std::invalid_argument("Dist search mode has weird value " +
                                dist_search_mode_str);
  }

  // if (beam_width != 1 && dist_search_mode !=
  // DistributedSearchMode::DISTRIBUTED_ANN) { throw std::invalid_argument(
  // "beam_width should be 1, other sizes not yet impl");
  // }



  if (dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
    if (beam_width > distributedann::MAX_BEAM_WIDTH_DISTRIBUTED_ANN) {
      throw std::invalid_argument("beam width too large for distributedann");
    }
  } else {
    if (beam_width > BALANCE_BATCH_MAX_BEAMWIDTH) {
      throw std::invalid_argument("Beam width too large for balance batch "
                                  "(includes scatter gather and state send");
    }
  }

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  StateSendClient<T> client(client_peer_id, address_list, num_client_thread,
                            dist_search_mode, dim, partition_assignment_file);
  // client.start_result_thread();
  // client.start_client_threads();
  client.start();

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

  
  // for debugging purposes
  // size_t num_queries_to_run = 1;
  // query_num = std::min(num_queries_to_run, query_num);

  auto run_tests = [&](uint32_t test_id, bool output) {
    uint64_t L = Lvec[test_id];

    query_result_ids[test_id].resize(K * query_num);
    query_result_dists[test_id].resize(K * query_num);
    query_result_tags[test_id].resize(K * query_num);

    std::vector<uint64_t> query_result_tags_64(K * query_num);
    std::vector<uint32_t> query_result_tags_32(K * query_num);

    std::vector<uint64_t> query_ids;
    for (int i = 0; i < (int64_t)query_num; i += 1) {
      uint64_t query_id = client.search(query + (i * query_dim), K, mem_L, L,
                                        beam_width, record_stats);
      query_ids.push_back(query_id);
      if (microsecond_sleep_time != 0) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(microsecond_sleep_time));
      }
    }
    client.wait_results(query_num);

    std::vector<std::shared_ptr<search_result_t>> results;
    std::vector<double> e2e_latencies;
    std::vector<double> send_timestamp;
    std::vector<double> receive_timestamp;
    std::vector<double> query_completion_time;
    double sum_e2e_latencies = 0;
    std::chrono::steady_clock::time_point first =
        std::chrono::steady_clock::now();
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
      send_timestamp.push_back(
          std::chrono::duration<double, std::nano>(sent.time_since_epoch())
              .count());
      auto received = client.get_query_result_time(query_id);
      receive_timestamp.push_back(
          std::chrono::duration<double, std::nano>(received.time_since_epoch())
              .count());

      std::chrono::microseconds elapsed =
          std::chrono::duration_cast<std::chrono::microseconds>(received -
                                                                sent);
      double lat = static_cast<double>(elapsed.count());
      e2e_latencies.push_back(lat);
      sum_e2e_latencies += lat;
      // sum_query_completion_time += result->query_time;

      first = std::min(first, sent);
      last = std::max(last, received);
      // std::cout << client.get_query_latency_milli(query_id) << std::endl;

      std::memcpy(query_result_tags_32.data() + i * K, result->node_id,
                  sizeof(uint32_t) * K);
      std::memcpy(query_result_dists[test_id].data() + i * K, result->distance,
                  sizeof(float) * K);
      i++;
    }
    std::string result_file =
      result_output_folder + "/result_L_" + std::to_string(L) + ".csv";
    if (write_query_csv) {
      LOG(INFO) << "WRITE_QUERY_CSV " << write_query_csv;
      write_results_csv(results, send_timestamp, receive_timestamp,
                        result_file);
    }
    std::sort(e2e_latencies.begin(), e2e_latencies.end());
    std::chrono::duration<double> total_elapsed = last - first;
    // std::cout << "total time is " << (double) total_elapsed.count() <<
    // std::endl;
    float qps = (float)((1.0 * (double)query_num) /
                        (1.0 * (double)total_elapsed.count()));

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
    float mean_cmps = (float)get_mean_stats(
        query_stats, query_num, [](const std::shared_ptr<QueryStats> &stats) {
          return stats ? stats->n_cmps : 0;
    });


    float mean_inter_partition_hops = (float)get_mean_stats(
        query_stats, query_num, [](const std::shared_ptr<QueryStats> &stats) {
          return stats ? stats->n_inter_partition_hops : 0;
    });
    
    

    // double mean_query_completion_time =
    // sum_query_completion_time / query_completion_time.size();
    double mean_e2e_latency = sum_e2e_latencies / e2e_latencies.size();
    // auto latency_999 = latencies[(uint64_t)(latencies.size() * 0.999)];
    // float mean_ios = 0, mean_hops = 0;
    size_t p50_idx = e2e_latencies.size() / 2;
    double p50_latency = e2e_latencies[p50_idx];

    // 99th percentile
    size_t p999_idx = static_cast<size_t>(e2e_latencies.size() * 0.999);
    double p999_latency = e2e_latencies[p999_idx];
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
                << std::setw(12) << qps << std::setw(12) << mean_e2e_latency
                << std::setw(12) << p999_latency << std::setw(12) << mean_hops
                << std::setw(12) << mean_inter_partition_hops << std::setw(12)
      << mean_ios << std::setw(12) << mean_cmps;
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
  std::cout << std::setw(6) << "L" << std::setw(12) << "I/O Width"
            << std::setw(12) << "QPS" << std::setw(12) << "AvgLat(us)"
            << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops"
            << std::setw(12) << "Mean inter" << std::setw(12) << "Mean IOs"
  << std::setw(12) << "Mean cmps" << std::setw(12);
  if (calc_recall_flag) {
    std::cout << std::setw(12) << recall_string << std::endl;
  } else
    std::cout << std::endl;
  std::cout << "=============================================="
               "======================================================="
  << "==============="
            << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    run_tests(test_id, true);
  }
  client.shutdown();

  return 0;
}

int main(int argc, char **argv) {
  po::options_description desc("Options here mate");

  std::string data_type;
  uint64_t num_client_thread;
  uint64_t dim;
  std::string query_bin;
  std::string truthset_bin;
  std::vector<uint64_t> Lvec;
  uint64_t beam_width;
  uint64_t K;
  uint64_t mem_L;
  bool record_stats;
  bool write_query_csv; 
  std::string dist_search_mode_str;
  uint64_t client_peer_id;
  uint64_t send_rate_per_second;
  std::vector<std::string> address_list;
  std::string result_output_folder;
  std::string partition_assignment_file;

  desc.add_options()("help,h", "show help message")(
      "num_client_thread",
      po::value<uint64_t>(&num_client_thread)->default_value(1))(
      "dim", po::value<uint64_t>(&dim)->required(),
      "Dimension")("query_bin", po::value<std::string>(&query_bin)->required(),
                   "Query binary file path")(
      "truthset_bin", po::value<std::string>(&truthset_bin)->required(),
      "Truthset binary file path")(
      "L", po::value<std::vector<uint64_t>>(&Lvec)->multitoken()->required(),
      "L values")("beam_width", po::value<uint64_t>(&beam_width)->required(),
                  "Beam width")("K", po::value<uint64_t>(&K)->required(),
                                "K value")(
      "mem_L", po::value<uint64_t>(&mem_L)->required(),
      "Memory L")("record_stats", po::value<bool>(&record_stats)->required(),
                  "Record statistics flag")(
      "dist_search_mode",
      po::value<std::string>(&dist_search_mode_str)->required(),
      "Distance search mode")("client_peer_id",
                              po::value<uint64_t>(&client_peer_id)->required(),
                              "Client peer ID")(
      "send_rate", po::value<uint64_t>(&send_rate_per_second)->required(),
      "Send rate per second")("address_list",
                              po::value<std::vector<std::string>>(&address_list)
                                  ->multitoken()
                                  ->required(),
                              "Address list")(
      "data_type", po::value<std::string>(&data_type)->required(),
      "data type")("result_output_folder",
                   po::value<std::string>(&result_output_folder)->required(),
                   "path to save result csv")(
      "partition_assignment_file",
      po::value<std::string>(&partition_assignment_file)->default_value(""),
      "path to partition_assignment_file for distributedann orchestration "
      "service")("write_query_csv",
                 po::value<bool>(&write_query_csv)->required(),
                 "if true then writes information about every single query for "
                 "each L value into a csv file.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);

  if (dist_search_mode_str == "DISTRIBUTED_ANN" &&
      !file_exists(partition_assignment_file)) {
    throw std::invalid_argument(
        "partition_assignment_file has to exist if mode is distributed ann: " +
        partition_assignment_file);
  }

  if (data_type == "uint8") {
    search_disk_index<uint8_t>(num_client_thread, dim, query_bin, truthset_bin,
                               Lvec, beam_width, K, mem_L, record_stats, write_query_csv, 
                               dist_search_mode_str, client_peer_id,
                               send_rate_per_second, address_list,
                               result_output_folder, partition_assignment_file);
  } else if (data_type == "int8") {
    search_disk_index<int8_t>(num_client_thread, dim, query_bin, truthset_bin,
                              Lvec, beam_width, K, mem_L, record_stats, write_query_csv, 
                              dist_search_mode_str, client_peer_id,
                              send_rate_per_second, address_list,
                              result_output_folder, partition_assignment_file);
    
  } else if (data_type == "float") {
    search_disk_index<float>(num_client_thread, dim, query_bin, truthset_bin,
                             Lvec, beam_width, K, mem_L, record_stats, write_query_csv, 
                             dist_search_mode_str, client_peer_id,
                             send_rate_per_second, address_list,
                             result_output_folder, partition_assignment_file);
  } else {
    throw std::invalid_argument(
        "data type in json file is not uint8, int8, float " + data_type);
  }
}
