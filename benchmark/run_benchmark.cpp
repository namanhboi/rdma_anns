#include "benchmark_client.hpp"
#include "benchmark_dataset.hpp"
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#define DEFAULT_BATCH_MIN_SIZE 0
#define DEFAULT_BATCH_MAX_SIZE 5
#define DEFAULT_BATCH_TIME_US 500
#define DEFAULT_DIMENSIONS 1024
#define DEFAULT_NUM_QUERIES 10
#define DEFAULT_NUM_RESULT_THREADS 1
#define DEFAULT_WARMUP 0


#define HEAD_INDEX_K 10
namespace po = boost::program_options;

template <typename data_type>
void benchmark(const std::string &query_file, const std::string &gt_file,
               uint32_t batch_min_size, uint32_t batch_max_size,
               uint32_t batch_time_us, uint32_t num_result_threads,
               uint32_t num_warmup, uint32_t send_rate) {
  BenchmarkDataset<data_type> dataset(query_file, gt_file);
  BenchmarkClient<data_type> client(num_warmup);

  uint32_t dim = dataset.get_dim();
  bool rate_control = false;
  std::chrono::nanoseconds iteration_time;
  if(send_rate != 0){
    rate_control = true;
    iteration_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)) / send_rate;
  }
  client.setup(batch_min_size, batch_max_size, batch_time_us, dim,
               num_result_threads);
  // warmup
  if(num_warmup > 0){
    std::cout << "warmup: sending " << num_warmup << " queries ..." << std::endl;
    auto warmup_start = std::chrono::steady_clock::now();
    auto extra_time = std::chrono::nanoseconds(0);
    for (uint64_t i = 0; i < num_warmup; i++) {
      std::cout << "start sending query " << i << std::endl;
      auto start = std::chrono::steady_clock::now();
      uint64_t next_query_index = dataset.get_next_query_index();
      const data_type *query_emb = dataset.get_query(next_query_index);
      uint64_t query_id = client.query(query_emb);
            
      auto end = std::chrono::steady_clock::now();
      if (rate_control) {
        auto elapsed = end - start + extra_time;
        auto sleep_time = iteration_time - elapsed;
        start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(sleep_time);
        extra_time = std::chrono::steady_clock::now() - start - sleep_time;
      }
    }
        
    std::cout << "  all sent, waiting results ..." << std::endl;
    client.wait_results();
        
    std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - warmup_start);
    std::cout << "  system warmed up in " << elapsed.count() << " ms" << std::endl;
    dataset.reset();
  }
  
  uint32_t num_queries = dataset.get_num_queries();
  std::cout << "starting benchmark: sending " << dataset.get_num_queries() << " queries ..." << std::endl;
  std::unordered_map<uint64_t,uint32_t> query_id_to_index;
  auto extra_time = std::chrono::nanoseconds(0);
  for(uint32_t i=0;i<num_queries;i++){
    auto start = std::chrono::steady_clock::now();
    if (i % 200 == 0){
      std::cout << "  sent " << i << std::endl;
    }
    uint64_t next_query_index = dataset.get_next_query_index();
    const data_type *query_emb = dataset.get_query(next_query_index);
    uint64_t query_id = client.query(query_emb);
    query_id_to_index[query_id] = next_query_index;
            
    auto end = std::chrono::steady_clock::now();
    if (rate_control) {
      auto elapsed = end - start + extra_time;
      auto sleep_time = iteration_time - elapsed;
      start = std::chrono::steady_clock::now();
      std::this_thread::sleep_for(sleep_time);
      extra_time = std::chrono::steady_clock::now() - start - sleep_time;
    }
  }
  std::cout << "  all queries sent!" << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // wait until all results are received
  std::cout << "waiting all results to arrive ..." << std::endl;
  client.wait_results();
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // recall calculation
  std::cout << "starting calculation" <<std::endl;
  uint32_t query_result[num_queries * HEAD_INDEX_K];
  data_type query_data[num_queries * dataset.query_dim];
  std::vector<uint32_t> bad_queries;
  std::byte cluster_0;
  for (auto &[query_id, query_index] : query_id_to_index) {

    std::shared_ptr<GreedySearchQuery<data_type>> greedy_search_q =
      client.get_result(query_id);

    if (greedy_search_q->get_cluster_id() != cluster_0) throw std::runtime_error("query cluster " + std::to_string(static_cast<uint32_t>(greedy_search_q->get_cluster_id())));
    if (greedy_search_q->get_query_id() != query_index) throw std::runtime_error("query bad id " + std::to_string(greedy_search_q->get_query_id()));
    if (greedy_search_q->get_candidate_queue_size() != HEAD_INDEX_K) bad_queries.push_back(query_index);
    // std::cout << "total size " << greedy_search_q->get_candidate_queue_size() << std::endl;
    std::memcpy(query_result + query_index * HEAD_INDEX_K,
                greedy_search_q->get_candidate_queue_ptr(),
                greedy_search_q->get_candidate_queue_size() * sizeof(uint32_t));
    std::memcpy(query_data + query_index * dataset.query_dim,
                greedy_search_q->get_embedding_ptr(),
                sizeof(data_type) * dataset.query_dim);
  }
  std::cout << "done with copying data" << std::endl;
  std::cout << dataset.query_dim << " " << dataset.query_aligned_dim << std::endl;
  // verify the validity of queries
  for (uint64_t i = 0; i < num_queries * dataset.query_dim; i++) {
    if (query_data[i] != dataset.query_data[i])
      throw std::runtime_error("query data doesn't match byte " +
                               std::to_string(i));
  }

  
  double recall = diskann::calculate_recall(
      dataset.get_num_queries(), dataset.gt_ids, dataset.gt_dists,
					    dataset.gt_dim, query_result, HEAD_INDEX_K, HEAD_INDEX_K);

  std::cout << "recall is " << recall << std::endl;
  std::cout << "bad queries number: " << bad_queries.size() << std::endl;
  for (auto query_id : bad_queries) std::cout << query_id << std::endl;
}



int main(int argc, char **argv) {
  po::options_description desc("Options here mate");
  std::string query_file;
  std::string data_type;
  std::string gt_file;
  uint32_t send_rate;
  uint32_t batch_min_size;
  uint32_t batch_max_size;
  uint32_t batch_time_us;
  uint32_t num_result_threads;
  uint32_t num_warmup;
  
  desc.add_options()("help,h", "show help message")(
      "query_file,Q", po::value<std::string>(&query_file)->required(),
      "Path to the query file")(
      "data_type,T", po::value<std::string>(&data_type)->required(),
      "Data type of vectors, could be float, uint8, int8")(
      "gt_file,G", po::value<std::string>(&gt_file)->required(),
      "Ground truth file")("send_rate,R",
                           po::value<uint32_t>(&send_rate)->default_value(0),
                           "rate (in queries/second) at which to send queries")(
      "batch_min_size,B",
      po::value<uint32_t>(&batch_min_size)
          ->default_value(DEFAULT_BATCH_MIN_SIZE),
      "Min size of batch")("batch_max_size,X",
                           po::value<uint32_t>(&batch_max_size)
                               ->default_value(DEFAULT_BATCH_MAX_SIZE),
                           "max size of batch")(
      "batch_time_us,U",
      po::value<uint32_t>(&batch_time_us)->default_value(DEFAULT_BATCH_TIME_US),
      "maximum time to wait for the batch minimum size, in microseconds")(
      "num_result_threads,N",
      po::value<uint32_t>(&num_result_threads)
          ->default_value(DEFAULT_NUM_RESULT_THREADS),
      "number of threads for processing results")(
      "num_warmup,W",
      po::value<uint32_t>(&num_warmup)->default_value(DEFAULT_WARMUP),
						  "number of quries to wamrup system");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);
  if (data_type != "uint8" && data_type != "int8" && data_type != "float")
    throw std::invalid_argument("wrong data_type");



  if (data_type == "float") {
    benchmark<float>(query_file, gt_file, batch_min_size, batch_max_size, batch_time_us, num_result_threads, num_warmup, send_rate);
  } else if (data_type == "uint8_t") {
    benchmark<uint8_t>(query_file, gt_file, batch_min_size, batch_max_size, batch_time_us, num_result_threads, num_warmup, send_rate);
  } else if (data_type == "int8_t") {
    benchmark<int8_t>(query_file, gt_file, batch_min_size, batch_max_size, batch_time_us, num_result_threads, num_warmup, send_rate);
  }
  
}
