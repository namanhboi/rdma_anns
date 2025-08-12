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


namespace po = boost::program_options;
/**
   the last 2 parameters are purely used to test udl2
*/
template <typename data_type>
void benchmark(const std::string &query_file, const std::string &gt_file,
               uint32_t batch_min_size, uint32_t batch_max_size,
               uint32_t batch_time_us, uint32_t num_result_threads,
               uint32_t num_warmup, uint32_t send_rate, uint32_t K, uint32_t L,
               uint8_t cluster_id = 0, uint32_t start_node_id = 0, size_t num_queries_to_send = 1) {
  
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
      auto start = std::chrono::steady_clock::now();
      uint64_t next_query_index = dataset.get_next_query_index();
      const data_type *query_emb = dataset.get_query(next_query_index);
      // std::cout << query_emb << std::endl;
#ifdef TEST_UDL2
      uint64_t query_id =
        client.query(query_emb, K, L, cluster_id, start_node_id);
#else
      uint64_t query_id = client.query(query_emb, K, L);
#endif       
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
  
  uint32_t num_queries = std::min(dataset.get_num_queries(), num_queries_to_send);
  std::cout << "starting benchmark: sending " << num_queries_to_send << " queries ..." << std::endl;
  std::unordered_map<uint64_t,uint32_t> query_id_to_index;
  auto extra_time = std::chrono::nanoseconds(0);
  for(uint32_t i=0;i<num_queries;i++){
    auto start = std::chrono::steady_clock::now();
    if (i % 200 == 0){
      std::cout << "  sent " << i << std::endl;
    }
    uint64_t next_query_index = dataset.get_next_query_index();
    const data_type *query_emb = dataset.get_query(next_query_index);
    // std::cout << query_emb << std::endl;
#ifdef TEST_UDL2
    uint64_t query_id =
      client.query(query_emb, K, L, cluster_id, start_node_id);

#else
    uint64_t query_id = client.query(query_emb, K, L);
#endif       
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
  uint32_t *query_result = new uint32_t[num_queries * K];
  data_type *query_data = new data_type[num_queries * dataset.query_dim];
  std::vector<uint32_t> bad_queries;
  uint8_t cluster_0 = 0;
  for (auto &[query_id, query_index] : query_id_to_index) {
#ifdef TEST_UDL1
    std::shared_ptr<GreedySearchQuery<data_type>> greedy_search_q =
      client.get_result(query_id);
    // std::cout << query_id << " " << query_index << std::endl;

    if (greedy_search_q->get_cluster_id() != cluster_0)
      throw std::runtime_error("query cluster " +
                               std::to_string(static_cast<uint32_t>(
								    greedy_search_q->get_cluster_id())));
    if (greedy_search_q->get_query_id() != query_index)
      throw std::runtime_error("query bad id " +
                               std::to_string(greedy_search_q->get_query_id()) +
                               std::to_string(query_index));

    if (greedy_search_q->get_candidate_queue_size() != K)
      bad_queries.push_back(query_index);

    std::memcpy(query_result + query_index * K,
                greedy_search_q->get_candidate_queue_ptr(),
                greedy_search_q->get_candidate_queue_size() * sizeof(uint32_t));
    std::memcpy(query_data + query_index * dataset.query_dim,
                greedy_search_q->get_embedding_ptr(),
                sizeof(data_type) * dataset.query_dim);
#else
    ANNSearchResult result = client.get_result(query_id);
    // std::cout << query_id << " " << query_index << std::endl;
    std::memcpy(query_result + query_index * K,
                result.get_search_results_ptr(), K * sizeof(uint32_t));
#endif
  }
  std::cout << "done with copying data" << std::endl;
  std::cout << dataset.query_dim << " " << dataset.query_aligned_dim << std::endl;

#ifdef TEST_UDL1
  for (uint64_t i = 0; i < num_queries * dataset.query_dim; i++) {
    if (query_data[i] != dataset.query_data[i])
      throw std::runtime_error("query data doesn't match byte " +
                               std::to_string(i));
  }
#endif
  double recall =
      diskann::calculate_recall(num_queries, dataset.gt_ids, dataset.gt_dists,
                                dataset.gt_dim, query_result, K, K);

  std::cout << "recall is " << recall << std::endl;
  std::cout << "bad queries number: " << bad_queries.size() << std::endl;

  delete[] query_result;
  delete[] query_data;
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
  uint32_t K, L;
  uint8_t cluster_id;
  uint32_t start_node_id;
  size_t num_queries_to_send;

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
      "number of quries to wamrup system")(
      "K", po::value<uint32_t>(&K)->required(), "recall at")(
      "L", po::value<uint32_t>(&L)->required(), "candidate queue size")(
      "cluster_id", po::value<uint8_t>(&cluster_id)->default_value(0),
      "cluster id to send queries to test udl2")(
      "start_node_id", po::value<uint32_t>(&start_node_id)->default_value(0),
      "start node id to send queries to test udl2, should be in the cluster "
      "with id cluster_id")(
      "num_queries_to_send",
      po::value<size_t>(&num_queries_to_send)->default_value(1),
      "num queries to send, if exceeds num queries in query file then just "
      "send all queries");
  
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
    benchmark<float>(query_file, gt_file, batch_min_size, batch_max_size,
                     batch_time_us, num_result_threads, num_warmup, send_rate,
                     K, L, cluster_id, start_node_id, num_queries_to_send);
  } else if (data_type == "uint8_t") {
    benchmark<uint8_t>(query_file, gt_file, batch_min_size, batch_max_size,
                       batch_time_us, num_result_threads, num_warmup, send_rate,
                       K, L, cluster_id, start_node_id, num_queries_to_send);
  } else if (data_type == "int8_t") {
    benchmark<int8_t>(query_file, gt_file, batch_min_size, batch_max_size,
                      batch_time_us, num_result_threads, num_warmup, send_rate,
                      K, L, cluster_id, start_node_id, num_queries_to_send);
  }
  
}
