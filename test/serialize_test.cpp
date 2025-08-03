#include <cascade/service_client_api.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../benchmark/benchmark_dataset.hpp"
#include "../src/serialize_utils.hpp"
#include <random>
// what nees to be tested?
// query_t -> embeddingquerybatcher ->
// EmbeddingQueryBatchManager->EmbeddingQuery


BenchmarkDataset<float> get_sift_dataset() {
  const std::string query_file = "/home/nam/workspace/rdma_anns/extern/DiskANN/"
                                 "build/data/sift/sift_query.fbin";
  const std::string gt_file = "/home/nam/workspace/rdma_anns/extern/DiskANN/"
                              "build/data/sift/sift_query_learn_gt100";
  BenchmarkDataset<float> dataset(query_file, gt_file);
  return dataset;
}

TEST_CASE("Testing EmbeddingQuery serialization and deserialization on sift dataset") {
  BenchmarkDataset<float> sift = get_sift_dataset();

  uint32_t default_client_id = 0;
  uint32_t K = 10;
  uint32_t L = 20;

  uint32_t min_batch_size = 1;
  uint32_t max_batch_size = 10;

  size_t start_batch_index = 0;
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> distr(min_batch_size, max_batch_size);

  EmbeddingQueryBatcher<float> batcher(sift.get_dim(), max_batch_size + 1);
  while (start_batch_index < sift.get_num_queries()) {
    uint32_t num_queries_left = sift.get_num_queries() - start_batch_index;
    uint32_t rand_batch_size = distr(gen);
    uint32_t batch_size = std::min(num_queries_left, rand_batch_size);

    for(uint32_t i = 0; i < batch_size; i++) {
      const float *query_emb = sift.get_query(start_batch_index + i);
      batcher.add_query(start_batch_index + i, default_client_id, query_emb, sift.get_dim(), K ,L);
    }
    std::shared_ptr<uint8_t[]> tmp(new uint8_t[batcher.get_serialize_size()]);
    batcher.write_serialize(tmp.get());

    // std::shared_ptr<derecho::cascade::Blob> blob = batcher.get_blob();
    // std::cerr << blob->size << std::endl;

    // if (blob->bytes == nullptr)
      // std::cerr << "nullptr" << std::endl;
    EmbeddingQueryBatchManager<float> manager(tmp.get(),
                                              batcher.get_serialize_size());
    
    const auto &queries = manager.get_queries();
    for (uint32_t i = 0; i < queries.size(); i++) {
      REQUIRE(queries[i]->get_query_id() == start_batch_index + i);
      REQUIRE(queries[i]->get_client_node_id() == default_client_id);
      REQUIRE(queries[i]->get_K() == K);
      REQUIRE(queries[i]->get_L() == L);
      REQUIRE(queries[i]->get_dim() == sift.get_dim());

      REQUIRE(std::memcmp(queries[i]->get_embedding_ptr(),
                          sift.get_query(start_batch_index + i),
                          sift.get_query_size()) == 0);
    }
    
    batcher.reset();
    start_batch_index += batch_size;
  }
}




