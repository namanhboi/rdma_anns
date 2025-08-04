#include <cascade/service_client_api.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../benchmark/benchmark_dataset.hpp"
#include "../src/serialize_utils.hpp"
#include <limits>
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

    // if (blob17->bytes == nullptr)
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

std::vector<uint32_t> generate_list_uint32_t(uint32_t L, bool exact_size = false) {
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<uint32_t> node_id_gen(
						      0, std::numeric_limits<uint32_t>::max());
  std::uniform_int_distribution<uint32_t> num_cand_gen(0, L);

  uint32_t num_cand = exact_size ? L : num_cand_gen(gen) ;
  // std::cout << num_cand << std::endl;
  std::set<uint32_t> ids;
  while (ids.size() < num_cand) {
    ids.insert(node_id_gen(gen));
  }
  std::vector<uint32_t> canq(ids.begin(), ids.end());
  return canq;
}


TEST_CASE("Testing GreedySearchQuery serialization and deserialization on sift dataset") {
  BenchmarkDataset<float> sift = get_sift_dataset();

  uint32_t default_client_id = 0;
  uint32_t K = 10;
  uint32_t L = 20;

  uint32_t min_batch_size = 1;
  uint32_t max_batch_size = 10;

  size_t start_batch_index = 0;
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> batch_size_gen(min_batch_size, max_batch_size);

  std::uniform_int_distribution<uint8_t> cluster_id_gen(0, 255);

  
  

  GreedySearchQueryBatcher<float> greedy_batcher(sift.get_dim(), max_batch_size + 1);
  EmbeddingQueryBatcher<float> emb_batcher(sift.get_dim(), max_batch_size + 1);  
  while (start_batch_index < sift.get_num_queries()) {
    uint32_t num_queries_left = sift.get_num_queries() - start_batch_index;
    uint32_t rand_batch_size = batch_size_gen(gen);
    uint32_t batch_size = std::min(num_queries_left, rand_batch_size);

    std::vector<uint8_t> greedy_cluster_ids;
    std::vector<std::vector<uint32_t>> greedy_cand_queues;
    
    
    for(uint32_t i = 0; i < batch_size; i++) {
      const float *query_emb = sift.get_query(start_batch_index + i);
      emb_batcher.add_query(start_batch_index + i, default_client_id, query_emb,
                            sift.get_dim(), K, L);
    }

    std::shared_ptr<uint8_t[]> emb_buffer(new uint8_t[emb_batcher.get_serialize_size()]);
    emb_batcher.write_serialize(emb_buffer.get());

    EmbeddingQueryBatchManager<float> manager(emb_buffer.get(),
                                              emb_batcher.get_serialize_size());

    const auto emb_queries = manager.get_queries();
    REQUIRE(emb_queries.size() == batch_size);
    for (uint32_t i = 0; i < emb_queries.size(); i++) {
      std::vector<uint32_t> greedy_cand_q = generate_list_uint32_t(L);

      uint8_t greedy_cluster_id = cluster_id_gen(gen);
      greedy_cluster_ids.emplace_back(greedy_cluster_id);
      greedy_cand_queues.push_back(greedy_cand_q);
      greedy_query_t<float> greedy_q(greedy_cluster_id, greedy_cand_q,
                                     emb_queries[i]);
      greedy_batcher.add_query(greedy_q);
    }
    std::shared_ptr<uint8_t[]> greedy_buffer(
					     new uint8_t[greedy_batcher.get_serialize_size()]);

    greedy_batcher.write_serialize(greedy_buffer.get());

    GreedySearchQueryBatchManager<float> greedy_manager(
							greedy_buffer.get(), greedy_batcher.get_serialize_size());

    const auto &greedy_queries = greedy_manager.get_queries();
    for (uint32_t i = 0; i < greedy_queries.size(); i++) {
      REQUIRE(greedy_queries[i]->get_cluster_id() == greedy_cluster_ids[i]);
      std::vector<uint32_t> greedy_query_cand_q(
          greedy_queries[i]->get_candidate_queue_ptr(),
          greedy_queries[i]->get_candidate_queue_ptr() +
          greedy_queries[i]->get_candidate_queue_size());
      REQUIRE(greedy_query_cand_q == greedy_cand_queues[i]);

      REQUIRE(greedy_queries[i]->get_query_id() == start_batch_index + i);
      REQUIRE(greedy_queries[i]->get_client_node_id() == default_client_id);
      REQUIRE(greedy_queries[i]->get_K() == K);
      REQUIRE(greedy_queries[i]->get_L() == L);
      REQUIRE(greedy_queries[i]->get_dim() == sift.get_dim());
      REQUIRE(std::memcmp(greedy_queries[i]->get_embedding_ptr(),
                          sift.get_query(start_batch_index + i),
                          sift.get_query_size()) == 0);
    }
    
    emb_batcher.reset();
    greedy_batcher.reset();
    start_batch_index += batch_size;
  }
}

TEST_CASE("Testing serialization of compute queries") {
  uint32_t num_queries = 10'000;

  uint32_t min_batch_size = 1;
  uint32_t max_batch_size = 10;

  size_t start_batch_index = 0;
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> batch_size_gen(min_batch_size, max_batch_size);

  std::uniform_int_distribution<uint8_t> cluster_id_gen(0, 255);

  std::uniform_real_distribution<float> distance_gen(
						     0.0, std::numeric_limits<float>::max());
  std::uniform_int_distribution<uint32_t> node_id_gen(
						      0, std::numeric_limits<uint32_t>::max());

  ComputeQueryBatcher batcher(max_batch_size + 1);
  while (start_batch_index < num_queries) {
    uint32_t num_queries_left = num_queries - start_batch_index;
    uint32_t rand_batch_size = batch_size_gen(gen);
    uint32_t batch_size = std::min(num_queries_left, rand_batch_size);


    std::vector<compute_query_t> generated;
    
    for (uint32_t i = 0; i < batch_size; i++) {
      compute_query_t query(node_id_gen(gen), node_id_gen(gen),
                            distance_gen(gen), cluster_id_gen(gen),
                            cluster_id_gen(gen));
      generated.push_back(query);
      batcher.push(query);
    }

    std::shared_ptr<uint8_t[]> buffer(
				      new uint8_t[batcher.get_serialize_size()]);
    batcher.write_serialize(buffer.get());
    
    ComputeQueryBatchManager manager(buffer.get(),
                                     batcher.get_serialize_size());
    const auto &queries = manager.get_queries();
    REQUIRE(queries.size() == batch_size);
    for (uint32_t i = 0; i < queries.size(); i++) {
      REQUIRE(queries[i].node_id == generated[i].node_id);
      REQUIRE(queries[i].query_id == generated[i].query_id);
      REQUIRE(queries[i].min_distance == generated[i].min_distance);
      REQUIRE(queries[i].cluster_receiver_id ==
              generated[i].cluster_receiver_id);
      REQUIRE(queries[i].cluster_sender_id == generated[i].cluster_sender_id);
    }
    batcher.reset();
    start_batch_index += batch_size;
  }
}

TEST_CASE("Testing serialization of compute result") {
  uint32_t num_queries = 10'000;
  uint32_t R = 32;
  uint32_t min_batch_size = 1;
  uint32_t max_batch_size = 10;

  size_t start_batch_index = 0;
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> batch_size_gen(min_batch_size, max_batch_size);

  std::uniform_int_distribution<uint8_t> cluster_id_gen(0, 255);

  std::uniform_real_distribution<float> distance_gen(
						     0.0, std::numeric_limits<float>::max());
  std::uniform_int_distribution<uint32_t> node_id_gen(
						      0, std::numeric_limits<uint32_t>::max());


  ComputeResultBatcher batcher(max_batch_size + 1);
  while (start_batch_index < num_queries) {
    uint32_t num_queries_left = num_queries - start_batch_index;
    uint32_t rand_batch_size = batch_size_gen(gen);
    uint32_t batch_size = std::min(num_queries_left, rand_batch_size);

    std::vector<compute_result_t> generated;
    for (uint32_t i = 0; i < batch_size; i++) {
      std::vector<uint32_t> neighbors = generate_list_uint32_t(R);
      uint32_t *nbrs = reinterpret_cast<uint32_t *>(
						    malloc(sizeof(uint32_t) * (neighbors.size() + 1)));
      nbrs[0] = static_cast<uint32_t>(neighbors.size());
      std::memcpy(nbrs + 1, neighbors.data(),
                  sizeof(uint32_t) * neighbors.size());
      std::shared_ptr<const uint32_t> nbr_ptr(nbrs, std::free);
      compute_result_t res(cluster_id_gen(gen), cluster_id_gen(gen),
                           {node_id_gen(gen), distance_gen(gen)},
                           node_id_gen(gen), neighbors.size(), nbr_ptr);
      generated.push_back(res);
      batcher.push(res);
    }

    std::shared_ptr<uint8_t[]> buffer(
				      new uint8_t[batcher.get_serialize_size()]);
    batcher.write_serialize(buffer.get());

    ComputeResultBatchManager manager(buffer.get(),
                                      batcher.get_serialize_size());

    const auto &results  = manager.get_results();

    REQUIRE(results.size() == batch_size);

    for (uint32_t i = 0; i < results.size(); i++) {
      const auto & result = results[i];
      REQUIRE(result.get_cluster_sender_id() == generated[i].cluster_sender_id);
      REQUIRE(result.get_cluster_receiver_id() ==
              generated[i].cluster_receiver_id);
      REQUIRE(result.get_node().id == generated[i].node.id);
      REQUIRE(result.get_node().distance == generated[i].node.distance);
      REQUIRE(result.get_query_id() == generated[i].query_id);
      REQUIRE(result.get_num_neighbors() == generated[i].num_neighbors);
      REQUIRE(std::memcmp(result.get_neighbors_ptr(),
                          generated[i].nbr_ptr.get() + 1,
                          sizeof(uint32_t) * result.get_num_neighbors()) == 0);
    }

    batcher.reset();
    start_batch_index += batch_size;
  }

}

TEST_CASE("Serialization of ann search result") {
  uint32_t num_queries = 10'000;
  uint32_t K = 10;
  uint32_t L = 20;
  uint32_t min_batch_size = 1;
  uint32_t max_batch_size = 10;

  size_t start_batch_index = 0;
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> batch_size_gen(min_batch_size, max_batch_size);

  std::uniform_int_distribution<uint8_t> cluster_id_gen(0, 255);

  std::uniform_real_distribution<float> distance_gen(
						     0.0, std::numeric_limits<float>::max());
  std::uniform_int_distribution<uint32_t> node_id_gen(
						      0, std::numeric_limits<uint32_t>::max());

  ANNSearchResultBatcher batcher(max_batch_size);
  while (start_batch_index < num_queries) {
    uint32_t num_queries_left = num_queries - start_batch_index;
    uint32_t rand_batch_size = batch_size_gen(gen);
    uint32_t batch_size = std::min(num_queries_left, rand_batch_size);

    std::vector<ann_search_result_t> generated;
    for (uint32_t i = 0; i < batch_size; i++) {
      std::vector<uint32_t> rand_search_results =
        generate_list_uint32_t(K, true);
      REQUIRE(rand_search_results.size() == K);
      std::shared_ptr<uint32_t[]> tmp(new uint32_t[rand_search_results.size()]);
      std::memcpy(tmp.get(), rand_search_results.data(),
                  sizeof(uint32_t) * rand_search_results.size());
      ann_search_result_t res = {
        node_id_gen(gen),
        node_id_gen(gen),
        K,
        L,
        tmp,
        cluster_id_gen(gen)
      };

      generated.push_back(res);
      batcher.push(res);
    }

    std::shared_ptr<uint8_t[]> buffer(
				      new uint8_t[batcher.get_serialize_size()]);
    batcher.write_serialize(buffer.get());

    ANNSearchResultBatchManager manager(buffer.get(), batcher.get_serialize_size());
    const auto &results = manager.get_results();
    REQUIRE(results.size() == batch_size);
    for (uint32_t i = 0; i < results.size(); i++) {
      const auto &result = results[i];
      REQUIRE(result.get_query_id() == generated[i].query_id);
      REQUIRE(result.get_client_id() == generated[i].client_id);
      REQUIRE(result.get_K() == generated[i].K);
      REQUIRE(result.get_L() == generated[i].L);
      REQUIRE(result.get_cluster_id() == generated[i].cluster_id);
      REQUIRE(std::memcmp(result.get_search_results_ptr(),
                          generated[i].search_result.get(),
                          sizeof(uint32_t) * K) == 0);
    }
    batcher.reset();
    start_batch_index += batch_size;
  }

}
