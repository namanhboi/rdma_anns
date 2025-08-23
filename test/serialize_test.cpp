#include <catch2/catch_template_test_macros.hpp>
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


std::vector<compute_query_t>
generate_random_compute_queries(uint32_t num_queries) {
  std::vector<compute_query_t> generated;
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator

  std::uniform_int_distribution<uint8_t> cluster_id_gen(0, 255);

  std::uniform_real_distribution<float> distance_gen(
						     0.0, std::numeric_limits<float>::max());
  std::uniform_int_distribution<uint32_t> node_id_gen(
						      0, std::numeric_limits<uint32_t>::max());
  for (uint32_t i = 0; i < num_queries; i++) {
    compute_query_t query(node_id_gen(gen), node_id_gen(gen), node_id_gen(gen),
                          distance_gen(gen), cluster_id_gen(gen),
                          cluster_id_gen(gen), node_id_gen(gen));
    
    generated.push_back(query);
  }
  return generated;
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
                            node_id_gen(gen), distance_gen(gen),
                            cluster_id_gen(gen), cluster_id_gen(gen),
                            node_id_gen(gen));
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
      REQUIRE(queries[i].client_node_id == generated[i].client_node_id);
      REQUIRE(queries[i].min_distance == generated[i].min_distance);
      REQUIRE(queries[i].cluster_receiver_id ==
              generated[i].cluster_receiver_id);
      REQUIRE(queries[i].cluster_sender_id == generated[i].cluster_sender_id);
    }
    batcher.reset();
    start_batch_index += batch_size;
  }
}


std::vector<std::shared_ptr<compute_result_t>>
generate_random_compute_results(uint32_t num_results, uint32_t R) {
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator  
  std::uniform_int_distribution<uint8_t> uint8_t_gen(0, 255);

  std::uniform_real_distribution<float> float_gen(
						     0.0, std::numeric_limits<float>::max());
  std::uniform_int_distribution<uint32_t> uint32_t_gen(
						      0, std::numeric_limits<uint32_t>::max());
  std::vector<std::shared_ptr<compute_result_t>> generated;
  for (uint32_t i = 0; i < num_results; i++) {
    std::vector<uint32_t> neighbors = generate_list_uint32_t(R);
    uint32_t num_neighbors = neighbors.size();
    std::shared_ptr<uint32_t[]> nbr_ids(new uint32_t[num_neighbors]);
    std::memcpy(nbr_ids.get(), neighbors.data(),
                sizeof(uint32_t) * num_neighbors);
    std::shared_ptr<float[]> nbr_distances(new float[num_neighbors]);
    for (auto j = 0; j < num_neighbors; j++) {
      nbr_distances.get()[j] = float_gen(gen);
    }
    uint32_t query_id = uint32_t_gen(gen);
    uint32_t node_id = uint32_t_gen(gen);
    uint32_t client_node_id = uint32_t_gen(gen);
    float expanded_dist = float_gen(gen);
    uint32_t receiver_thread_id = uint32_t_gen(gen);
    uint8_t cluster_sender_id = uint8_t_gen(gen);
    uint8_t cluster_receiver_id = uint8_t_gen(gen);
    generated.emplace_back(std::make_shared<compute_result_t>(
        num_neighbors, nbr_ids, nbr_distances, query_id, node_id,
        client_node_id, expanded_dist, receiver_thread_id, cluster_sender_id,
							      cluster_receiver_id));
  }
  return generated;
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

  std::uniform_int_distribution<uint8_t> uint8_t_gen(0, 255);

  std::uniform_real_distribution<float> float_gen(
						     0.0, std::numeric_limits<float>::max());
  std::uniform_int_distribution<uint32_t> uint32_t_gen(
						      0, std::numeric_limits<uint32_t>::max());
  ComputeResultBatcher batcher(max_batch_size + 1);
  while (start_batch_index < num_queries) {
    uint32_t num_queries_left = num_queries - start_batch_index;
    uint32_t rand_batch_size = batch_size_gen(gen);
    uint32_t batch_size = std::min(num_queries_left, rand_batch_size);

    std::vector<std::shared_ptr<compute_result_t>> generated =
      generate_random_compute_results(batch_size, R);
    
    for (uint32_t i = 0; i < batch_size; i++) {
      batcher.push(generated[i]);
    }

    std::shared_ptr<uint8_t[]> buffer(
				      new uint8_t[batcher.get_serialize_size()]);
    batcher.write_serialize(buffer.get());

    ComputeResultBatchManager manager(buffer.get(),
                                      batcher.get_serialize_size());

    const auto &results  = manager.get_results();

    REQUIRE(results.size() == batch_size);

    for (uint32_t i = 0; i < results.size(); i++) {
      const auto &result = results[i];
	REQUIRE(result->get_num_neighbors() == generated[i]->num_neighbors);
        REQUIRE(std::memcmp(result->get_nbr_ids(), generated[i]->nbr_ids.get(),
                            sizeof(uint32_t) * generated[i]->num_neighbors) == 0);
        REQUIRE(std::memcmp(result->get_nbr_distances(),
                            generated[i]->nbr_distances.get(),
                            sizeof(float) * generated[i]->num_neighbors) == 0);
        REQUIRE(result->get_query_id() == generated[i]->query_id);
        REQUIRE(result->get_node_id() == generated[i]->node_id);
        REQUIRE(result->get_client_node_id() == generated[i]->client_node_id);
        REQUIRE(result->get_expanded_dist() == generated[i]->expanded_dist);
        REQUIRE(result->get_receiver_thread_id() ==
                generated[i]->receiver_thread_id);
        REQUIRE(result->get_cluster_sender_id() ==
                generated[i]->cluster_sender_id);
        REQUIRE(result->get_cluster_receiver_id() ==
                generated[i]->cluster_receiver_id);
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



template <typename data_type>
std::vector<std::shared_ptr<EmbeddingQuery<data_type>>>
generate_random_embedding_queries(uint32_t num_queries, uint32_t dim, uint32_t K, uint32_t L) {
  std::vector<std::shared_ptr<EmbeddingQuery<data_type>>> res;

  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<uint32_t> node_id_gen(
						      0, std::numeric_limits<uint32_t>::max());

  std::uniform_int_distribution<uint8_t> uint8_t_gen(
						     std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());


  EmbeddingQueryBatcher<data_type> batcher(dim, num_queries + 1);

  std::shared_ptr<data_type> emb_data(new data_type[dim * num_queries]);
  std::vector<uint8_t> tmp(num_queries * dim * sizeof(data_type));
  std::generate(tmp.begin(), tmp.end(), [&]() { return uint8_t_gen(gen); });
  std::memcpy(emb_data.get(), tmp.data(), tmp.size());

  for (uint32_t i = 0; i < num_queries; i++) {
    query_t<data_type> query(node_id_gen(gen), node_id_gen(gen),
                             emb_data.get() + i * dim, dim, K, L);
    batcher.add_query(query);
  }
  std::shared_ptr<uint8_t[]> buffer(new uint8_t[batcher.get_serialize_size()]);
  batcher.write_serialize(buffer.get());

  EmbeddingQueryBatchManager<data_type> manager(buffer.get(),
                                                batcher.get_serialize_size());
  res = manager.get_queries();
  return res;
}

template <typename data_type>
std::vector<greedy_query_t<data_type>>
generate_random_greedy_queries(uint32_t num_queries, uint32_t dim, uint32_t K,
                        uint32_t L) {
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<uint8_t> cluster_id_gen(
							std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

  std::vector<greedy_query_t<data_type>> res;
  std::vector<std::shared_ptr<EmbeddingQuery<data_type>>> emb_queries =
    generate_random_embedding_queries<data_type>(num_queries, dim, K, L);


  for (uint32_t i = 0; i < num_queries; i++) {
    greedy_query_t<data_type> query(cluster_id_gen(gen),
                                    generate_list_uint32_t(L), emb_queries[i]);
    res.push_back(query);
  }
  return res;
}

TEMPLATE_TEST_CASE("testing global search message serialization", "[template]", float, uint8_t, int8_t) {
  uint32_t num_batches = 10'000;
  uint32_t min_batch_size = 0;
  uint32_t max_batch_size = 6;

  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator
  std::uniform_int_distribution<> batch_size_gen(min_batch_size,
                                                 max_batch_size);
  uint32_t dim = 128;
  uint32_t K = 10;
  uint32_t R = 32;
  uint32_t L = 20;
  
  GlobalSearchMessageBatcher<TestType> batcher(dim);
  for (uint32_t i = 0; i < num_batches; i++) {
    uint32_t emb_query_batch_size = batch_size_gen(gen);
    std::vector<std::shared_ptr<EmbeddingQuery<TestType>>> emb_queries =
      generate_random_embedding_queries<TestType>(emb_query_batch_size, dim,
						  K, L);
    uint32_t greedy_query_batch_size = batch_size_gen(gen);
    std::vector<greedy_query_t<TestType>> greedy_queries =
      generate_random_greedy_queries<TestType>(greedy_query_batch_size, dim,
					       K, L);

    uint32_t compute_result_batch_size = batch_size_gen(gen);
    std::vector<std::shared_ptr<compute_result_t>> compute_results =
      generate_random_compute_results(compute_result_batch_size, R);
        
    
    uint32_t compute_query_batch_size = batch_size_gen(gen);
    std::vector<compute_query_t> compute_queries =
      generate_random_compute_queries(compute_query_batch_size);

    for (const std::shared_ptr<EmbeddingQuery<TestType>> &emb_query : emb_queries) {
      batcher.push_embedding_query(emb_query);
    }
    for (const auto &search_q : greedy_queries) {
      batcher.push_search_query(search_q);
    }
    for (const auto &compute_res : compute_results) {
      batcher.push_compute_result(compute_res);
    }
    for (const auto &compute_query : compute_queries) {
      batcher.push_compute_query(compute_query);
    }

    std::shared_ptr<uint8_t[]> buffer(
				      new uint8_t[batcher.get_serialize_size()]);
    

    batcher.write_serialize(buffer.get());


    GlobalSearchMessageBatchManager<TestType> manager(
						      buffer.get(), batcher.get_serialize_size(), dim);
    const auto &managed_emb_queries = manager.get_embedding_queries();

    REQUIRE(managed_emb_queries.size() == emb_query_batch_size);
    for (uint32_t i = 0; i < managed_emb_queries.size(); i++) {
      const std::shared_ptr<EmbeddingQuery<TestType>> &query = managed_emb_queries[i];
      REQUIRE(query->get_query_id() == emb_queries[i]->get_query_id());
      REQUIRE(query->get_client_node_id() ==
              emb_queries[i]->get_client_node_id());
      REQUIRE(query->get_K() == emb_queries[i]->get_K());
      REQUIRE(query->get_L() == emb_queries[i]->get_L());
      REQUIRE(query->get_dim() == emb_queries[i]->get_dim());

      REQUIRE(std::memcmp(query->get_embedding_ptr(),
                          emb_queries[i]->get_embedding_ptr(),
                          emb_queries[i]->get_dim() * sizeof(TestType)) == 0);
    }
    

    const auto &managed_search_queries = manager.get_greedy_search_queries();
    REQUIRE(managed_search_queries.size() == greedy_query_batch_size);
    for (uint32_t i = 0; i < managed_search_queries.size(); i++) {
      const std::shared_ptr<GreedySearchQuery<TestType>> &query =
	managed_search_queries[i];


      REQUIRE(query->get_cluster_id() == greedy_queries[i].cluster_id);
      std::vector<uint32_t> greedy_query_cand_q(
						query->get_candidate_queue_ptr(),
						query->get_candidate_queue_ptr() +
						query->get_candidate_queue_size());

      REQUIRE(greedy_query_cand_q == greedy_queries[i].candidate_queue);
      const std::shared_ptr<EmbeddingQuery<TestType>> greedy_emb =
	greedy_queries[i].query;

      REQUIRE(query->get_query_id() ==
	      greedy_emb->get_query_id());

      REQUIRE(query->get_client_node_id() ==
	      greedy_emb->get_client_node_id());
      REQUIRE(query->get_K() == greedy_emb->get_K());
      REQUIRE(query->get_L() == greedy_emb->get_L());
      REQUIRE(query->get_dim() == greedy_emb->get_dim());

      REQUIRE(std::memcmp(query->get_embedding_ptr(),
			  greedy_emb->get_embedding_ptr(),
			  greedy_emb->get_dim() * sizeof(TestType)) == 0);
    }

    const auto &managed_compute_results = manager.get_compute_results();
    REQUIRE(managed_compute_results.size() == compute_results.size());

    for (uint32_t i = 0; i < managed_compute_results.size(); i++) {
      const std::shared_ptr<ComputeResult> &result = managed_compute_results[i];

      REQUIRE(result->get_num_neighbors() == compute_results[i]->num_neighbors);
      REQUIRE(std::memcmp(
                  result->get_nbr_ids(), compute_results[i]->nbr_ids.get(),
			  sizeof(uint32_t) * compute_results[i]->num_neighbors) == 0);
      REQUIRE(std::memcmp(result->get_nbr_distances(),
                          compute_results[i]->nbr_distances.get(),
                          sizeof(float) * compute_results[i]->num_neighbors) ==
              0);
      REQUIRE(result->get_query_id() == compute_results[i]->query_id);
      REQUIRE(result->get_node_id() == compute_results[i]->node_id);
      REQUIRE(result->get_client_node_id() ==
              compute_results[i]->client_node_id);
      REQUIRE(result->get_receiver_thread_id() ==
              compute_results[i]->receiver_thread_id);
      REQUIRE(result->get_expanded_dist() == compute_results[i]->expanded_dist);
      REQUIRE(result->get_cluster_sender_id() ==
              compute_results[i]->cluster_sender_id);
      REQUIRE(result->get_cluster_receiver_id() ==
              compute_results[i]->cluster_receiver_id);
    }
    const auto &managed_compute_queries = manager.get_compute_queries();
    REQUIRE(managed_compute_queries.size() == compute_queries.size());
    for (uint32_t i = 0; i < managed_compute_queries.size(); i++) {
      const auto &query = managed_compute_queries[i];
      REQUIRE(query.node_id == compute_queries[i].node_id);
      REQUIRE(query.query_id == compute_queries[i].query_id);
      REQUIRE(query.client_node_id == compute_queries[i].client_node_id);
      REQUIRE(query.min_distance == compute_queries[i].min_distance);
      REQUIRE(query.cluster_receiver_id ==
              compute_queries[i].cluster_receiver_id);
      REQUIRE(query.cluster_sender_id == compute_queries[i].cluster_sender_id);
    }

    batcher.reset();
  }
}

