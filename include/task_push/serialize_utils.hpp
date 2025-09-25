#pragma once
#include <immintrin.h> // needed to include this to make sure that the code compiles since in DiskANN/include/utils.h it uses this library.
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <cascade/service_client_api.hpp>
#include "neighbor.h"



/** free const pointers, will be used as deleter for std::shared_ptr when you
 * get() a blob*/

void free_const(const void* ptr) noexcept;

/**
   Explanation about the naming conventions in this file:
   - greedy_query_t, query_t,... : this is the intermedidate representation of
     stuff to be sent to the batcher to serialize.
     - CamelCaseClasses: these are the data structures that refer to the shared
       ptr buffer that the batch manager first manages. This is to reduce data
       copying + use RAII to clean up data neatly
       - BatchManager: de serializes the data recived from the handler to create the
	 appropriate CamelCaseClasses that all refer to the pointer to the data that
	 was recieved from the handler.
	 - Batcher: serializes the data that it manages so that this data can be moved
	   into a blob and sent to the appropriate object pool to trigger whatever udl
 */

/*
 * EmbeddingQuery encapsulates a single embedding query that is part of a batch.
 * Operations are performed on demand and directly from the buffer of the whole
 * batch.
 *
 *
 */
template<typename data_type>
class EmbeddingQuery {
public:
  std::shared_ptr<uint8_t[]> buffer; // this ptr is the start of a batch of queries, not just this one
  uint64_t buffer_size;
  uint32_t query_id;
  uint32_t client_node_id, embeddings_position, embeddings_size;
  uint32_t dim;
  uint32_t K;
  uint32_t L;

  // this constructor is used to easily test udl2
  EmbeddingQuery(const uint8_t *emb_buffer, uint64_t emb_buffer_size,
                 uint32_t query_id, uint32_t client_node_id, uint32_t K,
                 uint32_t L, uint32_t emb_dim) {
    std::shared_ptr<uint8_t[]> tmp(new uint8_t[emb_buffer_size]);
    std::memcpy(tmp.get(), emb_buffer, emb_buffer_size);
    this->buffer = std::move(tmp);

    this->buffer_size = emb_buffer_size;
    this->query_id = query_id;
    this->client_node_id = client_node_id;
    this->K = K;
    this->L = L;
    this->dim = emb_dim;
    this->embeddings_position = 0;
  }
  

  EmbeddingQuery(std::shared_ptr<uint8_t[]> buffer, uint64_t buffer_size,
                 uint32_t metadata_position, uint32_t embeddings_position, uint32_t emb_dim) {
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    const uint32_t *metadata = reinterpret_cast<const uint32_t *>(
								  buffer.get() + metadata_position);
    this->query_id = metadata[0];
    this->client_node_id = metadata[1];
    this->K = metadata[2];
    this->L = metadata[3];
    this->embeddings_position = embeddings_position;

    this->dim = emb_dim;
  }
  uint32_t get_query_id() { return this->query_id; }

  uint32_t get_K() { return this->K; }

  uint32_t get_L() {
    return this->L;
  }
  uint32_t get_client_node_id() { return this->client_node_id; }

  const data_type *get_embedding_ptr() {
    if (this->embeddings_position >= this->buffer_size) throw std::runtime_error("embeddings_position " + std::to_string(this->embeddings_position) +  " > " + std::to_string(this->buffer_size));
    return reinterpret_cast<const data_type*>(this->buffer.get() + this->embeddings_position);
  }

  uint64_t get_dim() { return this->dim; }

  uint64_t get_aligned_dim() { return ROUND_UP(dim, 8); }
  
};


template <typename data_type>
struct query_t {
  uint32_t query_id;
  uint32_t client_node_id;
  const data_type *query_emb;
  uint32_t dimension;
  uint32_t K;
  uint32_t L;


  query_t(const std::shared_ptr<EmbeddingQuery<data_type>> &query_ptr) {
    this->query_id = query_ptr->get_query_id();
    this->client_node_id = query_ptr->get_client_node_id();
    this->query_emb = query_ptr->get_embedding_ptr();
    this->dimension = query_ptr->get_dim();
    this->K = query_ptr->get_K();
    this->L = query_ptr->get_L();
  }

  query_t(uint32_t query_id, uint32_t client_node_id,
          const data_type *query_emb, uint32_t dimension, uint32_t K,
          uint32_t L)
  : query_id(query_id), client_node_id(client_node_id),
  query_emb(query_emb), dimension(dimension), K(K), L(L) {}

  static uint32_t get_metadata_size() {
    return sizeof(query_id) + sizeof(client_node_id) + sizeof(K) + sizeof(L);
  }
  void write_metadata(uint8_t *buffer) {
    uint32_t offset = 0;
    std::memcpy(buffer + offset, &query_id, sizeof(query_id));
    offset += sizeof(query_id);
    std::memcpy(buffer + offset, &client_node_id, sizeof(client_node_id));
    offset += sizeof(client_node_id);
    std::memcpy(buffer + offset, &K, sizeof(K));
    offset += sizeof(K);
    std::memcpy(buffer + offset, &L, sizeof(L));
  }
  uint32_t get_query_emb_size() { return sizeof(data_type) * dimension; }
  uint32_t get_K() { return K; }
  uint32_t get_L() { return L; }
  uint32_t get_query_id() {
    return query_id;
  }
  void write_emb(uint8_t *buffer) {
    std::memcpy(buffer, query_emb, sizeof(data_type) * dimension);
  }    
};

template <typename data_type> class EmbeddingQueryBatcher {
public:
  std::vector<query_t<data_type>> queries;
  uint64_t query_emb_size;
  std::shared_ptr<derecho::cascade::Blob> blob;
  uint32_t dimension;

  void write_header(uint8_t *buffer) {
    uint32_t num_queries = queries.size();
    std::memcpy(buffer, &num_queries, sizeof(num_queries));
    std::memcpy(buffer + sizeof(num_queries), &dimension, sizeof(dimension));
  }

  static uint32_t get_header_size() { return sizeof(uint32_t) * 2; }

  EmbeddingQueryBatcher() {
    this->query_emb_size = 0;
  }
  EmbeddingQueryBatcher(uint32_t emb_dim, uint64_t size_hint) {
    this->dimension = emb_dim;
    this->query_emb_size = emb_dim * sizeof(data_type);
    queries.reserve(size_hint);
  }
  
  void add_query(query_t<data_type> query) {
    queries.emplace_back(query);
  }

  void add_query(uint32_t query_id, uint32_t client_node_id,
                 const data_type *query_data, uint32_t dimension, uint32_t K, uint32_t L) {
    queries.emplace_back(query_id, client_node_id, query_data, dimension, K, L);
  }

  size_t get_serialize_size() const {
    size_t total_size =
      EmbeddingQueryBatcher<data_type>::get_header_size() +
      query_t<data_type>::get_metadata_size() * queries.size() +
      this->query_emb_size * queries.size();
    return total_size;
  }


  void write_serialize(uint8_t *buffer) {
    uint32_t metadata_position = EmbeddingQueryBatcher<data_type>::get_header_size();
    uint32_t embedding_position = metadata_position + 
				  query_t<data_type>::get_metadata_size() *
				  this->queries.size();
    write_header(buffer);
    for (query_t<data_type>& query : queries) {
      query.write_metadata(buffer + metadata_position);
      query.write_emb(buffer + embedding_position);

      metadata_position += query_t<data_type>::get_metadata_size();
      embedding_position += query.get_query_emb_size();
    }    

  }

  
  void serialize() {
    size_t total_size = get_serialize_size();

    this->blob = std::make_shared<derecho::cascade::Blob>(
							  [&](uint8_t *buffer, const std::size_t size) {
							    write_serialize(buffer);
							    return size;
							  }, total_size);
  }
  std::shared_ptr<derecho::cascade::Blob> get_blob() { return blob; }

  size_t get_count() {
    return queries.size();
  }

  void reset() {
    blob.reset();
    queries.clear();
  }

  bool is_empty() {
    return queries.empty();
  }    
};



/*
  EmbeddingQueryBatchManager perform operations on the whole embedding query
  batch received from the client or UDL1.
  Such operations include getting all EmbeddingQuery that are in the batch, or
  getting all the embeddings for processing all in batch.


 Head index search udl recieves a blob that it then deserialize into this object
 to distribute the different queries in this batch to the worker threads.

data layout for a batch of queries looks like the following:
header                          || metadata (for all the queries lined up
contiguously) || embeddings [contiguou] num_queries, embeddings_position
query_id  client_node_id emb_position, emb_size, .. uint32_t     uint32_t
uint64_t  uint32_t       uint32_t      uint32_t
 *
 */
// TODO rewrite this inlight of the new batcher
template<typename data_type>
class EmbeddingQueryBatchManager {
  std::shared_ptr<uint8_t[]> buffer; 
  uint64_t buffer_size;
  uint32_t emb_dim;
  uint32_t num_queries;
  uint32_t embeddings_position;
  uint32_t metadata_position;
  bool copy_embeddings = true;

  uint32_t metadata_size;

  std::vector<std::shared_ptr<EmbeddingQuery<data_type>>> queries;

  void create_queries() {
    for (uint32_t i = 0; i < num_queries; i++) {
      queries.emplace_back(
			   std::move(std::make_shared<EmbeddingQuery<data_type>>(
										 this->buffer, this->buffer_size, get_metadata_position(i),
										 get_embeddings_position(i), emb_dim)));
      
    }

  }

public:
  EmbeddingQueryBatchManager(std::shared_ptr<uint8_t[]> buffer,
                             uint64_t buffer_size, uint32_t data_position) {
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    initialize(data_position);

  }
  EmbeddingQueryBatchManager(const uint8_t *buffer, uint64_t buffer_size) { 
    this->buffer_size = buffer_size;
    std::shared_ptr<uint8_t[]> copy(new uint8_t[this->buffer_size]);
    // std::cerr << "done shared: " << this->buffer_size;
    std::memcpy(copy.get(), buffer, this->buffer_size);
    // std::cerr << "done copy";
    this->buffer = std::move(copy);
    // std::cerr << "before init";
    initialize(0);
    
  }

  void initialize(uint32_t data_position) {
    // std::cerr << "starting initizalition" << std::endl;
    const uint32_t *header = reinterpret_cast<const uint32_t *>(this->buffer.get() + data_position);
    this->num_queries = header[0];
    this->queries.reserve(num_queries + 1);
    this->emb_dim = header[1];

    this->metadata_size = query_t<data_type>::get_metadata_size();

    this->metadata_position = data_position + EmbeddingQueryBatcher<data_type>::get_header_size();
    this->embeddings_position =
      data_position + EmbeddingQueryBatcher<data_type>::get_header_size() +
      this->metadata_size * this->num_queries;
  }

  
  std::vector<std::shared_ptr<EmbeddingQuery<data_type>>> &get_queries() {
    if (queries.empty())
      this->create_queries();

    assert(queries.size() == num_queries);
    return queries;
  }
  uint32_t get_num_queries() const {
    return this->num_queries;
  }

  /** get the embedding position in buffer starting from query number query_id */
  uint32_t get_embeddings_position(uint32_t query_id) {
    return this->embeddings_position +
           (query_id * (this->emb_dim * sizeof(data_type)));
  }
  uint32_t get_metadata_position(uint32_t query_id) {
    return this->metadata_position + (query_id * this->metadata_size);
  }
};



template <typename data_type>
struct greedy_query_t {
  uint8_t cluster_id; // this is to ensure that the candidate queue is sent to the correct cluster
  std::vector<uint32_t> candidate_queue;
  std::shared_ptr<EmbeddingQuery<data_type>> query;

  greedy_query_t(uint8_t _cluster_id,
                 std::vector<uint32_t> _candidate_queue,
                 std::shared_ptr<EmbeddingQuery<data_type>> _query) {
    cluster_id = _cluster_id;
    candidate_queue = _candidate_queue;
    query = _query;
  }
      

  static uint32_t get_metadata_size() { return sizeof(uint32_t) * 5 + sizeof(uint8_t);}

  void write_metadata(uint8_t *buffer) {
    uint32_t offset = 0;
    uint32_t query_id = query->get_query_id();
    uint32_t client_node_id = query->get_client_node_id();
    uint32_t K = query->get_K();
    uint32_t L = query->get_L();
    uint32_t cand_q_size = candidate_queue.size();

    std::memcpy(buffer + offset, &query_id, sizeof(query_id));
    offset += sizeof(query_id);

    std::memcpy(buffer + offset, &client_node_id, sizeof(client_node_id));
    offset += sizeof(client_node_id);

    std::memcpy(buffer + offset, &K, sizeof(K));
    offset += sizeof(K);

    std::memcpy(buffer + offset, &L, sizeof(L));
    offset += sizeof(L);

    std::memcpy(buffer + offset, &cand_q_size, sizeof(cand_q_size));
    offset += sizeof(cand_q_size);

    std::memcpy(buffer + offset, &cluster_id, sizeof(cluster_id));
  }
  uint32_t get_query_id() { return query->get_query_id(); }
  uint32_t get_query_emb_size() {
    return this->query->get_dim() * sizeof(data_type);
  }

  uint32_t get_candidate_queue_size() {
    return this->candidate_queue.size() * sizeof(uint32_t);
  }
  void write_embedding(uint8_t *buffer) {
    std::memcpy(buffer, this->query->get_embedding_ptr(), this->get_query_emb_size());
  }
  
  void write_candidate_queue(uint8_t *buffer) {
    std::memcpy(buffer, this->candidate_queue.data(), get_candidate_queue_size());
  }    
  
};

// TODO rewrite the starting points into candidate queues.
/**
   gathers and serializes the result from Head index search with the query
   embedding to be sent to UDL 2 for greedy searching on the whole graph
 */
// todo: replicate the batcher above
template <typename data_type> class GreedySearchQueryBatcher {
public:
  
  uint32_t dimension;
  std::vector<greedy_query_t<data_type>> queries;
  uint32_t query_emb_size;
  std::shared_ptr<derecho::cascade::Blob> blob;

  GreedySearchQueryBatcher() : dimension(0) {}
  GreedySearchQueryBatcher(uint32_t emb_dim, uint64_t size_hint = 1000) {
    dimension = emb_dim;
    this->query_emb_size = emb_dim * sizeof(data_type);
    queries.reserve(size_hint);
  }

  void add_query(greedy_query_t<data_type> query) {
    queries.emplace_back(std::move(query));
  }
  void add_query(uint8_t cluster_id,
                 std::vector<uint32_t> candidate_queue,
                 std::shared_ptr<EmbeddingQuery<data_type>> query) {
    greedy_query_t<data_type> tmp(cluster_id, candidate_queue, query);
    queries.emplace_back(std::move(tmp));
  }

  static uint32_t get_header_size() { return sizeof(uint32_t) * 2; }

  void write_header(uint8_t *buffer) {
    uint32_t num_queries = queries.size();
    std::memcpy(buffer, &num_queries, sizeof(num_queries));
    std::memcpy(buffer + sizeof(num_queries), &dimension, sizeof(dimension));
  }  

  size_t get_serialize_size() {
    size_t total_size =
      GreedySearchQueryBatcher<data_type>::get_header_size() +
      greedy_query_t<data_type>::get_metadata_size() * queries.size() +
      this->query_emb_size * queries.size();
    for (greedy_query_t<data_type> &query : queries) {
      total_size += query.get_candidate_queue_size();
    }
    return total_size;
  }

  void write_serialize(uint8_t *buffer) {
    uint32_t metadata_position = GreedySearchQueryBatcher<data_type>::get_header_size();
    uint32_t embedding_position =
      metadata_position +
      greedy_query_t<data_type>::get_metadata_size() *
      queries.size();
    uint32_t candidate_queue_position =
      embedding_position + this->query_emb_size * queries.size();
    // std::cout << metadata_position << " " << embedding_position << " " << candidate_queue_position << std::endl;
          

    write_header(buffer);
    for (greedy_query_t<data_type>& query : queries) {
      query.write_metadata(buffer + metadata_position);
      query.write_embedding(buffer + embedding_position);
      query.write_candidate_queue(buffer + candidate_queue_position);

      metadata_position += greedy_query_t<data_type>::get_metadata_size();
      embedding_position += this->query_emb_size;
      candidate_queue_position += query.get_candidate_queue_size();
      // std::cout << metadata_position << " " << embedding_position << " " << candidate_queue_position << std::endl;
    }
  }
  
  void serialize() {
    size_t total_size = get_serialize_size();

    this->blob = std::make_shared<derecho::cascade::Blob>(
							  [&](uint8_t *buffer, const std::size_t size) {
							    write_serialize(buffer);
							    return size;
							  },
							  total_size);
  }
  std::shared_ptr<derecho::cascade::Blob> get_blob() { return blob; }
  
  size_t get_count() {
    return queries.size();
  }

  void reset() {
    blob.reset();
    queries.clear();
  }

  bool is_empty() {
    return queries.empty();
  }    
};

template <typename data_type> class GreedySearchQuery {
  std::shared_ptr<uint8_t[]> buffer; // this ptr is the start of a batch of queries, not just this one
  uint64_t buffer_size;
  uint32_t query_id;
  uint32_t client_node_id, embeddings_position, embeddings_size,
      candidate_queue_position, candidate_queue_size, K, L;
  // cand size means numbers of candidates, not number of bytes
  uint32_t dim;
  uint8_t cluster_id;
  std::vector<uint32_t> candidate_queue;
public:
  GreedySearchQuery(std::shared_ptr<uint8_t[]> buffer, uint64_t buffer_size,
                    uint32_t metadata_position, uint32_t embeddings_position,
                    uint32_t candidate_queue_position, uint32_t emb_dim) {
    
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    const uint32_t *metadata = reinterpret_cast<const uint32_t *>(
								  buffer.get() + metadata_position);
    this->query_id = metadata[0];
    this->client_node_id = metadata[1];
    this->K = metadata[2];
    this->L = metadata[3];
    this->candidate_queue_size = metadata[4];
    this->cluster_id = *reinterpret_cast<const uint8_t *>(
							  buffer.get() + metadata_position +
							  greedy_query_t<data_type>::get_metadata_size() - sizeof(uint8_t));
    this->embeddings_position = embeddings_position; 
    this->candidate_queue_position = candidate_queue_position;
    this->dim = emb_dim;
  }
  uint64_t get_query_id() { return this->query_id; }

  uint32_t get_K()const { return this->K; }

  uint32_t get_L() const{ return this->L; }
  
  uint32_t get_client_node_id() { return this->client_node_id; }

  const data_type *get_embedding_ptr() {
    if (this->embeddings_position >= this->buffer_size) throw std::runtime_error("embedding for greedy search query is wrong");
    return reinterpret_cast<const data_type*>(this->buffer.get() + this->embeddings_position);
  }

  const uint32_t *get_candidate_queue_ptr() {
    if (this->candidate_queue_position > this->buffer_size) {
      throw std::runtime_error("pointer for candidate queue doesn't make sense");
    }
    return reinterpret_cast<const uint32_t *>(buffer.get() + this->candidate_queue_position);

  }

  uint32_t get_candidate_queue_size() {
    return this->candidate_queue_size;
  }

  uint64_t get_dim() { return this->dim; }

  uint8_t get_cluster_id() { return this->cluster_id; }

  std::vector<uint32_t> get_candidate_queue() {
    std::vector<uint32_t> cand(get_candidate_queue_ptr(),
                               get_candidate_queue_ptr() +
                               get_candidate_queue_size());
    return cand;
  }
  
};


/**
   this class deserializes the blob from greedysearchquerybatcher into
   individual greedy search queries.
   
 */
template <typename data_type>
class GreedySearchQueryBatchManager {
  std::shared_ptr<uint8_t[]> buffer; 
  uint64_t buffer_size;

  uint32_t emb_dim;
  uint32_t num_queries;
  uint32_t metadata_position;
  uint32_t embeddings_position;

  uint32_t header_size;
  uint32_t metadata_size;
  uint32_t total_embeddings_size;
  uint32_t embedding_size;

  std::vector<std::shared_ptr<GreedySearchQuery<data_type>>> queries;

  void create_queries() {
    uint32_t candidate_queue_position =
      get_embeddings_position(this->num_queries - 1) + this->embedding_size;
    
    for (uint32_t i = 0; i < num_queries; i++) {
      uint32_t current_cand_q_size = reinterpret_cast<const uint32_t *>(
									this->buffer.get() + get_metadata_position(i))[4];
      std::shared_ptr<GreedySearchQuery<data_type>> g_query =
        std::make_shared<GreedySearchQuery<data_type>>(
						       this->buffer, this->buffer_size, get_metadata_position(i),
						       get_embeddings_position(i), candidate_queue_position, emb_dim);
      
      queries.push_back(g_query);
      candidate_queue_position += current_cand_q_size * sizeof(uint32_t);
      // candidates in q must be node id which is uint32_t
    }
  }

public:
  // for usage with the GlobalSearchMessageBatchManager, 
  GreedySearchQueryBatchManager(std::shared_ptr<uint8_t[]> buffer,
                                uint64_t buffer_size, uint32_t data_position) {
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    initialize(data_position);
  }

  // for testing purposes, if you want to own the bytes of the whole blob
  GreedySearchQueryBatchManager(const uint8_t *buffer, uint64_t buffer_size) {
    this->buffer_size = buffer_size;
    std::shared_ptr<uint8_t[]> copy(new uint8_t[this->buffer_size]);
    std::memcpy(copy.get(), buffer, this->buffer_size);
    this->buffer = std::move(copy);
    initialize(0);
  }

  void initialize(uint32_t data_position) {
    const uint32_t *header = reinterpret_cast<const uint32_t *>(this->buffer.get() + data_position);
    this->header_size = GreedySearchQueryBatcher<data_type>::get_header_size();
    this->num_queries = header[0];
    this->queries.reserve(num_queries + 1);
    this->emb_dim = header[1];
    this->embedding_size = sizeof(data_type) * this->emb_dim;
    this->metadata_size = greedy_query_t<data_type>::get_metadata_size();
    this->metadata_position = data_position + this->header_size;
    this->embeddings_position =
      metadata_position + this->metadata_size * this->num_queries;

  }
  
  std::vector<std::shared_ptr<GreedySearchQuery<data_type>>> &get_queries() {
    if (queries.empty())
      create_queries();

    assert(queries.size() == num_queries);
    return queries;
  }
  uint32_t get_num_queries() const {
    return this->num_queries;
  }

  uint32_t get_metadata_position(uint32_t query_id) {
    return this->metadata_position + query_id * (this->metadata_size);
  }

  uint32_t get_embeddings_position(uint32_t query_id) {
    return this->embeddings_position + query_id * (this->embedding_size);
  }
  
};



// since this struct is so small, no need for a ComputeQuery class for deserialization
struct compute_query_t {
  uint32_t node_id;
  uint32_t query_id;
  uint32_t client_node_id;
  uint64_t batch_id = 0;
  float min_distance;
  uint8_t cluster_sender_id;
  uint8_t cluster_receiver_id;
  uint32_t receiver_thread_id;

  compute_query_t()
  : node_id(0), query_id(0), min_distance(0), cluster_sender_id(0),
  cluster_receiver_id(0), receiver_thread_id(0), client_node_id(0) {}

  compute_query_t(uint32_t node_id, uint32_t query_id, uint32_t client_node_id,
                  float min_distance, uint8_t cluster_sender_id,
                  uint8_t cluster_receiver_id, uint32_t receiver_thread_id)
  : node_id{node_id}, query_id{query_id}, client_node_id(client_node_id),
  min_distance{min_distance}, cluster_sender_id{cluster_sender_id},
  cluster_receiver_id{cluster_receiver_id},
  receiver_thread_id(receiver_thread_id) {}

  compute_query_t(const uint8_t* buffer, uint64_t buffer_size,
                  uint32_t data_position) {
    uint32_t offset = data_position;
    this->node_id = *reinterpret_cast<const uint32_t *>(buffer + offset);
    offset += sizeof(this->node_id);

    this->query_id = *reinterpret_cast<const uint32_t *>(buffer + offset);
    offset += sizeof(this->query_id);

    this->client_node_id = *reinterpret_cast<const uint32_t *>(buffer + offset);
    offset += sizeof(this->client_node_id);

    this->batch_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(this->batch_id);

    this->min_distance = *reinterpret_cast<const float *>(buffer + offset);
    offset += sizeof(this->min_distance);

    this->cluster_sender_id = *(buffer + offset);
    offset += sizeof(cluster_sender_id);
    this->cluster_receiver_id = *(buffer + offset);
    offset += sizeof(cluster_receiver_id);
    this->receiver_thread_id = *(buffer + offset);
    offset += sizeof(receiver_thread_id);

    if (offset > buffer_size) {
      throw std::runtime_error(
			       "compute_query_t: offset >= buffer_size: " + std::to_string(offset) +
			       " " + std::to_string(buffer_size));
    }
  }

  size_t get_serialize_size() const {
    return sizeof(node_id) + sizeof(query_id) + sizeof(min_distance) +
           sizeof(cluster_sender_id) + sizeof(cluster_receiver_id) + sizeof(receiver_thread_id) + sizeof(client_node_id) + sizeof(batch_id);
  }
  void set_batch_id(uint64_t batch_id) {
    this->batch_id = batch_id;
  }
  /*
    used for logging purposes
   */
  uint64_t get_msg_id()const {
    uint64_t msg_id =
      (static_cast<uint64_t>(this->query_id) << 32) | this->node_id;
    return msg_id;
  }

  void write_serialize(uint8_t *buffer) const {
    uint32_t offset = 0;
    std::memcpy(buffer + offset, &node_id, sizeof(node_id));
    offset += sizeof(node_id);

    std::memcpy(buffer + offset, &query_id, sizeof(query_id));
    offset += sizeof(query_id);

    std::memcpy(buffer + offset, &client_node_id, sizeof(client_node_id));
    offset += sizeof(client_node_id);

    std::memcpy(buffer + offset, &batch_id, sizeof(batch_id));
    offset += sizeof(batch_id);

    std::memcpy(buffer + offset, &min_distance, sizeof(min_distance));
    offset += sizeof(min_distance);

    std::memcpy(buffer + offset, &cluster_sender_id, sizeof(cluster_sender_id));
    offset += sizeof(cluster_sender_id);

    std::memcpy(buffer + offset, &cluster_receiver_id, sizeof(cluster_receiver_id));
    offset += sizeof(cluster_receiver_id);
    
    std::memcpy(buffer + offset, &receiver_thread_id, sizeof(receiver_thread_id));
    offset += sizeof(receiver_thread_id);
        
  }    
};

class ComputeQueryBatcher {
  std::vector<compute_query_t> queries;
  std::shared_ptr<derecho::cascade::Blob> blob;

public:
  ComputeQueryBatcher() {}
  ComputeQueryBatcher(uint32_t size_hint) { queries.reserve(size_hint); }
  
  void push(compute_query_t &compute_query) {
    queries.emplace_back(compute_query);
  }

  static uint32_t get_header_size() { return sizeof(uint32_t); }
  
  // number of queries
  void write_header(uint8_t *buffer) {
    uint32_t num_queries = queries.size();
    std::memcpy(buffer, &num_queries, sizeof(num_queries));
  }

  size_t get_serialize_size() {
    size_t total_size =
      get_header_size() + queries.size() * queries[0].get_serialize_size();
    return total_size;
  }

  void write_serialize(uint8_t *buffer) {
    uint32_t offset = 0;
    write_header(buffer + offset);
    offset += get_header_size();
    for (const auto &query : queries) {
      query.write_serialize(buffer + offset);
      offset += query.get_serialize_size();
    }
  }
  void serialize() {
    size_t total_size = get_serialize_size();
    blob = std::make_shared<derecho::cascade::Blob>(
						    [&](uint8_t *buffer, const std::size_t size) {
						      write_serialize(buffer);
						      return size;
						    },
						    total_size);
  }

  std::shared_ptr<derecho::cascade::Blob> get_blob() { return blob; }

  size_t get_count() { return queries.size(); }
  
  void reset() {
    blob.reset();
    queries.clear();
  }
  bool is_empty() {
    return queries.empty();

  }    
};


class ComputeQueryBatchManager {
  uint32_t data_position;
  uint32_t num_queries;

  std::vector<compute_query_t> queries;

  void create_queries(const uint8_t *buffer, uint64_t buffer_size) {
    uint32_t offset = data_position + ComputeQueryBatcher::get_header_size();
    for (uint32_t i = 0; i < num_queries; i++) {
      queries.emplace_back(buffer, buffer_size, offset);
      offset += queries[i].get_serialize_size();
    }
  }

public:
  ComputeQueryBatchManager(std::shared_ptr<uint8_t[]> buffer,
                           uint64_t buffer_size, uint32_t data_position) {
    this->data_position = data_position;
    this->num_queries =
      *reinterpret_cast<const uint32_t *>(buffer.get() + data_position);
    this->queries.reserve(num_queries);
    create_queries(buffer.get(), buffer_size);
    assert(num_queries == queries.size());
  }
  // missing constructor
  ComputeQueryBatchManager(const uint8_t *buffer, uint64_t buffer_size) {
    this->data_position = 0;
    this->num_queries =
      *reinterpret_cast<const uint32_t *>(buffer + data_position);
    this->queries.reserve(num_queries);
    create_queries(buffer, buffer_size);
    assert(num_queries == queries.size());
    // copying here is not necessary since all the compute queries only involve
    // scalar values that can be copied
  }

  std::vector<compute_query_t> &get_queries() {
    return queries;
  }

  uint32_t get_num_queries()  const {
    return this->num_queries;
  }

};

/**
   Send back results for a compute query. Will need to include std::vector of
   diskann::Neighbors.

Upon receviing the result, all of these nodes. Only insert the neighbors that
are unvisited. Mark all of these nodes as visited

will also send back full embedding of node id as well for reranking purposes later.
 */
struct compute_result_t {
  uint32_t num_neighbors;
  std::shared_ptr<uint32_t[]> nbr_ids;
  std::shared_ptr<float[]> nbr_distances;
  uint32_t query_id;
  uint32_t node_id;
  uint32_t client_node_id;
  uint64_t batch_id = 0;
  float expanded_dist;
  uint32_t receiver_thread_id;
  uint8_t cluster_sender_id;
  uint8_t cluster_receiver_id;

  compute_result_t()
  : num_neighbors(0), nbr_ids(nullptr), nbr_distances(nullptr), query_id(0),
  node_id(0), client_node_id(0), cluster_sender_id(0),
  cluster_receiver_id(0), receiver_thread_id(0), expanded_dist(0) {}

  compute_result_t(uint32_t num_neighbors, std::shared_ptr<uint32_t[]> nbr_ids,
                   std::shared_ptr<float[]> nbr_distances, uint32_t query_id,
                   uint32_t node_id, uint32_t client_node_id,
                   float expanded_dist, uint32_t receiver_thread_id,
                   uint8_t cluster_sender_id, uint8_t cluster_receiver_id)
  : num_neighbors(num_neighbors), nbr_ids(nbr_ids),
  nbr_distances(nbr_distances), query_id(query_id), node_id(node_id),
  client_node_id(client_node_id), expanded_dist(expanded_dist),
  receiver_thread_id(receiver_thread_id),
  cluster_sender_id(cluster_sender_id),
  cluster_receiver_id(cluster_receiver_id) {}
  
  size_t get_serialize_size() const {
    return sizeof(num_neighbors) + sizeof(uint32_t) * num_neighbors +
           sizeof(float) * num_neighbors + sizeof(query_id) + sizeof(node_id) +
           sizeof(client_node_id) + sizeof(batch_id) + sizeof(expanded_dist) +
           sizeof(receiver_thread_id) + sizeof(cluster_receiver_id) +
           sizeof(cluster_sender_id);
  }


  /**
     msg id of result is ~((static_cast<uint64_t>(query_id) << 32) | node_id);
     mgs id of query is (static_cast<uint64_t>(query_id) << 32) | node_id
   */
  uint64_t get_msg_id() const {
    uint64_t msg_id = (static_cast<uint64_t>(query_id) << 32) | node_id;
    return msg_id;
  }

  void write_serialize(uint8_t *buffer) const {
    uint32_t offset = 0;
    std::memcpy(buffer + offset, &num_neighbors, sizeof(num_neighbors));
    offset += sizeof(num_neighbors);

    std::memcpy(buffer + offset, nbr_ids.get(),
                sizeof(uint32_t) * num_neighbors);
    offset += sizeof(uint32_t) * num_neighbors;

    std::memcpy(buffer + offset, nbr_distances.get(),
                sizeof(float) * num_neighbors);
    offset += sizeof(float) * num_neighbors;

    std::memcpy(buffer + offset, &query_id, sizeof(query_id));
    offset += sizeof(query_id);

    std::memcpy(buffer + offset, &node_id, sizeof(node_id));
    offset += sizeof(node_id);

    std::memcpy(buffer + offset, &client_node_id, sizeof(client_node_id));
    offset += sizeof(client_node_id);

    std::memcpy(buffer + offset, &batch_id, sizeof(batch_id));
    offset += sizeof(batch_id);

    std::memcpy(buffer + offset, &expanded_dist, sizeof(expanded_dist));
    offset += sizeof(expanded_dist);

    std::memcpy(buffer + offset, &receiver_thread_id,
                sizeof(receiver_thread_id));
    offset += sizeof(receiver_thread_id);

    std::memcpy(buffer + offset, &cluster_sender_id, sizeof(cluster_sender_id));
    offset += sizeof(cluster_sender_id);

    std::memcpy(buffer + offset, &cluster_receiver_id,
                sizeof(cluster_receiver_id));
    offset += sizeof(cluster_receiver_id);
  }
  void set_batch_id(uint64_t batch_id) { this->batch_id = batch_id; }
};

// reason for this class to exist is because of shared ptr to buffer. Don't want
// to deallocate memory when it's still needed.
class ComputeResult {
  std::shared_ptr<uint8_t[]> buffer;
  uint64_t buffer_size;

  uint32_t num_neighbors;
  const uint32_t *nbr_ids;
  const float *nbr_distances;
  uint32_t query_id;
  uint32_t node_id;
  uint32_t client_node_id;
  uint64_t batch_id = 0;
  float expanded_dist;
  uint32_t receiver_thread_id;
  
  uint8_t cluster_sender_id;
  uint8_t cluster_receiver_id;
public:
  ComputeResult()
  : buffer(nullptr), buffer_size(0), nbr_ids(nullptr),
  nbr_distances(nullptr), query_id(0), node_id(0), expanded_dist(0),
  num_neighbors(0), receiver_thread_id(0), cluster_sender_id(0),
  cluster_receiver_id(0) {}
    
  ComputeResult(std::shared_ptr<uint8_t[]> buffer, uint64_t buffer_size,
                uint32_t data_position) {
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    initialize(data_position);
  }

  void initialize(uint32_t data_position) {
    uint32_t offset = data_position;

    this->num_neighbors =
      *reinterpret_cast<const uint32_t *>(buffer.get() + offset);
    offset += sizeof(num_neighbors);
    
    this->nbr_ids = reinterpret_cast<const uint32_t *>(buffer.get() + offset);
    offset += sizeof(uint32_t) * num_neighbors;

    this->nbr_distances =
      reinterpret_cast<const float *>(buffer.get() + offset);
    offset += sizeof(float) * num_neighbors;

    this->query_id = *reinterpret_cast<const uint32_t *>(buffer.get() + offset);
    offset += sizeof(uint32_t);

    this->node_id = *reinterpret_cast<const uint32_t *>(buffer.get() + offset);
    offset += sizeof(uint32_t);

    this->client_node_id = *reinterpret_cast<const uint32_t *>(buffer.get() + offset);
    offset += sizeof(client_node_id);

    this->batch_id = *reinterpret_cast<const uint64_t *>(buffer.get() + offset);
    offset += sizeof(batch_id);

    this->expanded_dist = *reinterpret_cast<const float *>(buffer.get() + offset);
    offset += sizeof(float);

    this->receiver_thread_id = *reinterpret_cast<const uint32_t *>(buffer.get() + offset);
    offset += sizeof(uint32_t);

    this->cluster_sender_id = *reinterpret_cast<const uint8_t *>(buffer.get() + offset);
    offset += sizeof(uint8_t);

    this->cluster_receiver_id = *reinterpret_cast<const uint8_t *>(buffer.get() + offset);
    offset += sizeof(uint8_t);
  }

  size_t get_serialize_size() const {
    return sizeof(num_neighbors) + sizeof(uint32_t) * num_neighbors +
           sizeof(float) * num_neighbors + sizeof(query_id) + sizeof(node_id) +
           sizeof(client_node_id) + sizeof(batch_id) + sizeof(expanded_dist) +
           sizeof(receiver_thread_id) + sizeof(cluster_sender_id) +
           sizeof(cluster_receiver_id);
  }

  uint32_t get_num_neighbors() const {
    return num_neighbors;
  }
  
  const uint32_t* get_nbr_ids() const {
    return nbr_ids;
  }
  
  const float* get_nbr_distances() const {
    return nbr_distances;
  }
  
  uint32_t get_query_id() const {
    return query_id;
  }

  uint32_t get_node_id() const { return node_id; }

  uint32_t get_client_node_id() const { return client_node_id; }

  uint64_t get_batch_id() const { return batch_id; }

  uint32_t get_receiver_thread_id() const { return receiver_thread_id; }

  float get_expanded_dist() const { return expanded_dist; }

  uint8_t get_cluster_sender_id() const { return cluster_sender_id; }

  uint8_t get_cluster_receiver_id() const { return cluster_receiver_id; }

  uint64_t get_msg_id() const {
    uint64_t msg_id = ((static_cast<uint64_t>(query_id) << 32) | node_id);
    return msg_id;
  }
};


class ComputeResultBatcher {
  std::shared_ptr<derecho::cascade::Blob> blob;
  std::vector<std::shared_ptr<compute_result_t>> results;
public:
  ComputeResultBatcher() {}

  ComputeResultBatcher(uint32_t size_hint) { results.reserve(size_hint); }

  void push(std::shared_ptr<compute_result_t> result) {
    results.emplace_back(std::move(result));
  }

  static uint32_t get_header_size() { return sizeof(uint32_t); }

  void write_header(uint8_t *buffer) {
    uint32_t num_res = results.size();
    std::memcpy(buffer, &num_res, sizeof(num_res));
  }

  size_t get_serialize_size() {
    size_t total_size = ComputeResultBatcher::get_header_size();
    for (const auto &result : results) {
      total_size += result->get_serialize_size();
    }
    return total_size;
  }

  void write_serialize(uint8_t *buffer) {
    uint32_t offset = 0;
    write_header(buffer + offset);
    offset += ComputeResultBatcher::get_header_size();
    for (const auto &result : results) {
      result->write_serialize(buffer + offset);
      offset += result->get_serialize_size();
    }
  }
  void serialize() {
    size_t total_size = get_serialize_size();
    blob = std::make_shared<derecho::cascade::Blob>(
						    [&](uint8_t *buffer, const std::size_t size) {
						      write_serialize(buffer);
						      return size;
						    },
						    total_size);
  }
  std::shared_ptr<derecho::cascade::Blob> get_blob() { return blob; }

  size_t get_count() {
    return results.size();
  }
  
  void reset() {
    blob.reset();
    results.clear();
  }

  bool is_empty() {
    return results.empty();
  }    
};

class ComputeResultBatchManager {
  std::shared_ptr<uint8_t[]> buffer;
  uint64_t buffer_size;
  uint32_t num_results;
  uint32_t data_position;

  std::vector<std::shared_ptr<ComputeResult>> results;
  void create_results() {
    uint32_t offset = data_position + ComputeResultBatcher::get_header_size();
    for (uint32_t i = 0; i < num_results; i++) {
      std::shared_ptr<ComputeResult> tmp_res =
        std::make_shared<ComputeResult>(buffer, buffer_size, offset);
      results.emplace_back(std::move(tmp_res));
      offset += results[i]->get_serialize_size();
    }
  }
  
public:
  ComputeResultBatchManager(std::shared_ptr<uint8_t[]> buffer,
                            uint64_t buffer_size, uint32_t data_position) {
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    this->data_position = data_position;
    this->num_results = *reinterpret_cast<const uint32_t *>(this->buffer.get() + data_position);
  }
  ComputeResultBatchManager(const uint8_t *buffer, uint64_t buffer_size) {
    std::shared_ptr<uint8_t[]> copy(new uint8_t[buffer_size]);
    std::memcpy(copy.get(), buffer, buffer_size);
    this->buffer = std::move(copy);
    this->buffer_size = buffer_size;
    this->data_position = 0;
    this->num_results =
      *reinterpret_cast<const uint32_t *>(this->buffer.get() + data_position);
    this->results.reserve(num_results);
  }

  std::vector<std::shared_ptr<ComputeResult>> &get_results() {
    if (results.empty()) {
      create_results();
    }
    assert(results.size() == num_results);
    return results;
  }
  uint32_t get_num_results() const {
    return this->num_results;
  }    
  
};  
  

struct ann_search_result_t {
  uint32_t query_id;
  uint32_t client_id;
  uint32_t K;
  uint32_t L;
  std::shared_ptr<uint32_t[]> search_result;
  uint8_t cluster_id;


  size_t get_serialize_size() const {
    return sizeof(query_id) + sizeof(client_id) + sizeof(K) + sizeof(L) + sizeof(cluster_id) + sizeof(uint32_t) * K;
  }

  void write_serialize(uint8_t *buffer) const {
    uint32_t offset = 0;
    std::memcpy(buffer + offset, &query_id, sizeof(query_id));
    offset += sizeof(query_id);
    
    std::memcpy(buffer + offset, &client_id, sizeof(client_id));
    offset += sizeof(client_id);

    std::memcpy(buffer + offset, &K, sizeof(K));
    offset += sizeof(K);

    std::memcpy(buffer + offset, &L, sizeof(L));
    offset += sizeof(L);

    for (uint32_t i = 0; i < K; i++) {
      std::memcpy(buffer + offset, search_result.get() + i, sizeof(uint32_t));
      offset += sizeof(uint32_t);
    }
    std::memcpy(buffer + offset, &cluster_id, sizeof(cluster_id));
  }
};



class ANNSearchResultBatcher {
  std::vector<ann_search_result_t> search_results;
  std::shared_ptr<derecho::cascade::Blob> blob;
public:
  ANNSearchResultBatcher(uint32_t size_hint = 100) {
    search_results.reserve(size_hint);
  }

  void push(ann_search_result_t res) { search_results.emplace_back(res); }

  static size_t get_header_size() { return sizeof(uint32_t);}

  void write_header(uint8_t *buffer) {
    uint32_t num_results = search_results.size();
    std::memcpy(buffer, &num_results, sizeof(num_results));
  }

  size_t get_serialize_size() {
    size_t total_size =
      get_header_size() +
      search_results.size() * (search_results.size() != 0
                               ? search_results[0].get_serialize_size()
                               : 0);
    return total_size;
  }

  void write_serialize(uint8_t *buffer) {
    uint32_t offset = 0;
    write_header(buffer + offset);
    offset += get_header_size();
    for (const auto &result : search_results) {
      result.write_serialize(buffer + offset);
      offset += result.get_serialize_size();
    }
  }

  void serialize() {
    size_t total_size = get_serialize_size();
    blob = std::make_shared<derecho::cascade::Blob>(
						    [&](uint8_t *buffer, const std::size_t size) {
						      write_serialize(buffer);
						      return size;
						    },
						    total_size);
  }

  std::shared_ptr<derecho::cascade::Blob> get_blob() { return blob; }
  void reset() {
    blob.reset();
    search_results.clear();
  }
  size_t get_count() { return search_results.size(); }

  bool is_empty() { return search_results.empty(); }

  uint32_t get_num_results() {
    return search_results.size();
  }    

};

class ANNSearchResult {
  std::shared_ptr<uint8_t[]> buffer;
  uint64_t buffer_size;

  uint32_t query_id;
  uint32_t client_id;
  uint32_t K;
  uint32_t L;
  const uint32_t *search_results;
  uint8_t cluster_id;

public:
  ANNSearchResult()
  : buffer(nullptr), buffer_size(0), query_id(0), client_id(0), K(0), L(0),
  search_results(nullptr), cluster_id(0) {}
  
  ANNSearchResult(std::shared_ptr<uint8_t[]> buffer, uint64_t buffer_size,
                  uint32_t data_location) {
    this->buffer = buffer;
    this->buffer_size = buffer_size;

    uint32_t offset = data_location;
    this->query_id = *reinterpret_cast<uint32_t *>(buffer.get() + offset);
    offset += sizeof(query_id);
    this->client_id = *reinterpret_cast<uint32_t *>(buffer.get() + offset);
    offset += sizeof(client_id);
    this->K = *reinterpret_cast<uint32_t *>(buffer.get() + offset);
    offset += sizeof(K);
    this->L = *reinterpret_cast<uint32_t *>(buffer.get() + offset);
    offset += sizeof(L);
    this->search_results = reinterpret_cast<uint32_t *>(buffer.get() + offset);
    offset += sizeof(uint32_t) * K;
    this->cluster_id = *(buffer.get() + offset);


  }
  const uint32_t* get_search_results_ptr() const {
    return this->search_results;
  }

  size_t get_serialize_size() const {
    return sizeof(query_id) + sizeof(client_id) + sizeof(K) + sizeof(L) + sizeof(cluster_id) + sizeof(uint32_t) * K;
  }

  uint32_t get_query_id() const {return query_id;}
  uint32_t get_client_id() const {return client_id;}
  uint32_t get_K() const { return K;}
  uint32_t get_L() const {return L;}
  uint8_t get_cluster_id() const { return cluster_id; }
  
  
};  

class ANNSearchResultBatchManager {
  std::shared_ptr<uint8_t[]> buffer;
  uint64_t buffer_size;

  uint32_t data_position;
  uint32_t num_results;
  
  std::vector<ANNSearchResult> results;

  void create_results() {
    uint32_t offset = data_position + ANNSearchResultBatcher::get_header_size();
    for (uint32_t i = 0; i < num_results; i++) {
      results.emplace_back(buffer, buffer_size, offset);
      offset += results[i].get_serialize_size();
    }
  }

public:
  ANNSearchResultBatchManager(std::shared_ptr<uint8_t[]> buffer,
                              uint64_t buffer_size, uint32_t data_position) {
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    this->data_position = data_position;
    this->num_results =
      *reinterpret_cast<const uint32_t *>(this->buffer.get() + data_position);
  }
  ANNSearchResultBatchManager(const uint8_t *buffer, uint64_t buffer_size) {
    std::shared_ptr<uint8_t[]> copy(new uint8_t[buffer_size]);
    std::memcpy(copy.get(), buffer, buffer_size);
    this->buffer = std::move(copy);
    this->buffer_size = buffer_size;
    this->data_position = 0;
    this->num_results =
      *reinterpret_cast<const uint32_t *>(this->buffer.get() + data_position);
  }

  uint32_t get_num_results() const{
    return this->num_results;
  }
  const std::vector<ANNSearchResult>& get_results() {
    if (this->results.empty()) {
      create_results();
    }

    assert(results.size() == num_results);
    return results;
  }    
};  

template <typename data_type>
class GlobalSearchMessageBatcher {
  // Usage: used to batch messages together and send to another cluster or to
  // notify the client
  // will use this class for the head index udl as well to batch the greedy
  // search queries.
  // data layout: header || query embeddings for 2ndary partitions || greedy
  // search queries || compute result || compute query

  void write_header(uint8_t *buffer) {
    uint32_t search_queries_position =
      GlobalSearchMessageBatcher<data_type>::get_header_size() +
      query_embeddings_batcher.get_serialize_size();
    uint32_t compute_results_position =
      search_queries_position + search_queries_batcher.get_serialize_size();
    uint32_t compute_queries_position =
      compute_results_position + compute_results_batcher.get_serialize_size();
    
    uint32_t offset = 0;    
    std::memcpy(buffer + offset, &search_queries_position,
                sizeof(search_queries_position));
    offset += sizeof(search_queries_position);

    std::memcpy(buffer + offset, &compute_results_position,
                sizeof(compute_results_position));
    offset += sizeof(compute_results_position);

    std::memcpy(buffer + offset, &compute_queries_position,
                sizeof(compute_queries_position));
  }

  std::shared_ptr<derecho::cascade::Blob> blob;

  EmbeddingQueryBatcher<data_type> query_embeddings_batcher;
  GreedySearchQueryBatcher<data_type> search_queries_batcher;
  ComputeQueryBatcher compute_queries_batcher;
  ComputeResultBatcher compute_results_batcher;

  // std::vector<global_search_message<data_type>> messages;
  uint32_t query_emb_dim; // this is only used for head search udl
  // sending stuff to global search udl
public:
  
  GlobalSearchMessageBatcher(uint32_t emb_dim, uint32_t size_hint = 100) {
    this->query_emb_dim = emb_dim;
    this->search_queries_batcher =
      GreedySearchQueryBatcher<data_type>(emb_dim, size_hint);
    this->compute_queries_batcher = ComputeQueryBatcher(size_hint);
    this->compute_results_batcher = ComputeResultBatcher(size_hint);
    this->query_embeddings_batcher =
      EmbeddingQueryBatcher<data_type>(emb_dim, size_hint);
  }

  static uint32_t get_header_size() { return sizeof(uint32_t) * 3; }
  void push_search_query(greedy_query_t<data_type> search_query) {
    this->search_queries_batcher.add_query(std::move(search_query));
  }

  void push_compute_query(compute_query_t compute_query) {
    this->compute_queries_batcher.push(compute_query);
  }

  void push_compute_result(std::shared_ptr<compute_result_t>compute_result) {
    this->compute_results_batcher.push(std::move(compute_result));
  }
  void push_embedding_query(
			    std::shared_ptr<EmbeddingQuery<data_type>> emb_query) {
    query_t<data_type> q(emb_query);
    this->query_embeddings_batcher.add_query(q);
  }
      
  size_t get_serialize_size() {
    size_t total_size =
      GlobalSearchMessageBatcher<data_type>::get_header_size();
    total_size += search_queries_batcher.get_serialize_size();
    total_size += compute_queries_batcher.get_serialize_size();
    total_size += compute_results_batcher.get_serialize_size();
    total_size += query_embeddings_batcher.get_serialize_size();
    return total_size;
  }


  void write_serialize(uint8_t *buffer) {
    uint32_t offset = 0;
    write_header(buffer + offset);
    offset += GlobalSearchMessageBatcher<data_type>::get_header_size();
    
    query_embeddings_batcher.write_serialize(buffer + offset);
    offset += query_embeddings_batcher.get_serialize_size();

    search_queries_batcher.write_serialize(buffer + offset);
    offset += search_queries_batcher.get_serialize_size();

    compute_results_batcher.write_serialize(buffer + offset);
    offset += compute_results_batcher.get_serialize_size();

    compute_queries_batcher.write_serialize(buffer + offset);
    offset += compute_queries_batcher.get_serialize_size();
    if (offset != get_serialize_size()) {
      throw std::runtime_error(
			       "serializatoin error: offset not equal to get_serialize_size");
    }
  }

  void serialize() {
    size_t total_size = get_serialize_size();

    this->blob = std::make_shared<derecho::cascade::Blob>(
							  [&](uint8_t *buffer, size_t size) {
							    write_serialize(buffer);
							    return size;
							  }, total_size);
  }
  
  std::shared_ptr<derecho::cascade::Blob> get_blob() { return blob; }
  void reset() {
    blob.reset();
    query_embeddings_batcher.reset();
    search_queries_batcher.reset();
    compute_queries_batcher.reset();
    compute_results_batcher.reset();
  }

  bool is_empty() {
    return query_embeddings_batcher.is_empty() &&
           search_queries_batcher.is_empty() &&
           compute_queries_batcher.is_empty() &&
           compute_results_batcher.is_empty();
  }    
};


template <typename data_type> class GlobalSearchMessageBatchManager {

  std::shared_ptr<uint8_t[]> buffer;
  uint64_t buffer_size;
  std::unique_ptr<GreedySearchQueryBatchManager<data_type>> greedy_search_manager;
  std::unique_ptr<ComputeQueryBatchManager> compute_query_manager;
  std::unique_ptr<ComputeResultBatchManager> compute_result_manager;
  std::unique_ptr<EmbeddingQueryBatchManager<data_type>>
      embedding_query_manager;

  uint32_t num_embedding_queries, num_greedy_search_queries, num_compute_results, num_compute_queries;
public:
  GlobalSearchMessageBatchManager(const uint8_t *buffer, uint64_t buffer_size, uint32_t emb_dim) {
    std::shared_ptr<uint8_t[]> copy(new uint8_t[buffer_size]);
    std::memcpy(copy.get(), buffer, buffer_size);
    this->buffer = std::move(copy);
    this->buffer_size = buffer_size;


    const uint32_t *header = reinterpret_cast<const uint32_t *>(this->buffer.get());
    uint32_t embedding_queries_position =
      GlobalSearchMessageBatcher<data_type>::get_header_size();
    uint32_t search_queries_position = header[0];
    uint32_t compute_results_position = header[1];
    uint32_t compute_queries_position = header[2];

    this->num_embedding_queries =
      *reinterpret_cast<const uint32_t *>(this->buffer.get() + embedding_queries_position);
    this->num_greedy_search_queries = *reinterpret_cast<const uint32_t *>(
									  this->buffer.get() + search_queries_position);
    this->num_compute_results = *reinterpret_cast<const uint32_t *>(
								    this->buffer.get() + compute_results_position);
    this->num_compute_queries = *reinterpret_cast<const uint32_t *>(
								    this->buffer.get() + compute_queries_position);

    if (num_embedding_queries != 0) {
      embedding_query_manager =
        std::make_unique<EmbeddingQueryBatchManager<data_type>>(
								this->buffer, this->buffer_size, embedding_queries_position);
    }

    if (num_greedy_search_queries != 0) {
          greedy_search_manager =
              std::make_unique<GreedySearchQueryBatchManager<data_type>>(
									 this->buffer, this->buffer_size, search_queries_position);
    }
    if (num_compute_results != 0) {
      compute_result_manager = std::make_unique<ComputeResultBatchManager>(
									   this->buffer, this->buffer_size, compute_results_position);
    }
    if (num_compute_queries != 0) {
      compute_query_manager = std::make_unique<ComputeQueryBatchManager>(
									 this->buffer, this->buffer_size, compute_queries_position);
    }
  }

  uint32_t get_num_embedding_queries() const { return num_embedding_queries; }

  uint32_t get_num_greedy_search_queries() const {
    return num_greedy_search_queries;
  }

  uint32_t get_num_compute_results() const { return num_compute_results; }

  uint32_t get_num_compute_queries() const { return num_compute_queries; }
  
  // const std::vector<
  std::vector<std::shared_ptr<EmbeddingQuery<data_type>>> &
  get_embedding_queries() const {
    return this->embedding_query_manager->get_queries();
  }

  std::vector<std::shared_ptr<GreedySearchQuery<data_type>>> &
  get_greedy_search_queries() const {
    return this->greedy_search_manager->get_queries();
  }
  std::vector<compute_query_t> &get_compute_queries() {
    return this->compute_query_manager->get_queries();
  }

  std::vector<std::shared_ptr<ComputeResult>> &get_compute_results() {
    return this->compute_result_manager->get_results();
  }

  uint32_t get_num_messages() const {
    return this->embedding_query_manager->get_num_queries() +
           this->greedy_search_manager->get_num_queries() +
           this->compute_query_manager->get_num_queries() +
           this->compute_result_manager->get_num_results();
  }
};  

uint8_t get_cluster_id(const std::string &key);

std::pair<uint32_t, uint64_t> parse_client_and_batch_id(const std::string &key);
std::pair<uint8_t, uint64_t> parse_cluster_and_batch_id(const std::string&key);
