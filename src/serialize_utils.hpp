#pragma once

#include <cstdint>
#include <cstring>
#include <queue>
#include <memory>
#include <unordered_map>
#include <cascade/service_client_api.hpp>
/**
   One optimziation for both the batch manager which accepts a buffer pointer
from an blob object is to avoid doing memcpy and instead just let a shared_ptr
handle that address:

int* raw_ptr = new int[10];  // Allocates memory at address X
std::shared_ptr<int[]> sp(raw_ptr, std::default_delete<int[]>());

Should look into this after testing that the system works.
*/

template <typename data_type>
struct query_t {
  uint32_t query_id;
  uint32_t client_node_id;
  const data_type *query_emb;
  uint32_t dimension;

  query_t(uint32_t _query_id, uint32_t _client_node_id, const data_type *_query_emb,
          uint32_t _dimension): query_id(_query_id), client_node_id(_client_node_id), query_emb(_query_emb), dimension(_dimension) {}
  

  static uint32_t get_metadata_size() {
    return sizeof(query_id) + sizeof(client_node_id);
  }
  void write_metadata(uint8_t *buffer) {
    std::memcpy(buffer, &query_id, sizeof(query_id));
    std::memcpy(buffer + sizeof(query_id), &client_node_id,
                sizeof(client_node_id));
  }
  uint32_t get_query_emb_size() { return sizeof(data_type) * dimension; }
  uint32_t get_query_id() {
    return query_id;
  }
  void write_emb(uint8_t *buffer) {
    std::memcpy(buffer, query_emb, sizeof(data_type) * dimension);
  }    
};

template <typename data_type> class EmbeddingQueryBatcher {
public:
  struct {
    uint32_t num_queries;
    uint32_t dimension;

    uint32_t size() { return sizeof(num_queries) + sizeof(dimension); }
    void write_header(uint8_t *buffer) {
      std::memcpy(buffer, &num_queries, sizeof(num_queries));
      std::memcpy(buffer + sizeof(num_queries), &dimension, sizeof(dimension));
    }
  } header;

  static uint32_t get_header_size() { return sizeof(uint32_t) * 2; }
  
  std::vector<query_t<data_type>> queries;
  uint64_t query_emb_size;
  std::shared_ptr<derecho::cascade::Blob> blob;

  EmbeddingQueryBatcher(uint32_t emb_dim, uint64_t size_hint) {
    this->header.dimension = emb_dim;
    this->query_emb_size = emb_dim * sizeof(data_type);
    queries.reserve(size_hint);
  }
  
  void add_query(query_t<data_type> &query) {
    queries.push_back(query);
  }

  void add_query(uint32_t query_id, uint32_t client_node_id,
                 data_type *query_data, uint32_t dimension) {
    queries.emplace_back(query_id, client_node_id, query_data, dimension);
  }

  void serialize() {
    this->header.num_queries = queries.size();
    uint32_t total_size =
        this->header.size() +
        query_t<data_type>::get_metadata_size() * queries.size() +
        this->query_emb_size * queries.size();

    this->blob = std::make_shared<derecho::cascade::Blob>(
        [&](uint8_t *buffer, const std::size_t size) {
          uint32_t metadata_position = this->header.size();
          uint32_t embedding_position = metadata_position + 
					query_t<data_type>::get_metadata_size() *
					this->header.num_queries;

          this->header.write_header(buffer);
          for (query_t<data_type>& query : queries) {
            query.write_metadata(buffer + metadata_position);
            query.write_emb(buffer + embedding_position);

            metadata_position += query_t<data_type>::get_metadata_size();
            embedding_position += query.get_query_emb_size();
          }
          return size;
	}, total_size);
  }
  std::shared_ptr<derecho::cascade::Blob> get_blob() { return blob; }

  void reset() {
    blob.reset();
    queries.clear();
  }    
};




/*
 * EmbeddingQuery encapsulates a single embedding query that is part of a batch.
 * Operations are performed on demand and directly from the buffer of the whole
 * batch.
 *
 *
 */
// TODO rewrite to match the new batcher
template<typename data_type>
class EmbeddingQuery {
public:
  std::shared_ptr<uint8_t[]> buffer; // this ptr is the start of a batch of queries, not just this one
  uint64_t buffer_size;
  uint32_t query_id;
  uint32_t client_node_id, embeddings_position, embeddings_size;
  uint32_t dim;

  EmbeddingQuery(std::shared_ptr<uint8_t[]> buffer, uint64_t buffer_size,
                 uint32_t metadata_position, uint32_t embeddings_position, uint32_t emb_dim) {
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    const uint32_t *metadata = reinterpret_cast<const uint32_t *>(
							    buffer.get() + metadata_position);
    this->query_id = metadata[0];
    this->client_node_id = metadata[1];
    this->embeddings_position = embeddings_position;

    this->dim = emb_dim;
  }
  uint64_t get_query_id() {
    return this->query_id;
  }
  uint32_t get_client_node_id() { return this->client_node_id; }

  const data_type *get_embedding_ptr() {
    if (this->embeddings_position >= this->buffer_size) return nullptr;
    return reinterpret_cast<const data_type*>(this->buffer.get() + this->embeddings_position);
  }

  uint64_t get_dim() {
    return this->dim;
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
    bool copy_embeddings = true;

    uint32_t header_size;
    uint32_t metadata_size;
    uint32_t total_embeddings_size;

    std::vector<std::shared_ptr<EmbeddingQuery<data_type>>> queries;

    void create_queries() {
      for (uint32_t i = 0; i < num_queries; i++) {
        uint32_t metadata_position = header_size + (i * metadata_size);
        
        queries.emplace_back(std::move(std::make_shared<EmbeddingQuery<data_type>>(this->buffer, this->buffer_size, metadata_position, get_embeddings_position(i), emb_dim)));
      }

    }

public:
  EmbeddingQueryBatchManager(const uint8_t *buffer, uint64_t buffer_size,
                             bool copy_embeddings = true) { // why would you not want to copty the embedding?

    this->copy_embeddings = copy_embeddings;

    const uint32_t *header = reinterpret_cast<const uint32_t *>(buffer);
    this->header_size = EmbeddingQueryBatcher<data_type>::get_header_size();
    this->num_queries = header[0];
    this->emb_dim = header[1];

    this->metadata_size = query_t<data_type>::get_metadata_size();

    this->embeddings_position = this->header_size + this->metadata_size * this->num_queries;
    this->total_embeddings_size = buffer_size - this->embeddings_position;
    if (this->total_embeddings_size != this->emb_dim * this->num_queries * sizeof(data_type)) {
      std::string msg = "num_queries: " + std::to_string(num_queries) + ", dim: " + std::to_string(emb_dim);
      throw std::runtime_error("Embedding size is not what we expected, " +
                               msg + ", total_embedding_size as calculated by buffer_size - this->embeddings_position: " + std::to_string(total_embeddings_size));
    }

    if(copy_embeddings){ // why would you not want to copy the embeddings?
        this->buffer_size = buffer_size;
    } else {
        this->buffer_size = buffer_size - this->total_embeddings_size;
    }

    std::shared_ptr<uint8_t[]> copy(new uint8_t[this->buffer_size]);
    std::memcpy(copy.get(), buffer, this->buffer_size);
    this->buffer = std::move(copy);
  }
  const std::vector<std::shared_ptr<EmbeddingQuery<data_type>>> &get_queries() {
    if (this->queries.empty()) {
      this->create_queries();
    }
    // std::cout << "num queries " << queries.size() << std::endl;
    return queries;
  }
  uint32_t get_num_queries() {
    return this->num_queries;
  }

  /** get the embedding position in buffer starting from query number query_id */
  uint32_t get_embeddings_position(uint64_t query_id) {
    return this->embeddings_position +
           (query_id * (this->emb_dim * sizeof(data_type)));
  }
  
};





template <typename data_type>
struct greedy_query_t {
  std::byte cluster_id; // this is to ensure that the candidate queue is sent to the correct cluster
  std::vector<uint32_t> candidate_queue;
  std::shared_ptr<EmbeddingQuery<data_type>> query;

  greedy_query_t(std::byte _cluster_id,
                 std::vector<uint32_t> _candidate_queue,
                 std::shared_ptr<EmbeddingQuery<data_type>> _query) {
    cluster_id = _cluster_id;
    candidate_queue = _candidate_queue;
    query = _query;
  }
      

  static uint32_t get_metadata_size() { return sizeof(uint32_t) * 3 + sizeof(std::byte);}

  void write_metadata(uint8_t *buffer) {
    uint32_t query_id = query->get_query_id();
    uint32_t client_node_id = query->get_client_node_id();
    uint32_t cand_q_size = candidate_queue.size();
    std::memcpy(buffer, &query_id, sizeof(query_id));
    std::memcpy(buffer + sizeof(query_id), &client_node_id, sizeof(client_node_id));
    std::memcpy(buffer + sizeof(query_id) + sizeof(client_node_id),
                &cand_q_size, sizeof(cand_q_size));
    std::memcpy(buffer + sizeof(query_id) + sizeof(client_node_id) +
                    sizeof(cand_q_size),
                &cluster_id, sizeof(cluster_id));
  }
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
  struct {
    uint32_t num_queries;
    uint32_t dimension;

    uint32_t size() { return sizeof(num_queries) + sizeof(dimension); }
    void write_header(uint8_t *buffer) {
      std::memcpy(buffer, &num_queries, sizeof(num_queries));
      std::memcpy(buffer + sizeof(num_queries), &dimension, sizeof(dimension));
    }
  } header;

  static uint32_t get_header_size() { return sizeof(uint32_t) * 2; }

  std::vector<greedy_query_t<data_type>> queries;
  uint32_t query_emb_size;
  std::shared_ptr<derecho::cascade::Blob> blob;

  GreedySearchQueryBatcher(uint32_t emb_dim, uint64_t size_hint = 1000) {
    this->header.dimension = emb_dim;
    this->query_emb_size = emb_dim * sizeof(data_type);
    queries.reserve(size_hint);
  }

  void add_query(greedy_query_t<data_type> query) {
    queries.push_back(query);
  }
  void add_query(std::byte cluster_id,
                 std::vector<uint32_t> candidate_queue,
                 std::shared_ptr<EmbeddingQuery<data_type>> query) {
    greedy_query_t<data_type> tmp(cluster_id, candidate_queue, query);
    queries.push_back(tmp);
  }
  
  void serialize() {
    this->header.num_queries = queries.size();
    uint32_t total_size =
        this->header.size() +
        greedy_query_t<data_type>::get_metadata_size() * queries.size() +
        this->query_emb_size * queries.size();
    for (greedy_query_t<data_type> &query : queries) {
      total_size += query.get_candidate_queue_size();
    }

    this->blob = std::make_shared<derecho::cascade::Blob>(
        [&](uint8_t *buffer, const std::size_t size) {
          uint32_t metadata_position = this->header.size();
          uint32_t embedding_position =
              metadata_position +
              greedy_query_t<data_type>::get_metadata_size() *
              this->header.num_queries;
          uint32_t candidate_queue_position =
              embedding_position +
              this->query_emb_size * this->header.num_queries;
          

          this->header.write_header(buffer);
          for (greedy_query_t<data_type>& query : queries) {
            query.write_metadata(buffer + metadata_position);
            query.write_embedding(buffer + embedding_position);
            query.write_candidate_queue(buffer + candidate_queue_position);

            metadata_position += greedy_query_t<data_type>::get_metadata_size();
            embedding_position += this->query_emb_size;
            candidate_queue_position += query.get_candidate_queue_size();
          }
          return size;
	}, total_size);
  }
  std::shared_ptr<derecho::cascade::Blob> get_blob() { return blob; }
  

  void reset() {
    blob.reset();
    queries.clear();
  }    
};

template <typename data_type> class GreedySearchQuery {
  std::shared_ptr<uint8_t[]> buffer; // this ptr is the start of a batch of queries, not just this one
  uint64_t buffer_size;
  uint32_t query_id;
  uint32_t client_node_id, embeddings_position, embeddings_size,
      candidate_queue_position, candidate_queue_size;
  // cand size means numbers of candidates, not number of bytes
  uint32_t dim;
  std::byte cluster_id;
  std::vector<uint32_t> candidate_queue;
public:
  GreedySearchQuery(std::shared_ptr<uint8_t[]> buffer, uint64_t buffer_size,
                    uint32_t metadata_position, uint32_t embeddings_position, uint32_t candidate_queue_position,uint32_t emb_dim) {
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    const uint32_t *metadata = reinterpret_cast<const uint32_t *>(
							    buffer.get() + metadata_position);
    this->query_id = metadata[0];
    this->client_node_id = metadata[1];
    this->candidate_queue_size = metadata[2];
    this->cluster_id = *reinterpret_cast<const std::byte *>(
							    buffer.get() + metadata_position + sizeof(uint32_t) * 3);
    this->embeddings_position = embeddings_position;
    this->candidate_queue_position = candidate_queue_position;
    
    this->dim = emb_dim;
  }
  uint64_t get_query_id() {
    return this->query_id;
  }
  uint32_t get_client_node_id() { return this->client_node_id; }

  const data_type *get_embedding_ptr() {
    if (this->embeddings_position >= this->buffer_size) throw std::runtime_error("embedding for greedy search query is wrong");
    return reinterpret_cast<const data_type*>(this->buffer.get() + this->embeddings_position);
  }

  const uint32_t *get_candidate_queue_ptr() {
    if (this->candidate_queue_position >= this->buffer_size) {
      throw std::runtime_error("pointer for candidate queue doesn't make sense");
    }
    return reinterpret_cast<const uint32_t *>(buffer.get() + this->candidate_queue_position);

  }

  uint32_t get_candidate_queue_size() {
    return this->candidate_queue_size;
  }

  uint64_t get_dim() { return this->dim; }

  std::byte get_cluster_id() { return this->cluster_id; }
  
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
  uint32_t embeddings_position;
  bool copy_embeddings = true;

  uint32_t header_size;
  uint32_t metadata_size;
  uint32_t total_embeddings_size;
  uint32_t embedding_size;

  std::vector<std::shared_ptr<GreedySearchQuery<data_type>>> queries;

  void create_queries() {
    uint32_t candidate_queue_position =
      get_embeddings_position(this->num_queries - 1) + this->embedding_size;
    
    for (uint32_t i = 0; i < num_queries; i++) {
      uint32_t metadata_position = header_size + (i * metadata_size);
      uint32_t current_cand_q_size = reinterpret_cast<const uint32_t *>(
									this->buffer.get() + metadata_position)[2];
      std::shared_ptr<GreedySearchQuery<data_type>> g_query = std::make_shared<GreedySearchQuery<data_type>>(
          this->buffer, this->buffer_size, metadata_position,
									       get_embeddings_position(i), candidate_queue_position, emb_dim);
      
      queries.push_back(g_query);
      candidate_queue_position += current_cand_q_size * sizeof(uint32_t);
      // candidates in q must be node id which is uint32_t
    }

  }

public:
  GreedySearchQueryBatchManager(const uint8_t *buffer, uint64_t buffer_size,
                             bool copy_embeddings = true) { // why would you not want to copty the embedding?

    this->copy_embeddings = copy_embeddings;

    const uint32_t *header = reinterpret_cast<const uint32_t *>(buffer);
    this->header_size = GreedySearchQueryBatcher<data_type>::get_header_size();
    this->num_queries = header[0];
    this->emb_dim = header[1];
    this->embedding_size = sizeof(data_type) * this->emb_dim;
    this->total_embeddings_size = this->num_queries * this->embedding_size;

    this->metadata_size = greedy_query_t<data_type>::get_metadata_size();

    this->embeddings_position = this->header_size + this->metadata_size * this->num_queries;

    if(copy_embeddings){ // why would you not want to copy the embeddings?
        this->buffer_size = buffer_size;
    } else {
      throw std::runtime_error("Right now, copy emb must be true");
      this->buffer_size = buffer_size - this->total_embeddings_size;
      // this is currently not correct because of the candidate queue at the end
    }

    std::shared_ptr<uint8_t[]> copy(new uint8_t[this->buffer_size]);
    std::memcpy(copy.get(), buffer, this->buffer_size);
    this->buffer = std::move(copy);
  }
  const std::vector<std::shared_ptr<GreedySearchQuery<data_type>>> &get_queries() {
    if (this->queries.empty()) {
      this->create_queries();
    }
    return queries;
  }
  uint32_t get_num_queries() {
    return this->num_queries;
  }

  /** get the embedding position in buffer starting from query number query_id */
  uint32_t get_embeddings_position(uint32_t query_id) {
    return this->embeddings_position + query_id * (this->embedding_size);
  }
  
};


enum global_message_type : uint8_t {
  SEARCH_QUERY,
  DISTANCE_TASK,
  DISTANCE_RES
};

// template <typename data_type> class ComputationTaskQuery {
  
// public:
  // ComputationTaskQuery();
  

// };


// template <typename data_type>
//     class GlobalSearchMessage {
//   global_message_type msg_type;
//   union msg {
//     GreedySearchQuery<data_type> search_query;
    

//   };
// public:
//   GlobalSearchMessage(global_message_type msg_type, ) {

//   }

// };
