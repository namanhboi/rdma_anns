#include "ssd_partition_index.h"
#include "communicator.h"
#include "query_buf.h"
#include "types.h"
#include "utils.h"
#include <chrono>
#include <limits>
#include <stdexcept>
#include <unordered_map>

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::SSDPartitionIndex(
    pipeann::Metric m, uint8_t partition_id, uint32_t num_partitions,
    uint32_t num_search_threads, std::shared_ptr<AlignedFileReader> &fileReader,
    std::unique_ptr<P2PCommunicator> &communicator,
    DistributedSearchMode dist_search_mode, bool tags,
    pipeann::Parameters *params, uint64_t num_queries_balance, bool enable_locs,
    bool use_batching, uint64_t max_batch_size)
    : reader(fileReader), communicator(communicator),
      client_state_prod_token(global_state_queue),
      server_state_prod_token(global_state_queue),
      dist_search_mode(dist_search_mode), max_batch_size(max_batch_size),
      use_batching(use_batching) {
  if (dist_search_mode == DistributedSearchMode::LOCAL) {
    assert(communicator == nullptr);
  }

  if (num_queries_balance > max_queries_balance) {
    throw std::invalid_argument("number of queries to balance too big " +
                                std::to_string(num_queries_balance));
  }
  this->num_queries_balance = num_queries_balance;
  this->my_partition_id = partition_id;
  this->num_partitions = num_partitions;
  if (num_partitions > 1) {
    if (dist_search_mode != DistributedSearchMode::STATE_SEND) {
      throw std::invalid_argument(
          "dist search mode doesn't make sense with num_partitions, if "
          "num_partitions > 1 then dist search mode must be STATE SEND");
    }
  }
  if (num_search_threads > MAX_SEARCH_THREADS) {
    throw std::invalid_argument("num search threads > MAX_SEARCH_THREADS");
  }
  this->num_search_threads = num_search_threads;
  data_is_normalized = false;
  this->enable_tags = tags;
  this->enable_locs = enable_locs;
  if (m == pipeann::Metric::COSINE) {
    if (std::is_floating_point<T>::value) {
      LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                   "Changing distance to L2 to boost accuracy.";
      m = pipeann::Metric::L2;
      data_is_normalized = true;

    } else {
      LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                 << " This may result in erroneous results or poor recall."
                 << " Consider using L2 distance with integral data types.";
    }
  }

  this->dist_cmp.reset(pipeann::get_distance_function<T>(m));
  // this->pq_reader = new LinuxAlignedFileReader();
  if (params != nullptr) {
    this->beamwidth = params->Get<uint32_t>("beamwidth");
    this->l_index = params->Get<uint32_t>("L");
    this->range = params->Get<uint32_t>("R");
    this->maxc = params->Get<uint32_t>("C");
    this->alpha = params->Get<float>("alpha");
    LOG(INFO) << "Beamwidth: " << this->beamwidth << ", L: " << this->l_index
              << ", R: " << this->range << ", C: " << this->maxc;
  }
}

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::~SSDPartitionIndex() {
  if (load_flag) {
    reader->close();
  }
  if (medoids != nullptr) {
    delete[] medoids;
  }
}

template <typename T, typename TagT>
int SSDPartitionIndex<T, TagT>::load(const char *index_prefix,
                                     bool new_index_format,
                                     const char *cluster_assignment_file) {
  std::string pq_table_bin, pq_compressed_vectors, disk_index_file,
      centroids_file;

  std::string iprefix = std::string(index_prefix);
  pq_table_bin = iprefix + "_pq_pivots.bin";
  pq_compressed_vectors = iprefix + "_pq_compressed.bin";
  disk_index_file = iprefix + "_disk.index";
  this->_disk_index_file = disk_index_file;
  centroids_file = disk_index_file + "_centroids.bin";

  std::ifstream index_metadata(disk_index_file, std::ios::binary);

  size_t tags_offset = 0;
  size_t pq_pivots_offset = 0;
  size_t pq_vectors_offset = 0;
  uint64_t disk_nnodes;
  uint64_t disk_ndims;
  size_t medoid_id_on_file;
  uint64_t file_frozen_id;

  if (new_index_format) {
    uint32_t nr, nc;

    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, max_node_len);
    READ_U64(index_metadata, nnodes_per_sector);
    data_dim = disk_ndims;
    max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
    if (max_degree != this->range) {
      LOG(ERROR) << "Range mismatch: " << max_degree << " vs " << this->range
                 << ", setting range to " << max_degree;
      this->range = max_degree;
    }

    LOG(INFO) << "Meta-data: # nodes per sector: " << nnodes_per_sector
              << ", max node len (bytes): " << max_node_len
              << ", max node degree: " << max_degree << ", npts: " << nr
              << ", dim: " << nc << " disk_nnodes: " << disk_nnodes
              << " disk_ndims: " << disk_ndims;

    if (nnodes_per_sector > this->kMaxElemInAPage) {
      LOG(ERROR)
          << "nnodes_per_sector: " << nnodes_per_sector << " is greater than "
          << this->kMaxElemInAPage
          << ". Please recompile with a higher value of kMaxElemInAPage.";
      return -1;
    }

    READ_U64(index_metadata, this->num_frozen_points);
    READ_U64(index_metadata, file_frozen_id);
    if (this->num_frozen_points == 1) {
      this->frozen_location = file_frozen_id;
      // if (this->num_frozen_points == 1) {
      LOG(INFO) << " Detected frozen point in index at location "
                << this->frozen_location
                << ". Will not output it at search time.";
    }
    READ_U64(index_metadata, tags_offset);
    READ_U64(index_metadata, pq_pivots_offset);
    READ_U64(index_metadata, pq_vectors_offset);

    LOG(INFO) << "Tags offset: " << tags_offset
              << " PQ Pivots offset: " << pq_pivots_offset
              << " PQ Vectors offset: " << pq_vectors_offset;
  } else { // old index file format
    size_t actual_index_size = get_file_size(disk_index_file);
    size_t expected_file_size;
    READ_U64(index_metadata, expected_file_size);
    if (actual_index_size != expected_file_size) {
      LOG(INFO) << "File size mismatch for " << disk_index_file
                << " (size: " << actual_index_size << ")"
                << " with meta-data size: " << expected_file_size;
      return -1;
    }

    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, max_node_len);
    READ_U64(index_metadata, nnodes_per_sector);
    max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;

    LOG(INFO) << "Disk-Index File Meta-data: # nodes per sector: "
              << nnodes_per_sector;
    LOG(INFO) << ", max node len (bytes): " << max_node_len;
    LOG(INFO) << ", max node degree: " << max_degree;
  }

  this->num_points = this->init_num_pts = disk_nnodes;
  size_per_io =
      SECTOR_LEN *
      (nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(max_node_len, SECTOR_LEN));
  LOG(INFO) << "Size per IO: " << size_per_io;

  index_metadata.close();

  pq_pivots_offset = 0;
  pq_vectors_offset = 0;

  LOG(INFO) << "After single file index check, Tags offset: " << tags_offset
            << " PQ Pivots offset: " << pq_pivots_offset
            << " PQ Vectors offset: " << pq_vectors_offset;

  size_t npts_u64, nchunks_u64;
  pipeann::load_bin<uint8_t>(pq_compressed_vectors, data, npts_u64, nchunks_u64,
                             pq_vectors_offset);
  this->n_chunks = nchunks_u64;
  this->global_graph_num_points = npts_u64;
  this->cur_id = this->num_points;

  LOG(INFO) << "Load compressed vectors from file: " << pq_compressed_vectors
            << " offset: " << pq_vectors_offset << " num points: " << npts_u64
            << " n_chunks: " << nchunks_u64;

  pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64,
                                pq_pivots_offset);

  if (num_partitions == 1 && disk_nnodes != npts_u64) {
    LOG(INFO) << "Mismatch in #points for compressed data file and disk "
                 "index file: "
    << disk_nnodes << " vs " << npts_u64;
    throw std::invalid_argument(
        "Mismatch in #points for compressed data file and disk "
        "index file: " +
        std::to_string(disk_nnodes) + " " + std::to_string(npts_u64));
    return -1;
  }

  this->data_dim = pq_table.get_dim();
  this->aligned_dim = ROUND_UP(this->data_dim, 8);

  LOG(INFO) << "Loaded PQ centroids and in-memory compressed vectors. #points: "
            << num_points << " #dim: " << data_dim
            << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks;

  // read index metadata
  // open AlignedFileReader handle to index_file
  std::string index_fname(disk_index_file);
  reader->open(index_fname, false, false);

  // load tags
  if (this->enable_tags) {
    std::string tag_file = disk_index_file + ".tags";
    LOG(INFO) << "Loading tags from " << tag_file;
    this->load_tags(tag_file);
  }

  num_medoids = 1;
  medoids = new uint32_t[1];
  medoids[0] = (uint32_t)(medoid_id_on_file);
  LOG(INFO) << "num partitions is " << num_partitions;
  if (num_partitions > 1) {
    // loading the id2loc file
    if (enable_locs) {
      std::string id2loc_file = iprefix + "_ids_uint32.bin";
      if (!file_exists(id2loc_file)) {
	throw std::invalid_argument(
				    "number of partitions is " + std::to_string(num_partitions) +
				    ", but the id2loc file doesn't exist: " + id2loc_file);
      }
      LOG(INFO) << "Load id2loc from existing file: " << id2loc_file;
      std::vector<TagT> id2loc_v;
      size_t id2loc_num, id2loc_dim;
      pipeann::load_bin<TagT>(id2loc_file, id2loc_v, id2loc_num, id2loc_dim, 0);
      if (id2loc_dim != 1) {
	throw std::runtime_error(
				 "dim from id2loc file " + id2loc_file +
				 " had value not 1: " + std::to_string(id2loc_dim));
      }
      if (id2loc_num != num_points) {
	throw std::runtime_error(
				 "num points from id2loc file " + id2loc_file + " had value" +
				 std::to_string(id2loc_num) +
				 " not equal to numpoints from index: " + std::to_string(num_points));
      }
      for (uint32_t i = 0; i < id2loc_num; i++) {
	id2loc_.insert_or_assign(id2loc_v[i], i);
      }
    LOG(INFO) << "Id2loc file loaded successfully: " << id2loc_.size();
    }
    std::string cluster_file(cluster_assignment_file);
    if (!file_exists(cluster_file)) {
      throw std::invalid_argument(
          "number of partitions is " + std::to_string(num_partitions) +
          ", but the cluster assignment bin file doesn't exist: " +
          cluster_file);
    }
    size_t ca_num_pts, ca_dim;
    pipeann::load_bin<uint8_t>(cluster_assignment_file, cluster_assignment,
                               ca_num_pts, ca_dim);
    if (ca_num_pts != this->global_graph_num_points) {
      throw std::invalid_argument("Number of points differ between partition "
                                  "assignment file and the disk index");
    }
    LOG(INFO) << "cluster assignment file loaded successfully.";
  }

  LOG(INFO) << "SSDIndex loaded successfully.";

  load_flag = true;
  return 0;
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::load_tags(const std::string &tag_file_name,
                                           size_t offset) {
  size_t tag_num, tag_dim;
  std::vector<TagT> tag_v;
  this->tags.clear();

  if (!file_exists(tag_file_name)) {
    LOG(INFO) << "Tags file not found. Using equal mapping";
    // Equal mapping are by default eliminated in tags map.
  } else {
    LOG(INFO) << "Load tags from existing file: " << tag_file_name;
    pipeann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
    tags.reserve(tag_v.size());
    id2loc_.reserve(tag_v.size());

#pragma omp parallel for num_threads(num_search_threads)
    for (size_t i = 0; i < tag_num; ++i) {
      tags.insert_or_assign(i, tag_v[i]);
    }
  }
  LOG(INFO) << "Loaded " << tags.size() << " tags";
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::apply_tags_to_result(std::shared_ptr<search_result_t> result) {
  if (!enable_tags)
    return;

  for (auto i = 0; i < result->num_res; i++) {
    result->node_id[i] = id2tag(result->node_id[i]);
  }
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::compute_pq_dists(const uint32_t src,
                                                  const uint32_t *ids,
                                                  float *fp_dists,
                                                  const uint32_t count,
                                                  uint8_t *aligned_scratch) {
  const uint8_t *src_ptr = this->data.data() + (this->n_chunks * src);
  if (unlikely(aligned_scratch == nullptr || count >= 32768)) {
    LOG(ERROR) << "Aligned scratch buffer is null or count is too large: "
               << count << ". This will lead to memory issues.";
    crash();
  }
  // aggregate PQ coords into scratch
  ::aggregate_coords(ids, count, this->data.data(), this->n_chunks,
                     aligned_scratch);
  // compute distances
  this->pq_table.compute_distances_alltoall(src_ptr, aligned_scratch, fp_dists,
                                            count);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::notify_client_local(
    SearchState<T, TagT> *search_state) {
  std::shared_ptr<search_result_t> result = search_state->get_search_result();
  apply_tags_to_result(result);
  // auto &full_retset = search_state->full_retset;
  // std::sort(full_retset.begin(), full_retset.end(),
  // [](const pipeann::Neighbor &left, const pipeann::Neighbor &right) {
  // return left < right;
  // });

  // uint64_t t = 0;
  // for (uint64_t i = 0; i < full_retset.size() && t <
  // search_state->k_search; i++) { if (i > 0 && full_retset[i].id ==
  // full_retset[i - 1].id) { continue;  // deduplicate.
  // }
  // search_state->res_tags[t] = full_retset[i].id;  // use ID to replace
  // tags if (search_state->res_dists != nullptr) {
  // search_state->res_dists[t] = full_retset[i].distance;
  // }
  // t++;
  // }
  for (auto i = 0; i < result->num_res; i++) {
    search_state->res_tags[i] = result->node_id[i];
    if (search_state != nullptr) {
      search_state->res_dists[i] = result->distance[i];
    }
  }
  search_state->completion_count->fetch_add(1);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::search_ssd_index_local(
    const T *query_emb, const uint64_t query_id, const uint64_t k_search,
    const uint32_t mem_L, const uint64_t l_search, TagT *res_tags,
    float *res_dists, const uint64_t beam_width,
    std::shared_ptr<std::atomic<uint64_t>> completion_count) {
  if (beam_width != 1) {
    throw std::invalid_argument("beam width has to be 1 because of the design");
  }
  assert(k_search != 0);
  // need to create a state then issue io
  SearchState<T, TagT> *new_search_state = new SearchState<T, TagT>;
  new_search_state->client_type = ClientType::LOCAL;
  new_search_state->l_search = l_search;
  new_search_state->k_search = k_search;
  new_search_state->beam_width = beam_width;
  new_search_state->res_tags = res_tags;
  new_search_state->res_dists = res_dists;
  new_search_state->completion_count = completion_count;
  new_search_state->query_id = query_id;
  new_search_state->partition_history.push_back(this->my_partition_id);

  std::shared_ptr<QueryEmbedding<T>> q = std::make_shared<QueryEmbedding<T>>();
  memcpy(q->query, query_emb, this->data_dim * sizeof(T));
  pq_table.populate_chunk_distances(q->query, q->pq_dists);
  q->query_id = query_id;
  q->dim = this->data_dim;
  q->num_chunks = this->n_chunks;

  new_search_state->query_emb = q;

  query_emb_map.insert_or_assign(query_id, q);

  // memcpy(new_search_state->query, query_emb, this->data_dim * sizeof(T));
  // float *pq_dists = new_search_state->pq_dists;
  // pq_table.populate_chunk_distances(new_search_state->query, pq_dists);
  state_reset(new_search_state);
  // uint32_t best_medoid = medoids[0];
  // state_compute_and_add_to_retset(new_search_state, &best_medoid, 1);

  // std::sort(new_search_state->retset.begin(),
  // new_search_state->retset.begin() + new_search_state->cur_list_size);
  uint64_t thread_id =
      current_search_thread_id.fetch_add(1) % num_search_threads;

#ifdef BALANCE_ALL
  void *ctx = search_threads[thread_id]->ctx;
  if (ctx == nullptr) {
    std::stringstream err;
    err << "[" << __func__ << "] tried to issue io but ctx = nullptr"
        << std::endl;
    throw std::runtime_error(err.str());
  }
  new_search_state->issue_next_io_batch(ctx);
#else
  search_threads[thread_id]->push_state(new_search_state);
#endif
}


template <typename T, typename TagT> void SSDPartitionIndex<T, TagT>::start() {
  for (uint64_t thread_id = 0; thread_id < num_search_threads; thread_id++) {
    search_threads.push_back(std::make_unique<SearchThread>(this, thread_id));
  }
  for (uint64_t thread_id = 0; thread_id < num_search_threads; thread_id++) {
    search_threads[thread_id]->start();
  }


  batching_thread = std::make_unique<BatchingThread>(this);
  batching_thread->start();
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::shutdown() {
  for (uint64_t thread_id = 0; thread_id < num_search_threads; thread_id++) {
    search_threads[thread_id]->signal_stop();
  }
  for (uint64_t thread_id = 0; thread_id < num_search_threads; thread_id++) {

    search_threads[thread_id]->join();
  }
  batching_thread->signal_stop();
  batching_thread->join();
  std::cout << "DONE WITH SHUTOWN" << std::endl;
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::load_mem_index(
    pipeann::Metric metric, const size_t query_dim,
    const std::string &mem_index_path) {
  if (mem_index_path.empty()) {
    LOG(ERROR) << "mem_index_path is needed";
    exit(1);
  }
  mem_index_ = std::make_unique<pipeann::Index<T, uint32_t>>(
							     metric, query_dim, 0, false, false, true);
  mem_index_->load(mem_index_path.c_str());
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::notify_client_tcp(
						   SearchState<T, TagT> *search_state) {
  // LOG(INFO) << "finished query " << search_state->query_id;
  // std::cout << "notify called for query " << search_state->query_id
  // << std::endl;
  Region r;
  std::shared_ptr<search_result_t> result = search_state->get_search_result();
  // LOG(INFO) << "enable tags" << enable_tags;
  apply_tags_to_result(result);
  size_t region_size =
      sizeof(MessageType::RESULT) + result->get_serialize_size();
  r.length = region_size;
  r.addr = new char[region_size];

  size_t offset = 0;
  MessageType msg_type = MessageType::RESULT;
  std::memcpy(r.addr, &msg_type, sizeof(msg_type));
  offset += sizeof(msg_type);
  result->write_serialize(r.addr + offset);
  this->communicator->send_to_peer(search_state->client_peer_id, r);
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::notify_client(
    SearchState<T, TagT> *search_state) {
  if (search_state->client_type == ClientType::LOCAL) {
    notify_client_local(search_state);
  } else if (search_state->client_type == ClientType::TCP) {
    notify_client_tcp(search_state);
  }
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::receive_handler(const char *buffer,
                                                 size_t size) {
  MessageType msg_type;
  size_t offset = 0;
  std::memcpy(&msg_type, buffer, sizeof(msg_type));
  offset += sizeof(msg_type);
  if (msg_type == MessageType::QUERIES) {
    std::vector<std::shared_ptr<QueryEmbedding<T>>> queries =
      QueryEmbedding<T>::deserialize_queries(buffer + offset, size);
    for (auto query : queries) {
      // std::cout << "received new query "<< query->query_id << std::endl;
      // assert(query->dim == this->dim);
      query->num_chunks = this->n_chunks;
      // lets check how long this takes, if it takes long then we can do it
      // lazily (ie when the search thread first accesses it
      pq_table.populate_chunk_distances(query->query, query->pq_dists);
      query_emb_map.insert_or_assign(query->query_id, query);

      SearchState<T, TagT> *new_search_state = new SearchState<T, TagT>;

      new_search_state->client_type = ClientType::TCP;
      new_search_state->mem_l = query->mem_l;
      new_search_state->l_search = query->l_search;
      new_search_state->k_search = query->k_search;
      new_search_state->beam_width = query->beam_width;
      new_search_state->query_id = query->query_id;
      new_search_state->client_peer_id = query->client_peer_id;
      new_search_state->partition_history.push_back(this->my_partition_id);
      new_search_state->query_emb = query;
      new_search_state->cur_list_size = 0;
      if (query->record_stats) {
        new_search_state->stats = std::make_shared<QueryStats>();
      }
      state_reset(new_search_state);
      // uint32_t best_medoid = medoids[0];
      // state_compute_and_add_to_retset(new_search_state, &best_medoid, 1);
      // state_print(new_search_state);
#ifdef PER_THREAD_QUEUE
      uint64_t thread_id =
          current_search_thread_id.fetch_add(1) % num_search_threads;

      search_threads[thread_id]->push_state(new_search_state);
#else
      // global_state_queue.enqueue()
      global_state_queue.enqueue(client_state_prod_token, new_search_state);
#endif
    }
  } else if (msg_type == MessageType::STATES) {
    // LOG(INFO) << "States received";
    std::vector<SearchState<T, TagT> *> states =
      SearchState<T, TagT>::deserialize_states(buffer + offset, size);
    for (auto state : states) {
      assert(state->cur_list_size > 0);
      state->partition_history.push_back(my_partition_id);
      if (state->query_emb != nullptr) {
        if (query_emb_map.contains(state->query_id)) {
          throw std::runtime_error(
              "Query embedding map contains query_id already: " +
              std::to_string(state->query_id));
        }
        pq_table.populate_chunk_distances(state->query_emb->query,
                                          state->query_emb->pq_dists);
        query_emb_map.insert_or_assign(state->query_id, state->query_emb);
      } else {
        state->query_emb = query_emb_map.find(state->query_id);
      }
      // LOG(INFO) << "=======================";
      // this->state_print_detailed(state);
      // LOG(INFO) << "=======================";      
    }
    global_state_queue.enqueue_bulk(server_state_prod_token, states.begin(),
                                    states.size());
  } else if (msg_type == MessageType::RESULT_ACK) {
    // LOG(INFO) << "ack received";
    ack a = ack::deserialize(buffer + offset);
    query_emb_map.erase(a.query_id);
  } else {
    throw std::runtime_error("Weird message type value");

  }
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::send_state(
					    SearchState<T, TagT> *search_state){ 
  uint8_t receiver_partition_id = this->get_cluster_assignment(search_state->frontier[0]);
  assert(receiver_partition_id != this->my_partition_id);
  bool send_with_embedding;

  if (std::find(search_state->partition_history.begin(),
                search_state->partition_history.end(), receiver_partition_id) !=
      search_state->partition_history.end()) {
    send_with_embedding = false;
    /* v contains x */
  } else {
    /* v does not contain x */
    send_with_embedding = true;
  }
  
  Region r;
  std::vector<std::pair<SearchState<T, TagT> *, bool>> single_state_vec = {
    {search_state, send_with_embedding}};
  size_t region_size =
      sizeof(MessageType) +
      SearchState<T, TagT>::get_serialize_size_states(single_state_vec);
  r.length = region_size;
  r.addr = new char[region_size];

  size_t offset = 0;
  MessageType msg_type = MessageType::STATES;
  std::memcpy(r.addr, &msg_type, sizeof(msg_type));
  offset += sizeof(msg_type);
  // necessary because we are de serializing based on a batch of states

  SearchState<T, TagT>::write_serialize_states(r.addr + offset,
                                               single_state_vec);
  // write_serialize(r.addr + offset, send_with_embedding);
  this->communicator->send_to_peer(receiver_partition_id, r);
}




template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::query_emb_print(
						 std::shared_ptr<QueryEmbedding<T>> query_emb) {
  LOG(INFO) << "@@@@@@@@@@@@@@@@@@@@@@@@@";
  LOG(INFO) << "Query embedding  " << query_emb->query_id;
  LOG(INFO) << "dim " << query_emb->dim;
  std::stringstream query_str;
  for (auto i = 0; i < query_emb->dim; i++) {
    query_str << query_emb->query[i] << " ";
  }
  LOG(INFO) << "query " << query_str.str();

  std::stringstream pq_dists_str;
  LOG(INFO) << "num_chunks " << query_emb->num_chunks;
  for (auto i = 0; i < query_emb->num_chunks; i++) {
    pq_dists_str << query_emb->pq_dists[i] << " ";
  }

  LOG(INFO) << "@@@@@@@@@@@@@@@@@@@@@@@@@";
    
}

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::BatchingThread::BatchingThread(
    SSDPartitionIndex<T, TagT> *parent)
: parent(parent) {
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::BatchingThread::start() {
  running = true;
  real_thread =
    std::thread(&SSDPartitionIndex<T, TagT>::BatchingThread::main_loop, this);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::BatchingThread::signal_stop() {
  std::unique_lock<std::mutex> lock(msg_queue_mutex);
  running = false;
  assert(!msg_queue.contains(std::numeric_limits<uint64_t>::max()));
  msg_queue.emplace(std::numeric_limits<uint64_t>::max(), nullptr);
  // real_thread =
  // std::thread(&SSDPartitionIndex<T, TagT>::BatchingThread::main_loop, this);
  msg_queue_cv.notify_all();
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::BatchingThread::join() {
  if (real_thread.joinable()) real_thread.join();
}




template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::BatchingThread::push_result_to_batch(
								      SearchState<T, TagT> *state) {
  uint64_t recipient_peer_id = state->client_peer_id;
  std::unique_lock<std::mutex> lock(msg_queue_mutex);
  if (!peer_client_ids.contains(recipient_peer_id)) {
    peer_client_ids.insert(recipient_peer_id);
  }
  if (!msg_queue.contains(recipient_peer_id)) {
    msg_queue[recipient_peer_id] =
      std::make_unique<std::vector<SearchState<T, TagT> *>>();
    msg_queue[recipient_peer_id]->reserve(parent->max_batch_size);
  }
  msg_queue[recipient_peer_id]->emplace_back(state);
  msg_queue_cv.notify_all();
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::BatchingThread::push_state_to_batch(
								     SearchState<T, TagT> *state) {
  uint64_t recipient_peer_id = parent->state_top_cand_partition(state);
  std::unique_lock<std::mutex> lock(msg_queue_mutex);
  if (!msg_queue.contains(recipient_peer_id)) {
    msg_queue[recipient_peer_id] =
      std::make_unique<std::vector<SearchState<T, TagT> *>>();
    msg_queue[recipient_peer_id]->reserve(parent->max_batch_size);
  }
  msg_queue[recipient_peer_id]->emplace_back(state);
  msg_queue_cv.notify_all();
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::BatchingThread::main_loop() {
  // auto batch_time = std::chrono::microseconds(parent->batch_time_us);

  std::unique_lock<std::mutex> lock(msg_queue_mutex, std::defer_lock);
  // std::unordered_map<uint64_t, std::chrono::steady_clock::time_point>
  // wait_time_msgs;

  auto msg_queue_empty = [this]() {
    for (const auto &[peer_id, states] : this->msg_queue) {
      if (!states->empty())
        return false;
    }
    return true;
  };

  auto should_send_emb = [](SearchState<T, TagT> *state,
                            uint64_t server_peer_id) {
    return std::find(state->partition_history.cbegin(),
                     state->partition_history.cend(),
                     static_cast<uint8_t>(server_peer_id)) ==
           state->partition_history.cend();
  };
  
  while (running) {
    lock.lock();
    while (msg_queue_empty()) {
      msg_queue_cv.wait(lock);
    }
    if (msg_queue.contains(std::numeric_limits<uint64_t>::max())) {
      assert(!running);
      break;
    }
    
    if (!running)
      break;

    std::unordered_map<uint64_t,
                       std::unique_ptr<std::vector<SearchState<T, TagT> *>>>
    states_to_send;


    std::unordered_map<uint64_t,
                       std::unique_ptr<std::vector<SearchState<T, TagT> *>>>
    results_to_send;

    
    for (auto &[peer_id, msgs] : msg_queue) {
      if (peer_client_ids.contains(peer_id)) {
        results_to_send[peer_id] = std::move(msgs);
      } else {
        states_to_send[peer_id] = std::move(msgs);
      }
      msg_queue[peer_id] =
        std::make_unique<std::vector<SearchState<T, TagT> *>>();
      msg_queue[peer_id]->reserve(parent->max_batch_size);
    }
    lock.unlock();

    for (auto &[client_peer_id, states] : results_to_send) {
      uint64_t num_sent = 0;
      uint64_t total = states->size();

      while (num_sent < total) {
        uint64_t left = total - num_sent;
        uint64_t batch_size = std::min(parent->max_batch_size, left);
        Region r;
        std::vector<std::shared_ptr<search_result_t>> results;
        results.reserve(parent->max_batch_size);
        for (uint64_t i = num_sent; i < num_sent + batch_size; i++) {
          results.emplace_back(states->at(i)->get_search_result());
        }

	r.length = sizeof(MessageType) +
                   search_result_t::get_serialize_results_size(results);
	r.addr = new char[r.length];
	size_t offset = 0;
	MessageType msg_type = MessageType::RESULTS;
	std::memcpy(r.addr + offset, &msg_type, sizeof(msg_type));
	offset += sizeof(msg_type);
	search_result_t::write_serialize_results(r.addr + offset, results);
	parent->communicator->send_to_peer(client_peer_id, r);
        num_sent += batch_size;
      }
      for (auto &state : *states)
        delete state;      
    }

    // for (auto &[client_peer_id, states] : results_to_send) {
    //   Region r;
    //   std::vector<std::shared_ptr<search_result_t>> results;
    //   for (const auto &state : *states) {
    //     results.emplace_back(state->get_search_result());
    //   }
    //   r.length = sizeof(MessageType) +
    //              search_result_t::get_serialize_results_size(results);
    //   r.addr = new char[r.length];
    //   size_t offset = 0;
    //   MessageType msg_type = MessageType::RESULT;
    //   std::memcpy(r.addr + offset, &msg_type, sizeof(msg_type));
    //   offset += sizeof(msg_type);
    //   search_result_t::write_serialize_results(r.addr + offset, results);
    //   parent->communicator->send_to_peer(client_peer_id, r);

    //   for (auto &state : *states)
    //     delete state;
    // }



    for (auto &[server_peer_id, states] : states_to_send) {
      uint64_t num_sent = 0;
      uint64_t total = states->size();


      while (num_sent < total) {
        uint64_t left = total - num_sent;
        uint64_t batch_size = std::min(parent->max_batch_size, left);
        Region r;
        std::vector<std::pair<SearchState<T, TagT>*, bool>> state_batch;
        state_batch.reserve(parent->max_batch_size);
        for (uint64_t i = num_sent; i < num_sent + batch_size; i++) {
          state_batch.emplace_back(
				   states->at(i), should_send_emb(states->at(i), server_peer_id));
        }

        MessageType msg_type = MessageType::STATES;
	r.length = sizeof(MessageType) +
                   SearchState<T, TagT>::get_serialize_size_states(state_batch);
	r.addr = new char[r.length];
	size_t offset = 0;
	std::memcpy(r.addr, &msg_type, sizeof(msg_type));
	offset += sizeof(msg_type);
	SearchState<T, TagT>::write_serialize_states(r.addr + offset, state_batch);
	parent->communicator->send_to_peer(server_peer_id, r);
        num_sent += batch_size;
      }
      for (auto &state : *states)
        delete state;      
    }
    // for (auto &[server_peer_id, states] : states_to_send) {
    //   Region r;
    //   r.length = sizeof(MessageType::STATES);
    //   std::vector<std::pair<SearchState<T, TagT>*, bool>> state_arr; 
    //   for (const auto &state : *states) {
    //     state_arr.emplace_back(state, should_send_emb(state, server_peer_id));
    //   }
    //   r.length += SearchState<T, TagT>::get_serialize_size_states(state_arr);
    //   r.addr = new char[r.length];
    //   MessageType msg_type = MessageType::STATES;

    //   size_t offset = 0;
    //   std::memcpy(r.addr, &msg_type, sizeof(msg_type));
    //   offset += sizeof(msg_type);
    //   SearchState<T, TagT>::write_serialize_states(r.addr + offset, state_arr);
    //   parent->communicator->send_to_peer(server_peer_id, r);
    //   for (auto &state: *states) delete state;
    // }
    
  }
}





template class SSDPartitionIndex<float, uint32_t>;
template class SSDPartitionIndex<uint8_t, uint32_t>;
template class SSDPartitionIndex<int8_t, uint32_t>;
