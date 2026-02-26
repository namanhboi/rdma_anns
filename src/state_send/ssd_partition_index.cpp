#include "ssd_partition_index.h"
#include "communicator.h"
#include "disk_utils.h"
#include "query_buf.h"
#include "singleton_logger.h"
#include "types.h"
#include "utils.h"
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::SSDPartitionIndex(
    pipeann::Metric m, uint8_t partition_id, uint32_t num_worker_threads,
    std::shared_ptr<AlignedFileReader> &fileReader,
    std::unique_ptr<P2PCommunicator> &communicator,
    DistributedSearchMode dist_search_mode,
    pipeann::IndexBuildParameters *params, uint64_t num_queries_balance,
    bool use_batching, uint64_t max_batch_size, bool use_counter_thread,
    std::string counter_csv, uint64_t counter_sleep_ms, bool use_logging,
    const std::string &log_file)
    : reader(fileReader), communicator(communicator),
      client_state_prod_token(global_state_queue),
      server_state_prod_token(global_state_queue),
      distributed_ann_head_index_ptok(distributed_ann_task_queue),
      distributed_ann_scoring_ptok(distributed_ann_task_queue),
      dist_search_mode(dist_search_mode), max_batch_size(max_batch_size),
      use_batching(use_batching), use_counter_thread(use_counter_thread),
      pq_table(m), metric(m) {

  if (dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
    prealloc_distributedann_result =
        PreallocatedQueue<distributedann::result_t<T>>(
            MAX_PRE_ALLOC_ELEMENTS, distributedann::result_t<T>::reset);
    prealloc_distributedann_scoring_query =
        PreallocatedQueue<distributedann::scoring_query_t<T>>(
            MAX_PRE_ALLOC_ELEMENTS, distributedann::scoring_query_t<T>::reset);
    preallocated_query_emb_queue = PreallocatedQueue<QueryEmbedding<T>>(
        MAX_PRE_ALLOC_ELEMENTS, QueryEmbedding<T>::reset);

    LOG(INFO) << "Allocated "
              << prealloc_distributedann_result.get_num_elements() *
                     sizeof(distributedann::result_t<T>)
              << " bytes for distributedann results";

    LOG(INFO) << "Allocated "
              << prealloc_distributedann_scoring_query.get_num_elements() *
                     sizeof(distributedann::scoring_query_t<T>)
              << " bytes for distributedann scoring query";

    LOG(INFO) << "Allocated "
              << preallocated_query_emb_queue.get_num_elements() *
                     sizeof(QueryEmbedding<T>)
              << " bytes for queries";
  } else {
    preallocated_state_queue = PreallocatedQueue<SearchState<T, TagT>>(
        MAX_PRE_ALLOC_ELEMENTS, SearchState<T, TagT>::reset);
    preallocated_query_emb_queue = PreallocatedQueue<QueryEmbedding<T>>(
        MAX_PRE_ALLOC_ELEMENTS, QueryEmbedding<T>::reset);
    // preallocated_result_queue = PreallocatedQueue<search_result_t>(
    // MAX_PRE_ALLOC_ELEMENTS, search_result_t::reset);

    LOG(INFO) << "Allocated "
              << preallocated_state_queue.get_num_elements() *
                     sizeof(SearchState<T, TagT>)
              << " bytes for states";
    LOG(INFO) << "Allocated "
              << preallocated_query_emb_queue.get_num_elements() *
                     sizeof(QueryEmbedding<T>)
              << " bytes for queries";
  }

  use_logging = use_logging;
  // = spdlog::basic_logger_mt("logger", log_file);
  // logger->set_pattern("%v");
  if (use_logging) {
    SingletonLogger::get_logger(log_file, spdlog::level::info);
    // logger->set_level(spdlog::level::info);
    //
  } else {
    SingletonLogger::get_logger(log_file, spdlog::level::off);
  }

  LOG(INFO) << "DIST SEARCH MODE IS " << (int)dist_search_mode;
  if (use_batching && max_batch_size == 0)
    throw std::runtime_error("max_batch_size can't be 0 if we use batching");

  if (num_queries_balance > max_queries_balance) {
    throw std::invalid_argument("number of queries to balance too big " +
                                std::to_string(num_queries_balance));
  }
  this->num_queries_balance = num_queries_balance;
  this->my_partition_id = partition_id;
  this->num_worker_threads = num_worker_threads;
  if (num_worker_threads > MAX_WORKER_THREADS) {
    throw std::invalid_argument("num search threads > MAX_SEARCH_THREADS");
  }

  data_is_normalized = false;
  // this->enable_tags = tags;
  // this->enable_locs = enable_locs;
  // if (m == pipeann::Metric::COSINE) {
  //   if (std::is_floating_point<T>::value) {
  //     LOG(INFO) << "Cosine metric chosen for (normalized) float data."
  //                  "Changing distance to L2 to boost accuracy.";
  //     m = pipeann::Metric::L2;
  //     data_is_normalized = true;
  //   } else {
  //     LOG(ERROR) << "WARNING: Cannot normalize integral data types."
  //                << " This may result in erroneous results or poor recall."
  //                << " Consider using L2 distance with integral data types.";
  //   }
  // }

  this->dist_cmp.reset(pipeann::get_distance_function<T>(m));
  // this->pq_reader = new LinuxAlignedFileReader();
  if (params != nullptr) {
    this->beamwidth = params->beam_width;
    this->l_index = params->L;
    this->range = params->R;
    this->maxc = params->C;
    this->alpha = params->alpha;
    LOG(INFO) << "Beamwidth: " << this->beamwidth << ", L: " << this->l_index
              << ", R: " << this->range << ", C: " << this->maxc;
  }

  if (this->dist_search_mode == DistributedSearchMode::STATE_SEND ||
      this->dist_search_mode ==
          DistributedSearchMode::STATE_SEND_CLIENT_GATHER) {
    this->enable_locs = true;
  } else if (this->dist_search_mode == DistributedSearchMode::SCATTER_GATHER) {
    this->enable_tags = true;
  } else if (this->dist_search_mode == DistributedSearchMode::SINGLE_SERVER) {
    this->enable_tags = false;
    this->enable_locs = false;
  } else if (this->dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
    this->enable_tags = false;
    this->enable_locs = true;
  }

  if (dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
    for (uint64_t thread_id = 0; thread_id < num_worker_threads; thread_id++) {
      distributedann_worker_threads.emplace_back(
          std::make_unique<DistributedANNWorkerThread>(this));
    }
    if (use_batching == false) {
      throw std::invalid_argument("For distributedann, has to use batching");
    }
    distributedann_batching_thread =
        std::make_unique<DistributedANNBatchingThread>(this);
  } else {
    LOG(INFO) << "starting threads " << num_worker_threads;
    for (uint64_t thread_id = 0; thread_id < num_worker_threads; thread_id++) {
      search_threads.emplace_back(
          std::make_unique<SearchThread>(this, thread_id));
    }
    if (use_batching) {
      batching_thread = std::make_unique<BatchingThread>(this);
    }
  }
  if (use_counter_thread) {
    counter_thread =
        std::make_unique<CounterThread>(this, counter_csv, counter_sleep_ms);
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
                                     bool new_index_format) {
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
  std::cout << "max_degree is " << max_degree << std::endl;
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

  if (dist_search_mode == DistributedSearchMode::SINGLE_SERVER &&
      disk_nnodes != npts_u64) {
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
  // loading the id2loc file
  LOG(INFO) << "enable locs : " << enable_locs;
  if (enable_locs) {
    LOG(INFO) << "enabled locs";
    std::string id2loc_file = iprefix + "_ids_uint32.bin";
    if (!file_exists(id2loc_file)) {
      throw std::invalid_argument(
          "dist search mode is  " +
          dist_search_mode_to_string(dist_search_mode) +
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
  if (dist_search_mode == DistributedSearchMode::STATE_SEND ||
      dist_search_mode == DistributedSearchMode::STATE_SEND_CLIENT_GATHER) {
    std::string cluster_file = iprefix + "_partition_assignment.bin";
    ;
    // std::string cluster_file(cluster_assignment_file);
    if (!file_exists(cluster_file)) {
      throw std::invalid_argument(
          "dist search omde is   " +
          dist_search_mode_to_string(dist_search_mode) +
          ", but the cluster assignment bin file doesn't exist: " +
          cluster_file);
    }
    // for testing right now to see what the overhead is of nodes with multiple
    // clusters as home cluster
    size_t ca_num_pts, ca_dim;
    std::vector<uint8_t> tmp;
    pipeann::load_bin<uint8_t>(cluster_file, partition_assignment, ca_num_pts,
                               ca_dim);

    // for (const auto &node_id : tmp) {
    // partition_assignment.push_back({node_id});
    // }
    // uint8_t num_partitions;
    // load_partition_assignment_file(cluster_file, partition_assignment,
    //                                num_partitions);
    // for (auto &assignment : partition_assignment) {
    //   auto it =
    //     std::find(assignment.begin(), assignment.end(), my_partition_id);
    //   if (it != assignment.end()) {
    // 	assignment = {my_partition_id};
    //   }
    // }
    LOG(INFO) << "cluster assignment file loaded successfully.";
  }

  std::string norm_file = std::string(_disk_index_file) + "_max_base_norm.bin";
  if (this->metric == pipeann::Metric::INNER_PRODUCT) {
    if (file_exists(norm_file)) {
      uint64_t dumr, dumc;
      float *norm_val;
      pipeann::load_bin(norm_file, norm_val, dumr, dumc);
      this->_max_base_norm = norm_val[0];
      LOG(INFO) << "Setting rescaling factor of base vector to "
                << this->_max_base_norm;
      delete[] norm_val;
    } else {
      throw std::runtime_error(
          "distance metric is mips but max norm base  file doesn't exist");
    }
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

#pragma omp parallel for num_threads(num_worker_threads)
    for (size_t i = 0; i < tag_num; ++i) {
      tags.insert_or_assign(i, tag_v[i]);
    }
  }
  LOG(INFO) << "Loaded " << tags.size() << " tags";
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::apply_tags_to_result(uint32_t *node_id,
                                                      uint64_t num_res) {
  if (!enable_tags) {
    // LOG(INFO) << "tag not enabled";
    return;
  }
  for (auto i = 0; i < num_res; i++) {
    node_id[i] = id2tag(node_id[i]);
  }
}

template <typename T, typename TagT> void SSDPartitionIndex<T, TagT>::start() {
  if (dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
    for (uint64_t thread_id = 0; thread_id < num_worker_threads; thread_id++) {
      distributedann_worker_threads[thread_id]->start();
      LOG(INFO) << "STARTED THREAD " << thread_id;
    }
    distributedann_batching_thread->start();
  } else {
    for (uint64_t thread_id = 0; thread_id < num_worker_threads; thread_id++) {
      search_threads[thread_id]->start();
    }
    if (use_batching) {
      batching_thread->start();
    }
  }
  if (use_counter_thread) {
    counter_thread->start();
  }
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::shutdown() {
  LOG(INFO) << "SHUTDOWN CALLED";
  if (dist_search_mode == DistributedSearchMode::DISTRIBUTED_ANN) {
    for (uint64_t thread_id = 0; thread_id < num_worker_threads; thread_id++) {
      distributedann_worker_threads[thread_id]->signal_stop();
    }
    for (uint64_t thread_id = 0; thread_id < num_worker_threads; thread_id++) {
      distributedann_worker_threads[thread_id]->join();
    }
    distributedann_batching_thread->signal_stop();
    distributedann_batching_thread->join();
  } else {
    for (uint64_t thread_id = 0; thread_id < num_worker_threads; thread_id++) {
      search_threads[thread_id]->signal_stop();
    }
    for (uint64_t thread_id = 0; thread_id < num_worker_threads; thread_id++) {
      search_threads[thread_id]->join();
    }
    if (use_batching) {
      batching_thread->signal_stop();
      batching_thread->join();
    }
  }
  if (use_counter_thread) {
    counter_thread->signal_stop();
    counter_thread->join();
  }
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
  // pipeann::Metric mem_metric =
  // pipeann::Metric::L2; // Inner product also uses l2 because we normalized
  // the mips data into l2

  // if (metric == pipeann::Metric::INNER_PRODUCT) {

  // }
  LOG(INFO) << "query_dim is " << query_dim;
  mem_index_ = std::make_unique<pipeann::Index<T, uint32_t>>(metric, query_dim);
  mem_index_->load(mem_index_path.c_str());
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::notify_client_tcp(
    SearchState<T, TagT> *search_state) {
  if (use_batching) {
    batching_thread->push_result_to_batch(search_state);
    return;
  }
  throw std::runtime_error(
      "Need to recheck implementation of non-batched sending");

  // Region r;
  // std::shared_ptr<search_result_t> result =
  // search_state->get_search_result();
  // // LOG(INFO) << "enable tags" << enable_tags;
  // apply_tags_to_result(result);
  // size_t region_size =
  //     sizeof(MessageType::RESULT) + result->get_serialize_size();
  // r.length = region_size;
  // r.addr = new char[region_size];

  // size_t offset = 0;
  // MessageType msg_type = MessageType::RESULT;
  // std::memcpy(r.addr, &msg_type, sizeof(msg_type));
  // offset += sizeof(msg_type);
  // result->write_serialize(r.addr + offset);
  // this->communicator->send_to_peer(search_state->client_peer_id, r);
  // delete search_state;
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::notify_client(
    SearchState<T, TagT> *search_state) {
  if (dist_search_mode == DistributedSearchMode::SCATTER_GATHER) {
    QueryEmbedding<T> *query = query_emb_map.find(search_state->query_id);
    preallocated_query_emb_queue.free(query);
    query_emb_map.erase(search_state->query_id);
  }
  if (search_state->client_type == ClientType::TCP) {
    notify_client_tcp(search_state);
  } else {
    throw std::invalid_argument("Weird client type value ");
  }
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::receive_handler(const char *buffer,
                                                 size_t size) {
  uint64_t msg_id = msg_received_id.fetch_add(1);

  MessageType msg_type;
  size_t offset = 0;
  std::memcpy(&msg_type, buffer, sizeof(msg_type));
  offset += sizeof(msg_type);
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_HANDLER",
                                     get_timestamp_ns(), msg_id,
                                     message_type_to_string(msg_type));
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:MSG_SIZE {}",
                                     get_timestamp_ns(), msg_id,
                                     message_type_to_string(msg_type), size);

  if (msg_type == MessageType::QUERIES) {
    size_t num_queries;
    std::memcpy(&num_queries, buffer + offset, sizeof(num_queries));
    offset += sizeof(num_queries);
    preallocated_query_emb_queue.dequeue_exact(num_queries,
                                               query_scratch.data());

    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_DESERIALIZE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
    QueryEmbedding<T>::deserialize_queries(buffer + offset, num_queries,
                                           query_scratch.data());
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));

    for (uint64_t i = 0; i < num_queries; i++) {
      QueryEmbedding<T> *query = query_scratch[i];
      query->query[query->dim] =
          0; // this is for mips, to ensure that d + 1 is zero as well

      query->num_chunks = this->n_chunks;
      // lets check how long this takes, if it takes long then we can do it
      // lazily (ie when the search thread first accesses it
      // SingletonLogger::get_logger().info("[{}] [{}]
      // [{}]:BEGIN_QUERY_MAP_INSERT", get_timestamp_ns(), msg_id,
      // message_type_to_string(msg_type));
      query_emb_map.insert_or_assign(query->query_id, query);
      // SingletonLogger::get_logger().info("[{}] [{}]
      // [{}]:END_QUERY_MAP_INSERT", get_timestamp_ns(), msg_id,
      // message_type_to_string(msg_type));

      // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_CREATE_STATE",
      // get_timestamp_ns(), msg_id,
      // message_type_to_string(msg_type));
      preallocated_state_queue.dequeue_exact(1, state_scratch.data());
      SearchState<T, TagT> *new_search_state = state_scratch[0];
      // SearchState<T, TagT> *new_search_state = new SearchState<T, TagT>;
      // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_CREATE_STATE",
      // get_timestamp_ns(), msg_id,
      // message_type_to_string(msg_type));
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
      if (new_search_state->stats) {
        new_search_state->partition_history_hop_idx.push_back(
            new_search_state->stats->n_hops);
      }

      num_new_states_global_queue.fetch_add(1);
      // SingletonLogger::get_logger().info("[{}] [{}]
      // [{}]:BEGIN_ENQUEUE_STATE", get_timestamp_ns(), msg_id,
      // message_type_to_string(msg_type));
      global_state_queue.enqueue(client_state_prod_token, new_search_state);
      // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_ENQUEUE_STATE",
      // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
    }
  } else if (msg_type == MessageType::STATES) {
    // LOG(INFO) << size;
    size_t num_states, num_queries;
    std::memcpy(&num_states, buffer + offset, sizeof(num_states));
    offset += sizeof(num_states);
    std::memcpy(&num_queries, buffer + offset, sizeof(num_queries));
    offset += sizeof(num_queries);
    preallocated_state_queue.dequeue_exact(num_states, state_scratch.data());
    if (num_queries > 0) {
      preallocated_query_emb_queue.dequeue_exact(num_queries,
                                                 query_scratch.data());
    }

    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_DESERIALIZE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
    SearchState<T, TagT>::deserialize_states(buffer + offset, num_states,
                                             num_queries, state_scratch.data(),
                                             query_scratch.data());
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
    // LOG(INFO) << "States received " << states.size();
    // SingletonLogger::get_logger().info(
    // "[{}] [{}] [{}]:NUM_MSG {}", get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type), num_states);

    // SingletonLogger::get_logger().info("[{}] [{}]
    // [{}]:BEGIN_QUERY_MAP_INSERT", get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));
    for (uint64_t i = 0; i < num_queries; i++) {
      if (query_emb_map.contains(query_scratch[i]->query_id)) {
        throw std::runtime_error(
            "Query emb map already contains embedding for query id " +
            std::to_string(query_scratch[i]->query_id));
      }
      query_emb_map.insert_or_assign(query_scratch[i]->query_id,
                                     query_scratch[i]);
    }
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_QUERY_MAP_INSERT",
    // get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));
    for (uint64_t i = 0; i < num_states; i++) {
      state_scratch[i]->partition_history.push_back(my_partition_id);
      if (state_scratch[i]->stats) {
        state_scratch[i]->partition_history_hop_idx.push_back(
            state_scratch[i]->stats->n_hops);
        state_scratch[i]->stats->n_inter_partition_hops++;
      }
    }
    num_foreign_states_global_queue.fetch_add(num_states);
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_ENQUEUE_STATE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
    global_state_queue.enqueue_bulk(server_state_prod_token,
                                    state_scratch.data(), num_states);
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_ENQUEUE_STATE",
    // get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));
  } else if (msg_type == MessageType::RESULT_ACK) {
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:NUM_MSG {}",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type), 1);
    // LOG(INFO) << "ack received";
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_DESERIALIZE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
    // ack a = ack::deserialize(buffer + offset);
    uint64_t query_id;
    std::memcpy(&query_id, buffer + offset, sizeof(query_id));
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
    // SingletonLogger::get_logger().info("[{}] [{}]
    // [{}]:BEGIN_QUERY_MAP_ERASE", get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));
    QueryEmbedding<T> *query = query_emb_map.find(query_id);
    preallocated_query_emb_queue.free(query);

    query_emb_map.erase(query_id);
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_QUERY_MAP_ERASE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
  } else {
    throw std::runtime_error("Weird message type value");
  }
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_HANDLER",
                                     get_timestamp_ns(), msg_id,
                                     message_type_to_string(msg_type));
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::distributed_ann_receive_handler(
    const char *buffer, size_t size) {
  uint64_t msg_id = msg_received_id.fetch_add(1);
  MessageType msg_type;
  size_t offset = 0;
  std::memcpy(&msg_type, buffer, sizeof(msg_type));
  offset += sizeof(msg_type);
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_HANDLER",
                                     get_timestamp_ns(), msg_id,
                                     message_type_to_string(msg_type));

  SingletonLogger::get_logger().info("[{}] [{}] [{}]:MSG_SIZE {}",
                                     get_timestamp_ns(), msg_id,
                                     message_type_to_string(msg_type), size);

  if (msg_type == MessageType::QUERIES) {
    // LOG(INFO) << " RECEIVED HEAD INDEX QUERY";
    size_t num_queries;
    std::memcpy(&num_queries, buffer + offset, sizeof(num_queries));
    offset += sizeof(num_queries);
    preallocated_query_emb_queue.dequeue_exact(num_queries,
                                               query_scratch.data());

    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_DESERIALIZE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
    QueryEmbedding<T>::deserialize_queries(buffer + offset, num_queries,
                                           query_scratch.data());
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE",
    // get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));

    for (uint64_t i = 0; i < num_queries; i++) {
      QueryEmbedding<T> *query = query_scratch[i];
      // std::cout << "received new query "<< query->query_id << std::endl;
      // assert(query->dim == this->dim);
      query->num_chunks = this->n_chunks;
      // lets check how long this takes, if it takes long then we can do it
      // lazily (ie when the search thread first accesses it
      // SingletonLogger::get_logger().info("[{}] [{}]
      // [{}]:BEGIN_QUERY_MAP_INSERT", get_timestamp_ns(), msg_id,
      // message_type_to_string(msg_type));
      query_emb_map.insert_or_assign(query->query_id, query);
      // SingletonLogger::get_logger().info("[{}] [{}]
      // [{}]:END_QUERY_MAP_INSERT", get_timestamp_ns(), msg_id,
      // message_type_to_string(msg_type));
      // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_CREATE_STATE",
      // get_timestamp_ns(), msg_id,
      // message_type_to_string(msg_type));
      distributed_ann_task_scratch[i] = {
          distributedann::DistributedANNTaskType::HEAD_INDEX, query};
    }

    distributed_ann_task_queue.enqueue_bulk(distributed_ann_head_index_ptok,
                                            distributed_ann_task_scratch.data(),
                                            num_queries);

  } else if (msg_type == MessageType::SCORING_QUERIES) {
    // LOG(INFO) << " RECEIVED SCORING QUERY QUERY";
    size_t num_scoring_queries, num_query_embs;
    std::memcpy(&num_scoring_queries, buffer + offset,
                sizeof(num_scoring_queries));
    offset += sizeof(num_scoring_queries);
    std::memcpy(&num_query_embs, buffer + offset, sizeof(num_query_embs));
    offset += sizeof(num_query_embs);
    prealloc_distributedann_scoring_query.dequeue_exact(
        num_scoring_queries, scoring_query_scratch.data());
    if (num_query_embs > 0) {
      preallocated_query_emb_queue.dequeue_exact(num_query_embs,
                                                 query_scratch.data());
    }

    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_DESERIALIZE",
    // get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));
    distributedann::scoring_query_t<T>::deserialize_scoring_queries(
        buffer + offset, num_scoring_queries, num_query_embs,
        scoring_query_scratch.data(), query_scratch.data());
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE",
    // get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));

    // SingletonLogger::get_logger().info("[{}] [{}]
    // [{}]:BEGIN_QUERY_MAP_INSERT", get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));
    for (uint64_t i = 0; i < num_query_embs; i++) {
      if (query_emb_map.contains(query_scratch[i]->query_id)) {
        throw std::runtime_error(
            "Query emb map already contains embedding for query id " +
            std::to_string(query_scratch[i]->query_id));
      }
      query_emb_map.insert_or_assign(query_scratch[i]->query_id,
                                     query_scratch[i]);
    }
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_QUERY_MAP_INSERT",
    // get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));

    // SingletonLogger::get_logger().info(
    // "[{}] [{}] [{}]:BEGIN_ENQUEUE_SCORING_QUERY", get_timestamp_ns(),
    // msg_id, message_type_to_string(msg_type));

    for (auto i = 0; i < num_scoring_queries; i++) {
      distributed_ann_task_scratch[i] = {
          distributedann::DistributedANNTaskType::SCORING_QUERY,
          scoring_query_scratch[i]};
    }
    distributed_ann_task_queue.enqueue_bulk(distributed_ann_scoring_ptok,
                                            distributed_ann_task_scratch.data(),
                                            num_scoring_queries);
    // SingletonLogger::get_logger().info("[{}] [{}]
    // [{}]:END_ENQUEUE_SCORING_QUERY", get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));
  } else if (msg_type == MessageType::RESULT_ACK) {
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:NUM_MSG {}",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type), 1);
    // LOG(INFO) << "ack received";
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:BEGIN_DESERIALIZE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
    // ack a = ack::deserialize(buffer + offset);
    uint64_t query_id;
    std::memcpy(&query_id, buffer + offset, sizeof(query_id));
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_DESERIALIZE",
    // get_timestamp_ns(), msg_id, message_type_to_string(msg_type));
    // SingletonLogger::get_logger().info("[{}] [{}]
    // [{}]:BEGIN_QUERY_MAP_ERASE", get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));
    QueryEmbedding<T> *query = query_emb_map.find(query_id);
    preallocated_query_emb_queue.free(query);

    query_emb_map.erase(query_id);
    // SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_QUERY_MAP_ERASE",
    // get_timestamp_ns(), msg_id,
    // message_type_to_string(msg_type));
  } else {
    throw std::runtime_error("Weird msg_type value " +
                             message_type_to_string(msg_type));
  }
  SingletonLogger::get_logger().info("[{}] [{}] [{}]:END_HANDLER",
                                     get_timestamp_ns(), msg_id,
                                     message_type_to_string(msg_type));
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::send_state(
    SearchState<T, TagT> *search_state) {
  if (use_batching) {
    batching_thread->push_state_to_batch(search_state);
    return;
  }
  throw std::runtime_error("Non batching impl need to be revised");
  // uint8_t receiver_partition_id =
  //     this->get_partition_assignment(search_state->frontier[0]);
  // assert(receiver_partition_id != this->my_partition_id);
  // bool send_with_embedding;

  // if (std::find(search_state->partition_history.begin(),
  //               search_state->partition_history.end(), receiver_partition_id)
  //               !=
  //     search_state->partition_history.end()) {
  //   send_with_embedding = false;
  //   /* v contains x */
  // } else {
  //   /* v does not contain x */
  //   send_with_embedding = true;
  // }

  // Region r;
  // std::vector<std::pair<SearchState<T, TagT> *, bool>> single_state_vec = {
  //     {search_state, send_with_embedding}};
  // size_t region_size =
  //     sizeof(MessageType) +
  //     SearchState<T, TagT>::get_serialize_size_states(single_state_vec);
  // r.length = region_size;
  // r.addr = new char[region_size];

  // size_t offset = 0;
  // MessageType msg_type = MessageType::STATES;
  // std::memcpy(r.addr, &msg_type, sizeof(msg_type));
  // offset += sizeof(msg_type);
  // // necessary because we are de serializing based on a batch of states

  // SearchState<T, TagT>::write_serialize_states(r.addr + offset,
  //                                              single_state_vec);
  // // write_serialize(r.addr + offset, send_with_embedding);
  // this->communicator->send_to_peer(receiver_partition_id, r);
  // delete search_state;
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
    : parent(parent),
      preallocated_region_queue(Region::MAX_PRE_ALLOC_ELEMENTS, Region::reset) {
  preallocated_region_queue.allocate_and_assign_additional_block(
      Region::MAX_BYTES_REGION, Region::assign_addr);
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
  auto empty_vec = std::make_unique<std::vector<SearchState<T, TagT> *>>();
  empty_vec->push_back(nullptr);
  msg_queue.emplace(std::numeric_limits<uint64_t>::max(), std::move(empty_vec));
  // real_thread =
  // std::thread(&SSDPartitionIndex<T, TagT>::BatchingThread::main_loop, this);
  msg_queue_cv.notify_all();
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::BatchingThread::join() {
  if (real_thread.joinable())
    real_thread.join();
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::BatchingThread::push_result_to_batch(
    SearchState<T, TagT> *state) {
  // LOG(INFO) << "push result called";
  std::unique_lock<std::mutex> lock(msg_queue_mutex);
  uint64_t recipient_peer_id = state->client_peer_id;

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
  // LOG(INFO) << "push state to batch called";
  std::unique_lock<std::mutex> lock(msg_queue_mutex);
  uint64_t recipient_peer_id = parent->state_top_cand_random_partition(state);
  state->should_send_emb =
      parent->state_should_send_emb(state, recipient_peer_id);
  if (recipient_peer_id == parent->my_partition_id) {
    throw std::runtime_error(
        "if we are sending a state then its top cand partition id can't be "
        "from this server otherwise why send it");
  }

  if (!msg_queue.contains(recipient_peer_id)) {
    msg_queue[recipient_peer_id] =
        std::make_unique<std::vector<SearchState<T, TagT> *>>();
    msg_queue[recipient_peer_id]->reserve(parent->max_batch_size);
  }
  msg_queue[recipient_peer_id]->emplace_back(state);
  if (parent->dist_search_mode ==
      DistributedSearchMode::STATE_SEND_CLIENT_GATHER) {
    // LOG(INFO) << "sending result as well as state";
    if (!peer_client_ids.contains(state->client_peer_id)) {
      peer_client_ids.insert(state->client_peer_id);
    }
    if (!msg_queue.contains(state->client_peer_id)) {
      msg_queue[state->client_peer_id] =
          std::make_unique<std::vector<SearchState<T, TagT> *>>();
      msg_queue[state->client_peer_id]->reserve(parent->max_batch_size);
    }
    msg_queue[state->client_peer_id]->emplace_back(state);
  }
  msg_queue_cv.notify_all();
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::BatchingThread::main_loop() {
  std::unique_lock<std::mutex> lock(msg_queue_mutex, std::defer_lock);

  auto msg_queue_empty = [this]() {
    for (const auto &[peer_id, states] : this->msg_queue) {
      if (!states->empty())
        return false;
    }
    return true;
  };

  std::unordered_set<SearchState<T, TagT> *> states_used;
  constexpr int max_num_servers = 16;
  // std::vector<PreallocRegion> prealloced_regions(max_num_servers);

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
        size_t msg_size =
            sizeof(MessageType) + parent->states_get_serialize_result_sizes(
									    states->data() + num_sent, batch_size);
        Region *prealloc_r;
        preallocated_region_queue.dequeue_exact(1, &prealloc_r);
        prealloc_r->length = msg_size;
        if (unlikely(prealloc_r->length >= Region::MAX_BYTES_REGION)) {
          std::stringstream error;
          error << "trying to write result that is bigger than region buffer: "
                << "batch size  is" << batch_size << ", size of total msg "
          << msg_size;
          throw std::runtime_error(
				   error.str());
        }

        // Region r;
        // size_t msg_size = parent->states_get_serialize_result_sizes(
        //     states->data() + num_sent, batch_size);
        // r.length = sizeof(MessageType) + msg_size;
        // r.addr = new char[r.length];
        size_t offset = 0;
        MessageType msg_type = MessageType::RESULTS;
        std::memcpy(prealloc_r->addr + offset, &msg_type, sizeof(msg_type));
        offset += sizeof(msg_type);
        parent->states_write_results(states->data() + num_sent, batch_size,
                                     prealloc_r->addr + offset);
        parent->communicator->send_to_peer(client_peer_id, prealloc_r);
        num_sent += batch_size;
      }
      states_used.insert(states->begin(), states->end());

      // for (auto &state : *states) {
      //   if (!state->need_to_send_result_when_send_state && state->sent_state)
      //   {
      //     parent->preallocated_state_queue.free(state);
      //   }
      // }
    }

    for (auto &[server_peer_id, states] : states_to_send) {
      uint64_t num_sent = 0;
      uint64_t total = states->size();

      while (num_sent < total) {
        uint64_t left = total - num_sent;
        uint64_t batch_size = std::min(parent->max_batch_size, left);
        Region *prealloc_r;
        preallocated_region_queue.dequeue_exact(1, &prealloc_r);
        if (parent->dist_search_mode ==
            DistributedSearchMode::STATE_SEND_CLIENT_GATHER) {
          for (uint64_t i = num_sent; i < num_sent + batch_size; i++) {
            states->at(i)->full_retset.clear();
            // clear because we don't need this anymore
          }
        }

        MessageType msg_type = MessageType::STATES;
        prealloc_r->length = sizeof(MessageType) +
                             SearchState<T, TagT>::get_serialize_size_states(
									     states->data() + num_sent, batch_size);
        if (unlikely(prealloc_r->length >= Region::MAX_BYTES_REGION)) {
	  std::stringstream error;
          error << "trying to write state result that is bigger than region "
                   "buffer: "
                << "batch size  is" << batch_size << ", size of total msg "
          << prealloc_r->length;
          throw std::runtime_error(error.str());
        }
        size_t offset = 0;
        std::memcpy(prealloc_r->addr, &msg_type, sizeof(msg_type));
        offset += sizeof(msg_type);
        SearchState<T, TagT>::write_serialize_states(
						     prealloc_r->addr + offset, states->data() + num_sent, batch_size);
        parent->communicator->send_to_peer(server_peer_id, prealloc_r);
        num_sent += batch_size;
      }
      states_used.insert(states->begin(), states->end());
    }
    SingletonLogger::get_logger().info("[{}]: Num batched elements {}",
                                       SingletonLogger::get_timestamp_ns(),
                                       states_used.size());

    for (SearchState<T, TagT> *const &state : states_used) {
      parent->preallocated_state_queue.free(state);
    }
    states_used.clear();
  }
}

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::CounterThread::CounterThread(
    SSDPartitionIndex *parent, const std::string &log_output_path,
    uint64_t sleep_duration_ms)
    : parent(parent), sleep_duration_ms(sleep_duration_ms),
      csv_output(log_output_path, std::ios::out) {
  cached_csv_output << std::setprecision(15);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::CounterThread::start() {
  running = true;
  real_thread =
      std::thread(&SSDPartitionIndex<T, TagT>::CounterThread::main_loop, this);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::CounterThread::signal_stop() {
  running = false;
  std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration_ms));
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::CounterThread::join() {
  if (real_thread.joinable()) {
    real_thread.join();
  }
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::CounterThread::write_header_csv() {
  cached_csv_output << "timestamp_ns" << ",";
  cached_csv_output << "num_states_global_queue" << ","
                    << "num_foreign_states_global_queue" << ","
                    << "num_new_states_global_queue" << ",";
  for (auto i = 0; i < parent->num_worker_threads; i++) {
    std::string thread_header_prefix = "thread" + std::to_string(i);
    std::string num_state_pipeline = thread_header_prefix + "_num_states";
    std::string num_foreign_state_pipeline =
        thread_header_prefix + "_num_foreign_states";
    std::string num_own_state_pipeline =
        thread_header_prefix + "_num_own_states";
    cached_csv_output << num_state_pipeline << "," << num_foreign_state_pipeline
                      << "," << num_own_state_pipeline << ",";
  }
  size_t num_other_peer_ids = parent->communicator->get_num_peers() - 1;
  for (const auto &other_peer_id : parent->communicator->get_other_peer_ids()) {
    std::string peer_id_prefix = "peer_" + std::to_string(other_peer_id);
    std::string num_ele_to_send = peer_id_prefix + "_num_ele_to_send";
    cached_csv_output << num_ele_to_send;
    num_other_peer_ids--;
    if (num_other_peer_ids != 0)
      cached_csv_output << ",";
  }
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::CounterThread::write_one_row_to_csv() {
  cached_csv_output << parent->global_state_queue.size_approx() << ","
                    << parent->num_foreign_states_global_queue << ","
                    << parent->num_new_states_global_queue << ",";
  for (auto i = 0; i < parent->num_worker_threads; i++) {
    // std::string thread_header_prefix = "thread" + std::to_string(i);
    // std::string num_state_pipeline =
    // thread_header_prefix + "_num_states_pipeline";
    // std::string num_foreign_state_pipeline =
    // thread_header_prefix + "_num_foreign_states_pipeline";
    // std::string num_own_state_pipeline =
    // thread_header_prefix + "_num_own_states_pipeline";
    uint64_t num_state_pipeline =
        parent->search_threads[i]->number_concurrent_queries;
    uint64_t num_own_states = parent->search_threads[i]->number_own_states;
    uint64_t num_foreign_states =
        parent->search_threads[i]->number_foreign_states;
    cached_csv_output << num_state_pipeline << "," << num_foreign_states << ","
                      << num_own_states << ",";
  }
  auto num_msg_peers = parent->batching_thread->get_num_msg_peers();
  for (auto i = 0; i < num_msg_peers.size(); i++) {
    cached_csv_output << num_msg_peers[i];
    if (i != num_msg_peers.size() - 1) {
      cached_csv_output << ",";
    }
  }
  cached_csv_output << "\n";
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::CounterThread::main_loop() {
  auto duration_ms = std::chrono::milliseconds(sleep_duration_ms);
  LOG(INFO) << "sleep duration is" << sleep_duration_ms;
  write_header_csv();
  while (running) {
    write_one_row_to_csv();
    std::this_thread::sleep_for(duration_ms);
  }
  csv_output << cached_csv_output.str();
  csv_output.close();
}

template <typename T, typename TagT>
size_t
SSDPartitionIndex<T, TagT>::state_write_result(SearchState<T, TagT> *state,
                                               char *buffer) {
  static thread_local uint32_t node_id[MAX_L_SEARCH * 2];
  static thread_local float distance[MAX_L_SEARCH * 2];
  size_t offset = 0;
  write_data(buffer, reinterpret_cast<const char *>(&state->query_id),
             sizeof(state->query_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&state->client_peer_id),
             sizeof(state->client_peer_id), offset);
  write_data(buffer, reinterpret_cast<const char *>(&state->k_search),
             sizeof(state->k_search), offset);

  uint64_t num_res = 0;

  for (uint64_t i = 0;
       i < state->full_retset.size() &&
       (dist_search_mode == DistributedSearchMode::STATE_SEND_CLIENT_GATHER
            ? true
            : num_res < state->k_search);
       i++) {
    if (i > 0 && state->full_retset[i].id == state->full_retset[i - 1].id) {
      continue; // deduplicate.
    }
    // write_data(char *buffer, const char *data, size_t size, size_t &offset)
    node_id[num_res] = state->full_retset[i].id; // use ID to replace tags
    distance[num_res] = state->full_retset[i].distance;
    num_res++;
  }
  if (num_res > search_result_t::get_max_num_res()) {
    throw std::runtime_error("num res larger than max allowable");
  }
  apply_tags_to_result(node_id, num_res);
  write_data(buffer, reinterpret_cast<const char *>(&num_res), sizeof(num_res),
             offset);
  write_data(buffer, reinterpret_cast<const char *>(node_id),
             sizeof(uint32_t) * num_res, offset);
  write_data(buffer, reinterpret_cast<const char *>(distance),
             sizeof(float) * num_res, offset);
  size_t num_partitions = state->partition_history.size();
  write_data(buffer, reinterpret_cast<const char *>(&num_partitions),
             sizeof(num_partitions), offset);
  write_data(buffer,
             reinterpret_cast<const char *>(state->partition_history.data()),
             sizeof(uint8_t) * num_partitions, offset);

  size_t num_partition_history_idx = state->partition_history_hop_idx.size();
  write_data(buffer, reinterpret_cast<const char *>(&num_partition_history_idx),
             sizeof(num_partition_history_idx), offset);
  write_data(
      buffer,
      reinterpret_cast<const char *>(state->partition_history_hop_idx.data()),
      sizeof(uint32_t) * num_partition_history_idx, offset);
  bool record_stats = (state->stats != nullptr);
  write_data(buffer, reinterpret_cast<const char *>(&record_stats),
             sizeof(record_stats), offset);
  if (record_stats) {
    offset += state->stats->write_serialize(buffer + offset);
  }
  bool is_final_result = state_search_ends(state);

  write_data(buffer, reinterpret_cast<const char *>(&is_final_result),
             sizeof(is_final_result), offset);
  return offset;
}

template <typename T, typename TagT>
size_t SSDPartitionIndex<T, TagT>::state_get_serialize_result_size(
    SearchState<T, TagT> *state) {
  size_t num_bytes = 0;
  uint64_t num_res = 0;
  for (uint64_t i = 0;
       i < state->full_retset.size() &&
       (dist_search_mode == DistributedSearchMode::STATE_SEND_CLIENT_GATHER
            ? true
            : num_res < state->k_search);
       i++) {
    if (i > 0 && state->full_retset[i].id == state->full_retset[i - 1].id) {
      continue; // deduplicate.
    }
    num_res++;
  }
  bool is_final_result = state_search_ends(state);
  num_bytes +=
      sizeof(state->query_id) + sizeof(state->client_peer_id) +
      sizeof(state->k_search) + sizeof(num_res) + sizeof(uint32_t) * num_res +
      sizeof(float) * num_res + sizeof(size_t) +
      sizeof(uint8_t) * state->partition_history.size() + sizeof(size_t) +
      sizeof(uint32_t) * state->partition_history_hop_idx.size() +
      sizeof(bool) + sizeof(is_final_result);
  if (state->stats != nullptr) {
    num_bytes += state->stats->get_serialize_size();
  }
  return num_bytes;
}

template <typename T, typename TagT>
size_t SSDPartitionIndex<T, TagT>::states_write_results(
    SearchState<T, TagT> **states, size_t num_states, char *buffer) {
  size_t offset = 0;
  std::memcpy(buffer + offset, &num_states, sizeof(num_states));
  offset += sizeof(num_states);

  for (size_t i = 0; i < num_states; i++) {
    offset += state_write_result(states[i], buffer + offset);
  }
  return offset;
}

template <typename T, typename TagT>
size_t SSDPartitionIndex<T, TagT>::states_get_serialize_result_sizes(
    SearchState<T, TagT> **states, size_t num_states) {
  size_t num_bytes = sizeof(num_states);
  for (size_t i = 0; i < num_states; i++) {
    num_bytes += state_get_serialize_result_size(states[i]);
  }
  return num_bytes;
}

template <typename T, typename TagT>
bool SSDPartitionIndex<T, TagT>::state_should_send_emb(
    SearchState<T, TagT> *state, uint64_t server_peer_id) {
  return std::find(state->partition_history.cbegin(),
                   state->partition_history.cend(),
                   static_cast<uint8_t>(server_peer_id)) ==
         state->partition_history.cend();
}

template class SSDPartitionIndex<float, uint32_t>;
template class SSDPartitionIndex<uint8_t, uint32_t>;
template class SSDPartitionIndex<int8_t, uint32_t>;
