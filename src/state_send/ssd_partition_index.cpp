#include "types.h"
#include "ssd_partition_index.h"
#include "query_buf.h"
#include <limits>
#include <stdexcept>

template <typename T, typename TagT>
SSDPartitionIndex<T, TagT>::SSDPartitionIndex(
    pipeann::Metric m, uint8_t partition_id, uint32_t num_partitions,
    uint32_t num_search_threads, std::shared_ptr<AlignedFileReader> &fileReader,
    std::unique_ptr<P2PCommunicator> &communicator,
    bool tags, Parameters *params, bool is_local)
: reader(fileReader), is_local(is_local), communicator(communicator) {
  if (is_local) {
    assert(communicator == nullptr);
  }
  this->my_partition_id = partition_id;
  this->num_partitions = num_partitions;
  if (num_search_threads > MAX_SEARCH_THREADS) {
    throw std::invalid_argument("num search threads > MAX_SEARCH_THREADS");
  }
  this->num_search_threads = num_search_threads;
  data_is_normalized = false;
  this->enable_tags = tags;
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
                                     const char* cluster_assignment_file) {
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
  this->num_points = this->init_num_pts = npts_u64;
  this->n_chunks = nchunks_u64;

  this->cur_id = this->num_points;

  LOG(INFO) << "Load compressed vectors from file: " << pq_compressed_vectors
            << " offset: " << pq_vectors_offset << " num points: " << npts_u64
            << " n_chunks: " << nchunks_u64;

  pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64,
                                pq_pivots_offset);

  if (disk_nnodes != num_points) {
    LOG(INFO) << "Mismatch in #points for compressed data file and disk "
                 "index file: "
              << disk_nnodes << " vs " << num_points;
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
  reader->open(index_fname, true, false);

  // load tags
  if (this->enable_tags) {
    std::string tag_file = disk_index_file + ".tags";
    LOG(INFO) << "Loading tags from " << tag_file;
    this->load_tags(tag_file);
  }

  num_medoids = 1;
  medoids = new uint32_t[1];
  medoids[0] = (uint32_t)(medoid_id_on_file);

  if (num_partitions > 1) {
    // loading the id2loc file
    std::string id2loc_file = iprefix + "_ids_uint32_t.bin";
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
    for (size_t i = 0; i < num_points; i++) {
      id2loc_.insert_or_assign(id2loc_v[i], i);
    }
    LOG(INFO) << "Id2loc file loaded successfully.";

    std::string cluster_file(cluster_assignment_file);
    if (!file_exists(cluster_file)) {
      throw std::invalid_argument(
          "number of partitions is " + std::to_string(num_partitions) +
          ", but the cluster assignment bin file doesn't exist: " + id2loc_file);
    }
    std::ifstream cluster_assignment_in(cluster_assignment_file,
                                        std::ios::binary);
    uint32_t whole_graph_num_pts;
    uint8_t num_clusters;
    cluster_assignment_in.read((char *)&whole_graph_num_pts,
			       sizeof(whole_graph_num_pts));
    cluster_assignment_in.read((char *)&num_clusters, sizeof(num_clusters));
    assert(num_clusters == num_partitions);

    cluster_assignment = std::vector<uint8_t>(whole_graph_num_pts);
    cluster_assignment_in.read((char *)cluster_assignment.data(),
                               whole_graph_num_pts * sizeof(uint8_t));
    cluster_assignment_in.close();

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
  assert(is_local);
  auto &full_retset = search_state->full_retset;
  std::sort(full_retset.begin(), full_retset.end(),
            [](const pipeann::Neighbor &left, const pipeann::Neighbor &right) {
              return left < right;
            });

  uint64_t t = 0;
  for (uint64_t i = 0; i < full_retset.size() && t < search_state->k_search; i++) {
    if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
      continue;  // deduplicate.
    }
    search_state->res_tags[t] = full_retset[i].id;  // use ID to replace tags
    if (search_state->res_dists != nullptr) {
      search_state->res_dists[t] = full_retset[i].distance;
    }
    t++;
  }
  search_state->completion_count->fetch_add(1);
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::search_ssd_index_local(
    const T *query_emb, const uint64_t k_search, const uint32_t mem_L,
    const uint64_t l_search, TagT *res_tags, float *res_dists,
    const uint64_t beam_width,
							std::shared_ptr<std::atomic<uint64_t>> completion_count) {
  assert(is_local);
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


  memcpy(new_search_state->query, query_emb, this->data_dim * sizeof(T));
  float *pq_dists = new_search_state->pq_dists;
  pq_table.populate_chunk_distances(new_search_state->query, pq_dists);
  state_reset(new_search_state);
  uint32_t best_medoid = medoids[0];
  state_compute_and_add_to_retset(new_search_state, &best_medoid, 1);
  // std::sort(new_search_state->retset.begin(),
            // new_search_state->retset.begin() + new_search_state->cur_list_size);  
#ifdef BALANCE_ALL
  void *ctx = search_threads[current_search_thread_id]->ctx;
  if (ctx == nullptr) {
    std::stringstream err;
    err << "[" << __func__ << "] tried to issue io but ctx = nullptr" << std::endl;
    throw std::runtime_error(err.str());
  }
  new_search_state->issue_next_io_batch(ctx);
#else
  search_threads[current_search_thread_id]->push_state(new_search_state);
#endif
  current_search_thread_id =
    (current_search_thread_id + 1) % search_threads.size();
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::search_ssd_index(const T *query_emb,
                                                  const uint64_t k_search,
                                                  const uint32_t mem_L,
                                                  const uint64_t l_search,
                                                  const uint64_t beam_width,
                                                  const uint64_t client_peer_id) {
  assert(is_local);
  if (beam_width != 1) {
    throw std::invalid_argument("beam width has to be 1 because of the design");
  }
  // need to create a state then issue io
  SearchState<T, TagT> *new_search_state = new SearchState<T, TagT>;
  new_search_state->client_type = ClientType::LOCAL;
  new_search_state->l_search = l_search;
  new_search_state->k_search = k_search;
  new_search_state->beam_width = beam_width;
  new_search_state->client_peer_id = client_peer_id;

  memcpy(new_search_state->query, query_emb, this->data_dim * sizeof(T));
  float *pq_dists = new_search_state->pq_dists;
  pq_table.populate_chunk_distances(new_search_state->query, pq_dists);
  state_reset(new_search_state);
  uint32_t best_medoid = medoids[0];
  state_compute_and_add_to_retset(new_search_state, &best_medoid, 1);
}

template <typename T, typename TagT> void SSDPartitionIndex<T, TagT>::start() {
  for (uint64_t thread_id = 0; thread_id < num_search_threads; thread_id++) {
    search_threads.push_back(std::make_unique<SearchThread>(this, thread_id));
  }
  for (uint64_t thread_id = 0; thread_id < num_search_threads; thread_id++) {
    search_threads[thread_id]->start();
  }
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::shutdown() {
  for (uint64_t thread_id = 0; thread_id < num_search_threads; thread_id++) {
    search_threads[thread_id]->signal_stop();
  }
  for (uint64_t thread_id = 0; thread_id < num_search_threads; thread_id++) {

    search_threads[thread_id]->join();
  }
  std::cout << "DONE WITH SHUTOWN" <<std::endl;
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::load_mem_index(
    pipeann::Metric metric, const size_t query_dim,
    const std::string &mem_index_path) {
  if (mem_index_path.empty()) {
    LOG(ERROR) << "mem_index_path is needed";
    exit(1);
  }
  throw std::runtime_error("Just doing testing on main diskann rn");
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::notify_client(SearchState<T, TagT> *search_state) {
  if (is_local)
    notify_client_local(search_state);
}


template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::receive_handler(const char *buffer,
                                                 size_t size) {
  std::vector<SearchState<T, TagT> *> states =
    SearchState<T, TagT>::deserialize_states(buffer, size);
  for (const auto &state : states) {
    uint64_t thread_id = current_search_thread_id.fetch_add(1);
    thread_id = thread_id % num_search_threads;
    search_threads[thread_id]->push_state(state);
  }
}



    
template class SSDPartitionIndex<float, uint32_t>;
template class SSDPartitionIndex<uint8_t, uint32_t>;
template class SSDPartitionIndex<int8_t, uint32_t>;



