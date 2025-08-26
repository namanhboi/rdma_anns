#include <immintrin.h> // needed to include this to make sure that the code compiles since in DiskANN/include/utils.h it uses this library.
#include <libaio.h>
#include "aligned_file_reader.h"
#include "linux_aligned_file_reader.h"
#include "neighbor.h"
#include "pq_flash_index.h"
#include "serialize_utils.hpp"
#include "tsl/robin_map.h"
#include "utils.h"
#include <cascade/service_types.hpp>
#include <memory>
#include <stdexcept>
#include "get_request_manager.hpp"
#include "udl_path_and_index.hpp"
#include "pq.h"
#include "concurrent_search_data_structures.hpp"
#include "pq_scratch.h"
#define BEAMWIDTH 1
#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))

namespace derecho {
namespace cascade {

/**
   final ssd index class. It will use the pq data from kvstore.
   in search function, extra parameters that we need:
   - std::function to send compute task to other udls
   - concurrent priority queue
 */
  template <typename data_type> class SSDIndex {
    std::string udl_cluster_data_prefix;

    //thread data used for searching
    diskann::ConcurrentQueue<diskann::SSDThreadData<data_type> *> search_thread_data;

    // thread data for compute tasks
    diskann::ConcurrentQueue<diskann::SSDThreadData<data_type> *> compute_thread_data;
    
    // used to determine the order/index in which that node_id is written to the
    // cluster disk index file. This index is then used to calculate the sector
    // id and offset
    tsl::robin_map<uint64_t, uint64_t> node_id_to_index;

    // used to determine which cluster a node resides in.
    std::vector<uint8_t> cluster_assignment;

    diskann::FixedChunkPQTable pq_table;

    // distance comparator
    std::shared_ptr<diskann::Distance<data_type>> _dist_cmp;
    std::shared_ptr<diskann::Distance<float>> _dist_cmp_float;

    
    size_t num_points;
    size_t num_chunks = 32;

    size_t _aligned_dim;
    size_t dim;

    size_t _disk_bytes_per_point;


    std::shared_ptr<AlignedFileReader> reader;
    uint64_t _max_node_len = 0;
    uint64_t _nnodes_per_sector = 0; // 0 for multi-sector nodes, >0 for multi-node sectors
    uint64_t _max_degree = 0;

    uint8_t cluster_id;


    std::function<void(compute_query_t)> send_compute_query_fn;

#ifdef PQ_KV
    void aggregate_pq_coord(
        GetRequestVector<data_type, ObjectWithStringKey> &pq_requests,
			    uint8_t *pq_coord_scratch) {
      std::vector<std::pair<uint64_t, std::shared_ptr<const data_type>>>
      pq_results = pq_requests.get_results();

      for (auto i = 0; i < pq_results.size(); i++) {
        std::memcpy(pq_coord_scratch + i * this->num_chunks,
                    pq_results[i].second.get(),
                    this->num_chunks * sizeof(uint8_t));
      }
    }
#else
    uint8_t* pq_data;
    
#endif
    

    inline bool is_in_cluster(uint32_t node_id) const {
      return cluster_assignment[node_id] == this->cluster_id;
    }

    inline uint64_t get_node_sector(uint64_t node_id) {
      return 1 + (_nnodes_per_sector > 0
                      ? node_id_to_index[node_id] / _nnodes_per_sector
                      : node_id_to_index[node_id] *
                            DIV_ROUND_UP(_max_node_len,
                                         diskann::defaults::SECTOR_LEN));
    }
    inline char *offset_to_node(char *sector_buf, uint64_t node_id) {
      return sector_buf +
             (_nnodes_per_sector == 0
                  ? 0
                  : (node_id_to_index[node_id] % _nnodes_per_sector) *
                    _max_node_len);
    }

    inline uint32_t* offset_to_node_nhood(char *node_buf){
      return (unsigned *)(node_buf + _disk_bytes_per_point);
    }
    inline data_type* offset_to_node_coords(char *node_buf)
    {
      return (data_type *)(node_buf);
    }

  public:
    /**
       index_path_prefix will be used to determine this cluster's disk index
       file and its node id mapping.
       udl_cluster_data_prefix will be used to retrieve pq data

       cluster_assignment_file will be used to determine which cluster a node id
       belongs to.


no caching suppport rn, since there is the head index?
       */
    SSDIndex(const std::string &index_path_prefix, const uint8_t &cluster_id,
             const std::string &udl_cluster_data_prefix,
             const std::string &cluster_assignment_file,
             const std::string &pq_table_bin, uint64_t num_search_threads, uint64_t num_compute_threads,
             std::function<void(compute_query_t)> send_compute_query_fn, std::string pq_compressed_vectors = "")
        : cluster_id(cluster_id),
          udl_cluster_data_prefix(udl_cluster_data_prefix),
          search_thread_data(nullptr), send_compute_query_fn(send_compute_query_fn)
    {
      // TODO register aligned reader and open the disk index file
      std::string disk_index_file = index_path_prefix + "_disk.index";
      if (!std::filesystem::exists(disk_index_file)) {
        std::stringstream error_msg;
        error_msg << "disk index file doesn't exist " << disk_index_file
        << " cluster id " << cluster_id << std::endl;
        throw std::invalid_argument(error_msg.str());
      }

      reader.reset(new LinuxAlignedFileReader());
      reader->open(disk_index_file);

      
      std::ifstream index_metadata(disk_index_file, std::ios::binary);
      uint32_t nr, nc; // metadata itself is stored as bin format (nr is number
      // of metadata, nc should be 1)
            
      READ_U32(index_metadata, nr);
      READ_U32(index_metadata, nc);

      uint64_t disk_nnodes;
      uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
      READ_U64(index_metadata, disk_nnodes);
      READ_U64(index_metadata, disk_ndims);
      this->dim = disk_ndims;
      std::cout << "dimension is " << disk_ndims << std::endl;
      this->num_points = disk_nnodes;
      this->_disk_bytes_per_point = this->dim * sizeof(data_type);
      std::cout << "disk bytes per point is " << _disk_bytes_per_point
      << std::endl;
      std::cout << "size of data type is " << sizeof(data_type) << std::endl;
      
      size_t medoid_id_on_file;
      READ_U64(index_metadata,
               medoid_id_on_file); // dont really need this since head index
      // search will provide starting points
      READ_U64(index_metadata, _max_node_len);
      READ_U64(index_metadata, _nnodes_per_sector);
      std::cout << "max node len is " << _max_node_len << std::endl;
      std::cout << ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1<< std::endl;
      _max_degree =
        ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;
      std::cout << "max degree is " << _max_degree << std::endl;
      if (_max_degree > diskann::defaults::MAX_GRAPH_DEGREE){
          std::stringstream stream;
          stream << "Error loading index. Ensure that max graph degree (R) does "
                    "not exceed "
          << diskann::defaults::MAX_GRAPH_DEGREE << std::endl;
          throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
      }
      index_metadata.close();

      // TODO load in the node ids mapping
      uint32_t *node_ids = nullptr;
      std::string node_ids_file = index_path_prefix + "_ids_uint32_t.bin";
      size_t num_pts, one;
      diskann::load_bin<uint32_t>(node_ids_file, node_ids, num_pts, one);
      if (one != 1) {
        throw std::runtime_error("dim from load_bin for node_ids_file not 1");
      }
      if (num_pts != num_points) {
        throw std::runtime_error("num points from node ids file different from disk_index: " + std::to_string(num_pts) + " " + std::to_string(num_points));
      }

      for (auto i = 0; i < num_pts; i++) {
        node_id_to_index[((uint64_t)node_ids[i])] = ((uint64_t)i);
      }

      // TODO load in cluster assignemnt file
      std::ifstream cluster_assignment_in(cluster_assignment_file,
                                          std::ios::binary);
      uint32_t whole_graph_num_pts;
      uint8_t num_clusters;
      cluster_assignment_in.read((char *)&whole_graph_num_pts,
				 sizeof(whole_graph_num_pts));
      cluster_assignment_in.read((char *)&num_clusters, sizeof(num_clusters));

      cluster_assignment = std::vector<uint8_t>(whole_graph_num_pts);
      cluster_assignment_in.read((char *)cluster_assignment.data(),
                                 whole_graph_num_pts * sizeof(uint8_t));
      cluster_assignment_in.close();

      // loading pq table
      size_t pq_file_dim, pq_file_num_centroids;
      diskann::get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim,
				METADATA_SIZE);
      if (pq_file_num_centroids != 256) {
	throw std::runtime_error(
				 "Error. Number of PQ centroids is not 256. Exiting.");
      }
      if (pq_file_dim != disk_ndims) {
        throw std::runtime_error(
            "Error. Dimension of pq file different from disk file: " +
            std::to_string(pq_file_dim) + " " + std::to_string(disk_ndims));
      }
      this->_aligned_dim = ROUND_UP(pq_file_dim, 8);      

      pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), num_chunks);

      // setup distance comparison functions
      this->_dist_cmp.reset(
			    diskann::get_distance_function<data_type>(diskann::Metric::L2));
      this->_dist_cmp_float.reset(
				  diskann::get_distance_function<float>(diskann::Metric::L2));
      // setup_thread_data(num_search_threads, this->search_thread_data, 4096);
      // setup_thread_data(num_compute_threads, this->compute_thread_data, 0);
#ifdef PQ_FS
      size_t npts_u64, nchunks_u64;
      diskann::load_bin<uint8_t>(pq_compressed_vectors, this->pq_data, npts_u64,
                                 nchunks_u64);
#endif
    }
    void setup_search_thread_data(uint32_t visited_reserve) {
      diskann::SSDThreadData<data_type> *data =
          new diskann::SSDThreadData<data_type>(this->_aligned_dim,
                                                visited_reserve);
      this->reader->register_thread();
      data->ctx = this->reader->get_ctx();
      search_thread_data.push(data);
    }

    void setup_compute_thread_data() {
      diskann::SSDThreadData<data_type> *data =
        new diskann::SSDThreadData<data_type>(this->_aligned_dim, 0);
      this->reader->register_thread();
      data->ctx = this->reader->get_ctx();
      compute_thread_data.push(data);
    }

    void initialize_pq_scratch(
        diskann::PQScratch<data_type> *pq_query_scratch,
			       data_type* aligned_query_T) {
      pq_query_scratch->initialize(this->dim, aligned_query_T);
      pq_table.preprocess_query(pq_query_scratch->rotated_query);

      float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
      pq_table.populate_chunk_distances(pq_query_scratch->rotated_query,
                                        pq_dists);
    }

    std::shared_ptr<compute_result_t> execute_compute_query(
        DefaultCascadeContextType *typed_ctxt, compute_query_t compute_query,
        std::shared_ptr<EmbeddingQuery<data_type>> query_emb,
							    std::shared_ptr<diskann::PQScratch<data_type>> pq_query_scratch, std::once_flag &initialized_pq_scratch) {
      if (!is_in_cluster(compute_query.node_id)) {
        std::stringstream err;
        err << "Compute query " << compute_query.query_id << " for node id " << compute_query.node_id << " is not in the cluster " << cluster_id << std::endl;
        throw std::runtime_error(err.str());
      }

      TimestampLogger::log(LOG_GLOBAL_INDEX_COMPUTE_GET_SCRATCH_START,
                           compute_query.client_node_id,
                           compute_query.get_msg_id(), 0ull);

      diskann::ScratchStoreManager<diskann::SSDThreadData<data_type>> manager(
									      this->compute_thread_data);

      auto data = manager.scratch_space();
      IOContext &ctx = data->ctx;
      
      // // all data necessary for querying and pq operations will be accessed
      // // through these 2 pointers.
      auto query_scratch = &(data->scratch);

      query_scratch->reset();

      TimestampLogger::log(LOG_GLOBAL_INDEX_COMPUTE_GET_SCRATCH_END,
                           compute_query.client_node_id,
                           compute_query.get_msg_id(), 0ull);

      // // aligned malloc memset to 0, need to copy query to here to do
      // // computation since a lot of operations only work on aligned mem address
      data_type *aligned_query_T = query_scratch->aligned_query_T();

      TimestampLogger::log(LOG_GLOBAL_INDEX_COMPUTE_PREP_QUERY_START,
                           compute_query.client_node_id,
                           compute_query.get_msg_id(), 0ull);

      // // whether this is used or not depends on there is a rotation matrix file
      // // that has index_path_prefix as a path prefix
      float *query_rotated = pq_query_scratch->rotated_query;

      for (size_t i = 0; i < this->dim; i++) {
	aligned_query_T[i] = query_emb->get_embedding_ptr()[i];
      }
      std::call_once(initialized_pq_scratch,
                     &SSDIndex<data_type>::initialize_pq_scratch, this,
                     pq_query_scratch.get(), aligned_query_T);

      // // // this is where the full precision embeddings will be copied into to do
      // // full distance computation with
      data_type *full_precision_emb_buffer = query_scratch->coord_scratch;

      // write data from aio here
      char *sector_scratch = query_scratch->sector_scratch;
      uint64_t &sector_scratch_idx =
          query_scratch->sector_idx; // index into sector_scratch so that writes
      // dont overlap
      
      const uint64_t num_sectors_per_node =
          _nnodes_per_sector > 0
              ? 1
          : DIV_ROUND_UP(_max_node_len, diskann::defaults::SECTOR_LEN);

      // // // not sure why this is necessary in pq_flash_index search method since
      // // // won't we be overwriting it constantly?
      // // _mm_prefetch((char *)full_precision_emb_buffer, _MM_HINT_T1);

      float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;

      float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
      uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;
      TimestampLogger::log(LOG_GLOBAL_INDEX_COMPUTE_PREP_QUERY_END,
                           compute_query.client_node_id,
                           compute_query.get_msg_id(), 0ull);

#ifdef PQ_KV
      auto compute_dists =
          [this, pq_coord_scratch, pq_dists](
              GetRequestVector<data_type, ObjectWithStringKey> &pq_requests,
              float *dists_out) {
            size_t num_requests = pq_requests.size();
            aggregate_pq_coord(pq_requests, pq_coord_scratch);
            diskann::pq_dist_lookup(pq_coord_scratch, num_requests,
                                    this->num_chunks, pq_dists, dists_out);
          };
      std::string in_cluster_pq_prefix = this->udl_cluster_data_prefix + "_pq_";
      GetRequestVector<data_type, ObjectWithStringKey> in_cluster_pq_requests(
									      typed_ctxt, _max_degree);
#elif PQ_FS
      auto compute_dists = [this, pq_coord_scratch,
                            pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                      float *dists_out) {
	diskann::aggregate_coords(ids, n_ids, this->pq_data,
                                  this->num_chunks, pq_coord_scratch);
	diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->num_chunks,
				pq_dists, dists_out);
      };
#endif

      
      uint32_t node_id = compute_query.node_id;
      std::pair<uint32_t, char *> fnhood;
      fnhood.first = node_id;
      fnhood.second = sector_scratch +
                      num_sectors_per_node * sector_scratch_idx *
                      diskann::defaults::SECTOR_LEN;
      std::vector<AlignedRead> frontier_read_reqs;
      frontier_read_reqs.emplace_back(
          get_node_sector((size_t)node_id) * diskann::defaults::SECTOR_LEN,
				      num_sectors_per_node * diskann::defaults::SECTOR_LEN, fnhood.second);
      TimestampLogger::log(LOG_GLOBAL_INDEX_COMPUTE_READ_START,
                           compute_query.client_node_id,
                           compute_query.get_msg_id(), 0ull);      
      reader->read(frontier_read_reqs, ctx);
      TimestampLogger::log(LOG_GLOBAL_INDEX_COMPUTE_READ_END,
                           compute_query.client_node_id,
                           compute_query.get_msg_id(), 0ull);

      TimestampLogger::log(LOG_GLOBAL_INDEX_COMPUTE_CALC_START,
                           compute_query.client_node_id,
                           compute_query.get_msg_id(), 0ull);
      char *node_disk_buf = offset_to_node(fnhood.second, fnhood.first);
      uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
      uint64_t num_neighbors = (uint64_t)(*node_buf);
      data_type *node_fp_coords = offset_to_node_coords(node_disk_buf);
      memcpy(full_precision_emb_buffer, node_fp_coords, _disk_bytes_per_point);
      float expanded_dist = _dist_cmp->compare(
					       aligned_query_T, full_precision_emb_buffer, (uint32_t)_aligned_dim);

      uint32_t *node_nbrs = (node_buf + 1);
      std::shared_ptr<uint32_t[]> nbr_ids(new uint32_t[num_neighbors]);
      std::memcpy(nbr_ids.get(), node_nbrs, sizeof(uint32_t) * num_neighbors);
      std::shared_ptr<float[]> nbr_distances(new float[num_neighbors]);


#ifdef PQ_KV
      for (auto i = 0; i < num_neighbors; i++) {
        in_cluster_pq_requests.submit_request(
					      node_nbrs[i], in_cluster_pq_prefix + std::to_string(node_nbrs[i]));
      }
      compute_dists(in_cluster_pq_requests, nbr_distances.get());
#elif PQ_FS
      compute_dists(nbr_ids.get(), num_neighbors, nbr_distances.get());
#endif
      TimestampLogger::log(LOG_GLOBAL_INDEX_COMPUTE_CALC_END,
                           compute_query.client_node_id,
                           compute_query.get_msg_id(), 0ull);

      return std::make_shared<compute_result_t>(
          num_neighbors, nbr_ids, nbr_distances, compute_query.query_id,
          compute_query.node_id, compute_query.client_node_id, expanded_dist,
          compute_query.receiver_thread_id, compute_query.cluster_receiver_id,
						compute_query.cluster_sender_id);
    }

    //beam width is 1
    void search(DefaultCascadeContextType *typed_ctxt, const data_type *query,
                const uint32_t query_id, const uint32_t client_node_id,
                uint32_t search_thread_id, const int64_t K, const uint64_t L,
                uint64_t *indices, float *distances,
                std::vector<uint32_t> start_node_ids,
                std::shared_ptr<ConcurrentSearchData> search_data,
                uint32_t beam_width = 1) {
      
      TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_GET_SCRATCH_START,
                           client_node_id,
                           query_id, 0ull);
      
      diskann::ScratchStoreManager<diskann::SSDThreadData<data_type>> manager(
									      this->search_thread_data);
      auto data = manager.scratch_space();
      IOContext &ctx = data->ctx;
      
      // // all data necessary for querying and pq operations will be accessed
      // // through these 2 pointers.
      auto query_scratch = &(data->scratch);
      // this contains preallocated ptrs to pq table, pq processed query, etc
      auto pq_query_scratch = query_scratch->pq_scratch();

      query_scratch->reset();

      TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_GET_SCRATCH_END,
                           client_node_id, query_id, 0ull);
      
      // // aligned malloc memset to 0, need to copy query to here to do
      // // computation since a lot of operations only work on aligned mem address
      data_type *aligned_query_T = query_scratch->aligned_query_T();

      TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_PREP_QUERY_START,
                           client_node_id, query_id, 0ull);

      for (size_t i = 0; i < this->dim; i++) {
	aligned_query_T[i] = query[i];
      }

      initialize_pq_scratch(pq_query_scratch, aligned_query_T);

      // // // this is where the full precision embeddings will be copied into to do
      // // full distance computation with
      data_type *full_precision_emb_buffer = query_scratch->coord_scratch;

      // write data from aio here
      char *sector_scratch = query_scratch->sector_scratch;
      uint64_t &sector_scratch_idx =
          query_scratch->sector_idx; // index into sector_scratch so that writes
      // dont overlap
      
      const uint64_t num_sectors_per_node =
          _nnodes_per_sector > 0
              ? 1
          : DIV_ROUND_UP(_max_node_len, diskann::defaults::SECTOR_LEN);

      // // // not sure why this is necessary in pq_flash_index search method since
      // // // won't we be overwriting it constantly?
      // // _mm_prefetch((char *)full_precision_emb_buffer, _MM_HINT_T1);

      float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;

      float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
      uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

      TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_PREP_QUERY_END,
                           client_node_id, query_id, 0ull);
      
      ConcurrentNeighborPriorityQueue retset = search_data->get_retset();
      ConcurrentVisitedSet visited = search_data->get_visited();
      retset.reserve(L);
      ConcurrentResultVector full_retset = search_data->get_full_retset();
      

      std::vector<float> start_node_pq_dists(start_node_ids.size(), 0.0);
      TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_INIT_DISTANCES_START,
                           client_node_id, query_id, 0ull);
#ifdef PQ_KV
      auto compute_dists =
          [this, pq_coord_scratch, pq_dists](
              GetRequestVector<data_type, ObjectWithStringKey> &pq_requests,
              float *dists_out) {
            size_t num_requests = pq_requests.size();
            aggregate_pq_coord(pq_requests, pq_coord_scratch);
            diskann::pq_dist_lookup(pq_coord_scratch, num_requests,
                                    this->num_chunks, pq_dists, dists_out);
          };
      std::string in_cluster_pq_prefix = this->udl_cluster_data_prefix + "_pq_";
      GetRequestVector<data_type, ObjectWithStringKey> in_cluster_pq_requests(typed_ctxt, _max_degree);
      GetRequestVector<data_type, ObjectWithStringKey> off_cluster_pq_requests(
									       typed_ctxt, _max_degree);
      
      for (auto &node_id : start_node_ids) {
        in_cluster_pq_requests.submit_request(
					      node_id, in_cluster_pq_prefix + std::to_string(node_id));
      }

      compute_dists(in_cluster_pq_requests, start_node_pq_dists.data());

#elif PQ_FS
      auto compute_dists = [this, pq_coord_scratch,
                            pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                      float *dists_out) {
	diskann::aggregate_coords(ids, n_ids, this->pq_data,
                                  this->num_chunks, pq_coord_scratch);
	diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->num_chunks,
				pq_dists, dists_out);
      };
      compute_dists(start_node_ids.data(), start_node_ids.size(),
                    start_node_pq_dists.data());
#endif

      for (size_t i = 0; i < start_node_ids.size(); i++) {
        retset.insert(
		       diskann::Neighbor(start_node_ids[i], start_node_pq_dists[i]));
	visited.insert(start_node_ids[i]);
      }

      TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_INIT_DISTANCES_END,
                           client_node_id, query_id, 0ull);
      // used to determine which nodes to do get request, reason this exist is
      // because we might want to increase beamwidth in the future

      TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_SEARCH_START,
                           client_node_id, query_id, 0ull);
      while (retset.has_unexpanded_node()) {
        sector_scratch_idx = 0;

        auto nbr = retset.closest_unexpanded();
        uint32_t node_id = nbr.id;
        std::vector<AlignedRead> read_reqs;
        std::pair<uint32_t, char *> fnhood;
        if (is_in_cluster(node_id)) {
          fnhood.first = node_id;
          fnhood.second = sector_scratch + num_sectors_per_node *
                                               sector_scratch_idx *
                                               diskann::defaults::SECTOR_LEN;
          sector_scratch_idx++;
          read_reqs.emplace_back(
              get_node_sector((size_t)node_id) * diskann::defaults::SECTOR_LEN,
              num_sectors_per_node * diskann::defaults::SECTOR_LEN,
              fnhood.second);
        } else {
          size_t size = retset.size();
          float min_distance = retset.operator[](size - 1).distance;
          uint8_t nbr_cluster = cluster_assignment[node_id];
          compute_query_t query(node_id, query_id, client_node_id, min_distance,
                                this->cluster_id, nbr_cluster,
                                search_thread_id);
          TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_COMPUTE_SEND,
                               client_node_id, query.get_msg_id(), 0ull);
          this->send_compute_query_fn(query);
          continue;
        }

        uint64_t read_msg_id =
            (static_cast<uint64_t>(query_id) << 32) | node_id;

        TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_READ_START, client_node_id,
                             read_msg_id, 0ull);
        
        reader->read(read_reqs, ctx);
        TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_READ_END, client_node_id,
                             read_msg_id, 0ull);
        uint64_t compute_msg_id =
            static_cast<uint64_t>(query_id) << 32 | node_id;
        TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_COMPUTE_DIST_START,
                             client_node_id, compute_msg_id, 0ull);

        char *node_disk_buf = offset_to_node(fnhood.second, node_id);
        uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
        uint64_t nnbrs = (uint64_t)(*node_buf);
        data_type *node_fp_coords = offset_to_node_coords(node_disk_buf);
        memcpy(full_precision_emb_buffer, node_fp_coords,
               _disk_bytes_per_point);
        float cur_expanded_dist;
        cur_expanded_dist = _dist_cmp->compare(
            aligned_query_T, full_precision_emb_buffer, (uint32_t)_aligned_dim);
        full_retset.push_back(
            diskann::Neighbor(fnhood.first, cur_expanded_dist));
        uint32_t *node_nbrs = (node_buf + 1);
#ifdef PQ_KV
        auto num_off_cluster_reqs = off_cluster_pq_requests.size();
        auto off_cluster_req_ids = off_cluster_pq_requests.get_requests_ids();
        if (num_off_cluster_reqs != 0) {
          compute_dists(off_cluster_pq_requests, dist_scratch);
          for (auto i = 0; i < num_off_cluster_reqs; i++) {
            uint32_t id = off_cluster_req_ids[i];
            if (visited.insert(id).second) {
              float dist = dist_scratch[i];
              diskann::Neighbor nn(id, dist);
              retset->insert(nn);
            }
          }
        }
        for (auto i = 0; i < nnbrs; i++) {
          if (is_in_cluster(node_nbrs[i])) {
            in_cluster_pq_requests.submit_request(
                node_nbrs[i],
                in_cluster_pq_prefix + std::to_string(node_nbrs[i]));

          } else {
            uint8_t nbr_cluster = cluster_assignment[node_nbrs[i]];
            std::string off_cluster_pq_key =
                UDL2_DATA_PREFIX "/cluster_" + std::to_string(nbr_cluster) +
                "_pq_" + std::to_string(node_nbrs[i]);
            off_cluster_pq_requests.submit_request(node_nbrs[i],
                                                   off_cluster_pq_key);
          }
        }
        compute_dists(in_cluster_pq_requests, dist_scratch);
#elif PQ_FS
        compute_dists(node_nbrs, nnbrs, dist_scratch);
#endif
        for (uint32_t m = 0; m < nnbrs; m++) {
          uint32_t id = node_nbrs[m];
          // not in visited i guess
          if (visited.insert(id)) {
            float dist = dist_scratch[m];
            diskann::Neighbor nn(id, dist);
            retset.insert(nn);
          }
        }
        TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_COMPUTE_DIST_END,
                             client_node_id, compute_msg_id, 0ull);
      }

      full_retset.sort_and_write(indices, distances, K);
      TimestampLogger::log(LOG_GLOBAL_INDEX_SEARCH_SEARCH_END, client_node_id,
                           query_id, 0ull);
    }
    ~SSDIndex() {
#ifdef PQ_FS
      if (pq_data != nullptr) {
        delete[] pq_data;
      }
#endif
      diskann::cout << "Clearing scratch" << std::endl;
      diskann::ScratchStoreManager<diskann::SSDThreadData<data_type>>
          search_manager(this->search_thread_data);
      search_manager.destroy();
      diskann::ScratchStoreManager<diskann::SSDThreadData<data_type>>
          compute_manager(this->compute_thread_data);
      compute_manager.destroy();
    }
  };

/**
   currently these 2 classes are barebones impl for benchmarking whether cascade
   persistent kv can function as the abstracted datastore for ssd based anns.

Both don't support any medoid data stuff, only 1 entry point, and also doesn't
have any caching right now

right now kv version performs much better because of the in memeory cache of
cascade persistent store

*/

// basically a wrapper for diskann flash index
template <typename data_type> class SSDIndexFileSystem {
  std::unique_ptr<diskann::PQFlashIndex<data_type>> _pFlashIndex;
  std::shared_ptr<AlignedFileReader> reader;
public:
  /**
     num_search_threads is needed for diskann to preallocate memory for search
   */
  SSDIndexFileSystem(const std::string &index_path_prefix,
                     uint64_t num_search_threads) {
    reader.reset(new LinuxAlignedFileReader());
    _pFlashIndex.reset(
		       new diskann::PQFlashIndex<data_type>(reader, diskann::Metric::L2));

    int res = _pFlashIndex->load(num_search_threads, index_path_prefix.c_str());
    if (res != 0) {
      throw std::runtime_error("cant load ssd index");
    }
  }

  void search(const data_type *query, const int64_t K, const uint64_t L,
              uint64_t *indices, float *distances) {
    _pFlashIndex->cached_beam_search(query, K, L, indices, nullptr, 1);
  }
};

  template <typename data_type> class SSDIndexKV {
    std::string cluster_data_prefix;
    
    uint8_t* pq_data;
    size_t num_points;
    size_t num_chunks;

    size_t _aligned_dim;
    size_t dim;

    size_t vector_embedding_size;

    diskann::FixedChunkPQTable pq_table;

    // distance comparator
    std::shared_ptr<diskann::Distance<data_type>> _dist_cmp;
    std::shared_ptr<diskann::Distance<float>> _dist_cmp_float;

    // for caching
    //  nhood_cache; the uint32_t in nhood_Cache are offsets into nhood_cache_buf
    unsigned *_nhood_cache_buf = nullptr;
    tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>> _nhood_cache;

    // coord_cache; The T* in coord_cache are offsets into coord_cache_buf
    data_type *_coord_cache_buf = nullptr;
    tsl::robin_map<uint32_t, data_type *> _coord_cache;

    // queue containing the necessary data for each search thread
    // once search() is called, a ssdthreaddata obj is popped off queue
    diskann::ConcurrentQueue<diskann::SSDThreadData<data_type> *> _thread_data;

  public:
    /*
      num_threads is used to initlize the right number of scratch data for each
      thread start_node_id is needed because we want to simulate loading the
      metadata from file. We would do it by placing start_node_id in a kv pair but
      that's overkill. We can specify start_node_id in config and create ssdindexkv
      object by passing that into ctor.

Note:this constructor still uses the pq
compressed bin path and not from the pq data from kv store.
     */

    SSDIndexKV(const std::string &index_path_prefix,
               std::string cluster_data_prefix, uint64_t num_threads)
        : cluster_data_prefix(cluster_data_prefix),
        _thread_data(nullptr) {

      // loading pq data
      std::string pq_table_bin =
        std::string(index_path_prefix) + "_pq_pivots.bin";
      std::string pq_compressed_vectors =
        std::string(index_path_prefix) + "_pq_compressed.bin";
      size_t npts_u64, nchunks_u64;
      diskann::load_bin<uint8_t>(pq_compressed_vectors, this->pq_data, npts_u64,
                                 nchunks_u64);
      this->num_points = npts_u64;
      this->num_chunks = nchunks_u64;
      // std::cout << "num points " << num_points << " " << "num_chunkcs"
      // << num_chunks << std::endl;


      size_t pq_file_dim, pq_file_num_centroids;
      diskann::get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim,
				METADATA_SIZE);
      if (pq_file_num_centroids != 256) {
	throw std::runtime_error(
				 "Error. Number of PQ centroids is not 256. Exiting.");
      }
      
      this->dim = pq_file_dim;
      this->_aligned_dim = ROUND_UP(pq_file_dim, 8);      

      pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);

      // setup distance comparison functions
      this->_dist_cmp.reset(
			    diskann::get_distance_function<data_type>(diskann::Metric::L2));
      this->_dist_cmp_float.reset(
				  diskann::get_distance_function<float>(diskann::Metric::L2));
      this->vector_embedding_size = dim * sizeof(data_type);
      // std::cout << "size of vector embedding si " << vector_embedding_size << std::endl;
      // setup thread data
#pragma omp parallel for num_threads((int)num_threads)
      for (int64_t i = 0; i < (int64_t)num_threads; i++) {
#pragma omp critical
        {
          diskann::SSDThreadData<data_type> *data = new diskann::SSDThreadData<data_type>(this->_aligned_dim, 4096);
            this->_thread_data.push(data);
        }
      }
    }

    /** we don't need ssd thread data since the difference between it and
     * ssdquerydata is just iocontext which we don't need since we are getting
     * our data from cascade kv store instead of iouring
     *
     * This search currently uses the pq data from file and not from kv store.
     */
    void search_pq_fs(DefaultCascadeContextType *typed_ctxt,
                      const data_type *query, const int64_t K, const uint64_t L,
                      uint64_t *indices, float *distances,
                      std::vector<uint32_t> start_node_ids) {
      diskann::ScratchStoreManager<diskann::SSDThreadData<data_type>> manager(
									      this->_thread_data);
      auto data = manager.scratch_space();

      // // all data necessary for querying and pq operations will be accessed
      // // through these 2 pointers.
      auto query_scratch = &(data->scratch);
      // this contains preallocated ptrs to pq table, pq processed query, etc
      auto pq_query_scratch = query_scratch->pq_scratch();

      query_scratch->reset();

      // // aligned malloc memset to 0, need to copy query to here to do
      // // computation since a lot of operations only work on aligned mem address
      data_type *aligned_query_T = query_scratch->aligned_query_T();

      // // aligned malloc for pq computation,
      // // converts whatever type the query is to a float,
      // // will contain the same data as aligned_query_T
      float *query_float = pq_query_scratch->aligned_query_float;

      // // whether this is used or not depends on there is a rotation matrix file
      // // that has index_path_prefix as a path prefix
      float *query_rotated = pq_query_scratch->rotated_query;

      for (size_t i = 0; i < this->dim; i++) {
	aligned_query_T[i] = query[i];
      }

      // // copies data from aligned query to query float and query rotated
      pq_query_scratch->initialize(this->dim, aligned_query_T);

      // // // this is where the full precision embeddings will be copied into to do
      // // full distance computation with
      data_type *full_precision_emb_buffer = query_scratch->coord_scratch;

      // // // not sure why this is necessary in pq_flash_index search method since
      // // // won't we be overwriting it constantly?
      // // _mm_prefetch((char *)full_precision_emb_buffer, _MM_HINT_T1);

      pq_table.preprocess_query(query_rotated);

      float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
      pq_table.populate_chunk_distances(query_rotated, pq_dists);
      // after this, pq query distance to all centroids in pq table is
      // calcucated in pq_dists, and pq_dists is used for fast look up for pq
      // distance calculation

      // // this is where you write the pq coordinates of points to do pq
      // // computation with
      float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
      uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;


      // // lambda to batch compute query<-> node distances in PQ space from pq data
      // // loading from file
      auto compute_dists = [this, pq_coord_scratch,
                          pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                    float *dists_out) {
	diskann::aggregate_coords(ids, n_ids, this->pq_data,
                                  this->num_chunks, pq_coord_scratch);
	diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->num_chunks,
				pq_dists, dists_out);
      };

      tsl::robin_set<uint64_t> &visited = query_scratch->visited;
      diskann::NeighborPriorityQueue &retset = query_scratch->retset;
      retset.reserve(L);
      std::vector<diskann::Neighbor> &full_retset = query_scratch->full_retset;

      std::vector<float> start_node_pq_dists(start_node_ids.size(), 0.0);
      compute_dists(start_node_ids.data(), start_node_ids.size(),
                    start_node_pq_dists.data());

      for (size_t i = 0; i < start_node_ids.size(); i++) {
	retset.insert(
		      diskann::Neighbor(start_node_ids[i], start_node_pq_dists[i]));
	visited.insert(start_node_ids[i]);
      }

      // used to determine which nodes to do get request, reason this exist is
      // because we might want to increase beamwidth in the future
      std::vector<uint32_t> frontier;
      frontier.reserve(2 * BEAMWIDTH);

      GetRequestManager<uint8_t, ObjectWithStringKey> get_requests_manager;

      while (retset.has_unexpanded_node()) {
	frontier.clear();

	auto nbr = retset.closest_unexpanded();
	frontier.push_back(nbr.id);
        if (!frontier.empty()) {
          for (size_t i = 0; i < frontier.size();i++) {
            const std::string vector_key = this->cluster_data_prefix + "_vec_" + std::to_string(frontier[i]);
            get_requests_manager.submit_request(
                frontier[i], typed_ctxt->get_service_client_ref().get(
								      vector_key, CURRENT_VERSION, true));
          }
        }

        // can add cached node impl here
        auto [node_ids, vector_data] =
          get_requests_manager.get_all_requests();
        assert(node_ids.size() == frontier.size());
        for (size_t i = 0; i < node_ids.size(); i++) {
          std::memcpy(full_precision_emb_buffer, vector_data[i].get(),
                      this->vector_embedding_size);
          float cur_expanded_dist = this->_dist_cmp->compare(
							     aligned_query_T, full_precision_emb_buffer,
							     (uint32_t)_aligned_dim);
          full_retset.emplace_back(node_ids[i], cur_expanded_dist);
          const uint32_t *nbr_ptr = reinterpret_cast<const uint32_t *>(
								       vector_data[i].get() + vector_embedding_size);
          uint32_t num_nbrs = nbr_ptr[0];
          // std::cout << num_nbrs << std::endl;
          const uint32_t *node_nbrs = nbr_ptr + 1;
          compute_dists(node_nbrs, num_nbrs, dist_scratch);
          for (uint32_t m = 0; m < num_nbrs; m++) {
            uint32_t id = node_nbrs[m];
            // not in visited i guess
            if (visited.insert(id).second) {
              float dist = dist_scratch[m];
	      diskann::Neighbor nn(id, dist);
              retset.insert(nn);
            }
          }
        }
      }
      std::sort(full_retset.begin(), full_retset.end());
      for (uint64_t i = 0; i < K; i++) {
        indices[i] = full_retset[i].id;
        if (distances != nullptr)
          distances[i] = full_retset[i].distance;
      }
    }


    

    ~SSDIndexKV() {
    if (pq_data != nullptr)
    {
        delete[] pq_data;
    }      
      if (_nhood_cache_buf != nullptr) {
	delete[] _nhood_cache_buf;
	diskann::aligned_free(_coord_cache_buf);
      }

      diskann::cout << "Clearing scratch" << std::endl;
      diskann::ScratchStoreManager<diskann::SSDThreadData<data_type>> manager(this->_thread_data);
      manager.destroy();
    }
  };
} // namespace cascade
} // namespace derecho
