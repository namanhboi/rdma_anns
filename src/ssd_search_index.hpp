#include <immintrin.h> // needed to include this to make sure that the code compiles since in DiskANN/include/utils.h it uses this library.
#include "linux_aligned_file_reader.h"
#include "neighbor.h"
#include "pq_flash_index.h"
#include "utils.h"
#include <cascade/service_types.hpp>
#include <memory>
#include "get_request_manager.hpp"
#include "udl_path_and_index.hpp"

#define BEAMWIDTH 1




namespace derecho {
namespace cascade {

/**
   currently these 2 classes are barebones impl for benchmarking whether cascade
   persistent kv can function as the abstracted datastore for ssd based anns.

Both don't support any medoid data stuff, only 1 entry point, and also doesn't
have any caching right now

*/

// basically a wrapper for diskann flash index
template <typename data_type> class SSDIndexFileSystem {
  std::unique_ptr<diskann::PQFlashIndex<data_type>> _pFlashIndex;

public:
  /**
     num_search_threads is needed for diskann to preallocate memory for search
   */
  SSDIndexFileSystem(const std::string &index_path_prefix,
                     uint64_t num_search_threads) {
    std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
    _pFlashIndex = std::make_unique<diskann::PQFlashIndex<data_type>>(
        reader, diskann::Metric::L2);
    int res = _pFlashIndex->load(num_search_threads, index_path_prefix.c_str());
    if (res != 0) {
      throw std::runtime_error("cant load ssd index");
    }
  }

  void search(const data_type *query, const int64_t K, const uint64_t L,
              uint64_t *indices, float *distances) {
    _pFlashIndex->cached_beam_search(query, K, L, indices, distances, BEAMWIDTH,
                                     false, nullptr);
  }
};

  template <typename data_type> class SSDIndexKV {
    std::string cluster_data_prefix;
    
    std::unique_ptr<uint8_t[]> pq_data;
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
      uint8_t *pq_data_ptr;
      std::string pq_table_bin =
        std::string(index_path_prefix) + "_pq_pivots.bin";
      std::string pq_compressed_vectors =
        std::string(index_path_prefix) + "_pq_compressed.bin";
      size_t npts_u64, nchunks_u64;
      diskann::load_bin<uint8_t>(pq_compressed_vectors, pq_data_ptr, npts_u64,
				 nchunks_u64);
      this->num_points = npts_u64;
      this->num_chunks = nchunks_u64;

      pq_data.reset(pq_data_ptr);

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

      // setup thread data
      for (uint64_t i; i < num_threads; i++) {
	diskann::SSDThreadData<data_type> *data =
          new diskann::SSDThreadData<data_type>(this->dim, 4096);
	this->_thread_data.push(data);
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

      // all data necessary for querying and pq operations will be accessed
      // through these 2 pointers.
      auto query_scratch = &(data->scratch);
      // this contains preallocated ptrs to pq table, pq processed query, etc
      auto pq_query_scratch = query_scratch->pq_scratch();

      query_scratch->reset();

      // aligned malloc memset to 0, need to copy query to here to do
      // computation since a lot of operations only work on aligned mem address
      data_type *aligned_query_T = query_scratch->aligned_query_T();

      // aligned malloc for pq computation,
      // converts whatever type the query is to a float,
      // will contain the same data as aligned_query_T
      float *query_float = pq_query_scratch->aligned_query_float;

      // whether this is used or not depends on there is a rotation matrix file
      // that has index_path_prefix as a path prefix
      float *query_rotated = pq_query_scratch->rotated_query;

      for (size_t i = 0; i < this->dim; i++) {
	aligned_query_T[i] = query[i];
      }

      // copies data from aligned query to query float and query rotated
      pq_query_scratch->initialize(this->dim, aligned_query_T);

      // this is where the full precision embeddings will be copied into to do
      // full distance computation with
      data_type *full_precision_emb_buffer = query_scratch->coord_scratch;

      // not sure why this is necessary in pq_flash_index search method since
      // won't we be overwriting it constantly?
      _mm_prefetch((char *)full_precision_emb_buffer, _MM_HINT_T1);

      pq_table.preprocess_query(query_rotated);

      float *pq_dists = pq_query_scratch->aligned_dist_scratch;
      pq_table.populate_chunk_distances(query_rotated, pq_dists);
      // after this, pq query distance to all centroids in pq table is
      // calcucated in pq_dists, and pq_dists is used for fast look up for pq
      // distance calculation

      // this is where you write the pq coordinates of points to do pq
      // computation with
      float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
      uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;


      // lambda to batch compute query<-> node distances in PQ space from pq data
      // loading from file
      auto compute_dists = [this, pq_coord_scratch,
                          pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                    float *dists_out) {
	diskann::aggregate_coords(ids, n_ids, this->pq_data.get(),
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
      // done with initilizing candidate queue



      // used to determine which nodes to do get request, reason this exist is
      // because we might want to increase beamwidth in the future
      std::vector<uint32_t> frontier;
      frontier.reserve(2 * BEAMWIDTH);
      // pointer is to the start of full embedding + nbr ids blob bytes. The
      // embeddings will be copied to full_precision_emb_buffer to do distance
      // compute
      std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
      frontier_nhoods.reserve(2 * BEAMWIDTH);


      GetRequestManager<uint8_t, ObjectWithStringKey> get_requests_manager;

      while (retset.has_unexpanded_node()) {
	frontier.clear();
	frontier_nhoods.clear();

	auto nbr = retset.closest_unexpanded();
	frontier.push_back(nbr.id);
        if (!frontier.empty()) {
          for (size_t i = 0; i < frontier.size();i++) {
            const std::string vector_key = this->cluster_data_prefix + "/vec_" + std::to_string(frontier[i]);
            get_requests_manager.submit_request(
                frontier[i], typed_ctxt->get_service_client_ref().get(
								      vector_key, CURRENT_VERSION, true));
          }

          // can add cached node impl here
          auto [node_ids, vector_data] =
            get_requests_manager.get_all_requests();
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
      }
      std::sort(full_retset.begin(), full_retset.end());
      for (uint64_t i = 0; i < K; i++) {
        indices[i] = full_retset[i].id;
        if (distances != nullptr)
          distances[i] = full_retset[i].distance;
      }
    }

    ~SSDIndexKV() {
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
