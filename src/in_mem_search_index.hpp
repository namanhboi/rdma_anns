#include <immintrin.h> // needed to include this to make sure that the code compiles since in DiskANN/include/utils.h it uses this library.
#include <cascade/cascade_interface.hpp>
#include <cascade/service_types.hpp>
#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <filesystem>
#include "serialize_utils.hpp"
#include "tsl/robin_set.h"
#include <boost/dynamic_bitset.hpp>
#include "get_request_manager.hpp"
#include "concurrent_queue.h"
#define MAX_POINTS_FOR_USING_BITSET 10000000

namespace derecho {
namespace cascade {

template<typename data_type>
class GlobalIndex {
  uint32_t num_points;

  // contains the mapping of all the nodes to clusters.
  std::vector<uint8_t> cluster_assignment;
  std::string cluster_data_prefix;
  uint32_t dim;
  uint8_t cluster_id;
  uint8_t num_clusters;


  std::shared_ptr<diskann::Distance<data_type>> dist_fn;
public:
  GlobalIndex(uint32_t dim, uint8_t cluster_id,
              std::string cluster_assignment_bin_file,
              std::string cluster_data_prefix) {
    this->cluster_id = cluster_id;
    this->cluster_data_prefix = cluster_data_prefix;
    // std::cout << "this cluster is " << cluster_id <<std::endl;

    if (!std::filesystem::exists(cluster_assignment_bin_file)) {
      throw std::runtime_error("cluster assignment file :" +
                               cluster_assignment_bin_file + " doesn't exist");
    }
    this->dim = dim;
    // std::cout << "global index dim " << this->dim <<std::endl;
    std::ifstream in(cluster_assignment_bin_file, std::ios::binary);
    in.read((char *)&num_points, sizeof(num_points));
    // std::cout << "number of points is" << num_points << std::endl;
    cluster_assignment.resize(num_points);
    in.read((char *)&num_clusters, sizeof(num_clusters));
    in.read((char *)cluster_assignment.data(), sizeof(uint8_t) * num_points);
    // std::cout << "Done initializing global index" << std::endl;
      this->dist_fn.reset(
			    diskann::get_distance_function<data_type>(diskann::Metric::L2));    
  }

  bool is_in_cluster(uint32_t node_id) const {
    return cluster_assignment[node_id] == this->cluster_id;
  }

  /** get neighbor of a node in the cluster
      returns shared ptr with deleter free_const pointing to the number of
     neighbors (first uint32_t) and the neighbors of the requested node (rest of
     the uint32_ts) this method is only for retrieving neighbors of nodes that
     are in this cluster.
     */
  std::shared_ptr<const uint32_t>
  get_neighbors(uint32_t node_id, DefaultCascadeContextType *typed_ctxt) const {
    if (!is_in_cluster(node_id))
      throw std::runtime_error(
          "Node id " + std::to_string(node_id) + " not in cluster " +
          std::to_string(this->cluster_id) + " " + __PRETTY_FUNCTION__);

    std::string nbr_key =
      cluster_data_prefix + "_nbr_" + std::to_string(node_id);
    auto nbr = typed_ctxt->get_service_client_ref().get(nbr_key,
                                                        CURRENT_VERSION, true);
    auto &nbr_reply = nbr.get().begin()->second.get();
    Blob nbr_blob = std::move(const_cast<Blob &>(nbr_reply.blob));
    nbr_blob.memory_mode = object_memory_mode_t::EMPLACED;
    uint32_t num_nbrs = nbr_blob.size / sizeof(uint32_t);
    std::shared_ptr<const uint32_t> nbr_ptr(
        reinterpret_cast<const uint32_t *>(nbr_blob.bytes), free_const);

    return nbr_ptr;
  }

  // these shared_ptrs must have free_const as deleter
  /*
    this method is only for retrieving embeddings that are in this cluster.
   */
  std::vector<std::shared_ptr<const data_type>>
  retrieve_local_embeddings(const std::vector<uint32_t> &node_ids,
                            DefaultCascadeContextType *typed_ctxt) const {
    std::vector<std::shared_ptr<const data_type>> embeddings;
    // std::cout << node_ids << std::endl;
    for (size_t i = 0; i < node_ids.size(); i++) {
      uint32_t node_id = node_ids[i];
      // std::cout << " starting retrieve process " << std::endl;
      if (!is_in_cluster(node_id)) {
        // std::cout << " not in cluster " << std::endl;
        throw std::runtime_error("Node id " + std::to_string(node_id) +
                                 " not in cluster " +
                                 std::to_string(this->cluster_id) + " " +
                                 __PRETTY_FUNCTION__ + " " + __FILE__);
      }

      std::string emb_key =
        cluster_data_prefix + "_emb_" + std::to_string(node_id);
      // std::cout << "embedding key is " << emb_key << std::endl;
      auto emb = typed_ctxt->get_service_client_ref().get(
          emb_key, CURRENT_VERSION, true);
      auto &emb_reply = emb.get().begin()->second.get();
      Blob emb_blob = std::move(const_cast<Blob &>(emb_reply.blob));
      emb_blob.memory_mode = object_memory_mode_t::EMPLACED;
      // std::cout  << "size of emb blob " << emb_blob.size << std::endl;
      // for (uint32_t j = 0; j < dim; j++) {
      // std::cout << (reinterpret_cast<const data_type*>(emb_blob.bytes))[j] <<
      // std::endl;
      // }
      // std::cout << std::endl;

      std::shared_ptr<const data_type> embedding(
          reinterpret_cast<const data_type *>(emb_blob.bytes), free_const);
      embeddings.emplace_back(std::move(embedding));
    }
    return embeddings;
  }

  std::pair<uint32_t, uint32_t>
  search_global_baseline(DefaultCascadeContextType *typed_ctxt,
                         std::shared_ptr<GreedySearchQuery<data_type>> query,
                         const uint32_t K, const uint32_t L, uint64_t *indices,
                         float *distances) {
    if (!query)
      throw std::invalid_argument("Query cannot be null");
    if (!indices)
      throw std::invalid_argument("Indices array cannot be null");

    if (L == 0 || K == 0 || K > L) {
      throw std::invalid_argument("Invalid L/K parameters");
    }

    if (query->get_candidate_queue_size() == 0) {
      throw std::invalid_argument("query " +
                                  std::to_string(query->get_query_id()) +
                                  " has candidate queue empty");
    }

    diskann::NeighborPriorityQueue candidate_queue;
    candidate_queue.reserve(L);
    tsl::robin_set<uint32_t> inserted_into_pool_rs;
    boost::dynamic_bitset<> inserted_into_pool_bs;

    bool fast_iterate = this->num_points <= MAX_POINTS_FOR_USING_BITSET;

    if (fast_iterate) {
      if (inserted_into_pool_bs.size() < num_points) {
        // hopefully using 2X will reduce the number of allocations.
        auto resize_size = 2 * num_points > MAX_POINTS_FOR_USING_BITSET
                               ? MAX_POINTS_FOR_USING_BITSET
                               : 2 * num_points;
        inserted_into_pool_bs.resize(resize_size);
      }
    }

    // Lambda to determine if a node has been visited
    auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs,
                           &inserted_into_pool_rs](const uint32_t id) {
      return fast_iterate ? inserted_into_pool_bs[id] == 0
                          : inserted_into_pool_rs.find(id) ==
                                inserted_into_pool_rs.end();
    };
    auto mark_visited = [this, fast_iterate, &inserted_into_pool_bs,
                         &inserted_into_pool_rs](uint32_t node_id) {
      if (fast_iterate) {
        inserted_into_pool_bs[node_id] = 1;
      } else {
        inserted_into_pool_rs.insert(node_id);
      }
    };

    // for (uint32_t i = 0; i < query->get_dim(); i++) {
    // std::cout << query->get_embedding_ptr()[i] << " " ;
    // }
    // std::cout << std::endl;
    // Lambda to batch compute query<-> node distnaces
    auto compute_dists =
        [this, query](const std::vector<std::shared_ptr<const data_type>> &embs,
                      std::vector<float> &dists_out) {
          for (size_t i = 0; i < embs.size(); i++) {
            // std::cout << "dim is " << this->dim << std::endl;
            // std::cout << "query emb ptr" << query->get_embedding_ptr() <<
            // std::endl; std::cout << "embedding ptr" << embs[i].get() << " "
            // << *embs[i].get() << std::endl;
            float distance = (dist_fn->compare(
                query->get_embedding_ptr(), embs[i].get(), this->dim));
            // std::cout << "distance is " << distance << std::endl;
            dists_out.push_back(distance);
          }
        };

    std::vector<uint32_t> init_node_ids(query->get_candidate_queue_ptr(),
                                        query->get_candidate_queue_ptr() +
                                            query->get_candidate_queue_size());
    // std::cout << "starting to retrieve local embs" << std::endl;
    std::vector<std::shared_ptr<const data_type>> init_embs =
        retrieve_local_embeddings(init_node_ids, typed_ctxt);
    // std::cout << "done retrievnig local embs " << init_embs.size() <<
    // std::endl;
    std::vector<float> init_distances;

    compute_dists(init_embs, init_distances);
    // std::cout << "done calculating distance for init" << std::endl;
    for (uint32_t i = 0; i < query->get_candidate_queue_size(); i++) {
      diskann::Neighbor nbr(query->get_candidate_queue_ptr()[i],
                            init_distances[i]);
      candidate_queue.insert(nbr);
    }

    uint32_t hops = 0;
    uint32_t cmps = 0;

    // used during search
    std::vector<uint32_t> primary_node_ids;
    std::vector<float> primary_dist;

    std::vector<float> secondary_dist;

    GetRequestManager<data_type, derecho::cascade::ObjectWithStringKey>
        emb_get_request_manager;

    std::unordered_map<
        uint32_t,
        std::shared_ptr<derecho::rpc::QueryResults<const ObjectWithStringKey>>>
        nbr_get_requests;
    // std::cout << "starting search" << std::endl;
    while (candidate_queue.has_unexpanded_node()) {
      diskann::Neighbor node = candidate_queue.closest_unexpanded();
      uint32_t node_id = node.id;
      hops++;
      primary_node_ids.clear();
      primary_dist.clear();

      secondary_dist.clear();

      std::shared_ptr<const uint32_t> neighbors;
      if (!is_in_cluster(node_id)) {
        // if node_id not in cluster then the get request for its neighbors
        // has already been sent and the future obj resides in
        // nbr_get_requests
        // std::cout << "node id " << node_id << "not in current cluster " <<
        // parent->cluster_id << std::endl;
        if (nbr_get_requests[node_id] == nullptr) {
          throw std::runtime_error("request for neighbor not issued even "
                                   "though this is the candidate node");
        }
        auto &reply = nbr_get_requests[node_id]->get().begin()->second.get();
        Blob nbr_blob = std::move(const_cast<Blob &>(reply.blob));
        nbr_blob.memory_mode = derecho::cascade::object_memory_mode_t::EMPLACED;
        std::shared_ptr<const uint32_t> tmp(
            reinterpret_cast<const uint32_t *>(nbr_blob.bytes), free_const);
        neighbors = std::move(tmp);
      } else {
        // std::cout << "start getting neighbors " << std::endl;
        neighbors = get_neighbors(node_id, typed_ctxt);
        // std::cout << "done getting neighbors" << std::endl;
      }

      uint32_t num_nbrs = neighbors.get()[0];
      // std::cout << "number of neighbors " << std::endl;
      const uint32_t *neighbors_ptr = neighbors.get() + 1;
      for (size_t i = 0; i < num_nbrs; i++) {
        uint32_t nbr_node_id = neighbors_ptr[i];
        uint8_t nbr_cluster_id = cluster_assignment[nbr_node_id];
        if (nbr_cluster_id == this->cluster_id) {
          if (is_not_visited(nbr_node_id)) {
            primary_node_ids.push_back(nbr_node_id);
            mark_visited(nbr_node_id);
          }
        } else {
          // std::cout << "node not in cluster " << nbr_node_id << std::endl;
          if (is_not_visited(nbr_node_id)) {
            const std::string &emb_key = cluster_data_prefix + "_emb_" +
                                         std::to_string(nbr_node_id);
            emb_get_request_manager.submit_request(
                nbr_node_id, typed_ctxt->get_service_client_ref().get(
                                 emb_key, CURRENT_VERSION, true));
            // there is probably a better way to do this without sending a
            // request for the neighbors of a a neighboring node
            const std::string &nbr_key = cluster_data_prefix + "_nbr_" +
                                         std::to_string(nbr_node_id);
            nbr_get_requests[nbr_node_id] = std::make_shared<
                derecho::rpc::QueryResults<const ObjectWithStringKey>>(
                std::move(typed_ctxt->get_service_client_ref().get(
                    nbr_key, CURRENT_VERSION, true)));
            mark_visited(nbr_node_id);
          }
        }
      }
      // std::cout << "retrieveing embeddings for nodes in this cluster " <<
      // std::endl;
      std::vector<std::shared_ptr<const data_type>> embs =
          retrieve_local_embeddings(primary_node_ids, typed_ctxt);

      // std::cout << primary_node_ids.size() << std::endl;
      // std::cout << "done retrieveing embeddings for nodes in this cluster "
      // << std::endl;
      primary_dist.reserve(embs.size());
      // std::cout << "start distance calc for nodes in this cluster " <<
      // std::endl;
      compute_dists(embs, primary_dist);
      // std::cout << "done distance calc for nodes in this cluster " <<
      // std::endl;
      // for (float x : primary_dist) {
      // std::cout << x << " " ;
      // }
      // std::cout << std::endl;
      cmps += primary_dist.size();
      for (size_t m = 0; m < primary_node_ids.size(); m++) {
        candidate_queue.insert(
            diskann::Neighbor(primary_node_ids[m], primary_dist[m]));
      }

      // std::cout << "done with nodes in this cluster " << std::endl;

      // get embedding data of whichever requests has arrived.
      // const std::pair<std::vector<uint64_t,>
      //                 std::vector<std::shared_ptr<const data_type>>>
      //     &data_secondary_partition =
      //       emb_get_request_manager.get_available_requests();
      // const std::vector<uint64_t> &id = data_secondary_partition.first;
      // const std::vector<std::shared_ptr<const data_type>> &embeddings =
      // data_secondary_partition.second;

      // secondary_dist.reserve(embeddings.size());
      // compute_dists(embeddings, secondary_dist);
      // cmps += secondary_dist.size();
      // for (size_t m = 0; m < embeddings.size(); m++) {
      //   candidate_queue.insert(diskann::Neighbor(id[m], secondary_dist[m]));
      // }
    }
    for (size_t i = 0; i < std::min(K, (uint32_t)candidate_queue.size()); i++) {
      indices[i] = candidate_queue[i].id;
      if (distances != nullptr)
        distances[i] = candidate_queue[i].distance;
    }
    // std::cout << "number of hops " << hops << std::endl;
    return std::make_pair(hops, cmps);
  }

  // Since in mem is dead, leave this here for reference
  /**
     does task_push search on the global graph index.
     the server that is doing the search is called the primary partition while
     the servers doing distance compute tasks are called secondary partitions.

     If the current node in the candidate queue is stored in the local
     server's storage then explore their neighbors as usual. If the current
     node is not local then issue a distance computation request to the server
     with that node, which in turn will send back node_id, distance pairs
     to be added to the candidate queue as well as its neighbors.
     For this, we need to know which nodes are local and which are not. Also
     there should be a queue where the 2nd partitions can send back results
     which we will check upon each new iteration of the search loop.
   */
  // std::pair<uint32_t, uint32_t> search_task_push(
  //     DefaultCascadeContextType *typed_ctxt,
  //     std::shared_ptr<GreedySearchQuery<data_type>> query,
  //     std::shared_ptr<diskann::ConcurrentQueue<ComputeResult>> compute_res_q,
  //     uint32_t thread_id, const uint32_t K, const uint32_t L, uint32_t *indices,
  //     float *distances) const {
  //   // this can only contain nodes that are in the current cluster, while
  //   // expanding the nodes in here, if a neighbor of the node being expanded is
  //   // not in the cluster then send a compute request or a request to get its
  //   // data. (pull/push mode)
  //   if (!query)
  //     throw std::invalid_argument("Query cannot be null");
  //   if (!compute_res_q)
  //     throw std::invalid_argument("Compute queue cannot be null");
  //   if (!indices)
  //     throw std::invalid_argument("Indices array cannot be null");
  //   // if (!distances) throw std::invalid_argument("Distances array cannot be
  //   // null");

  //   if (L == 0 || K == 0 || K > L) {
  //     throw std::invalid_argument("Invalid L/K parameters");
  //   }

  //   if (query->get_candidate_queue_size() == 0) {
  //     throw std::invalid_argument("query " +
  //                                 std::to_string(query->get_query_id()) +
  //                                 " has candidate queue empty");
  //   }

  //   diskann::NeighborPriorityQueue candidate_queue;
  //   candidate_queue.reserve(L);
  //   tsl::robin_set<uint32_t> inserted_into_pool_rs;
  //   boost::dynamic_bitset<> inserted_into_pool_bs;

  //   bool fast_iterate = this->num_points <= MAX_POINTS_FOR_USING_BITSET;

  //   if (fast_iterate) {
  //     if (inserted_into_pool_bs.size() < num_points) {
  //       // hopefully using 2X will reduce the number of allocations.
  //       auto resize_size = 2 * num_points > MAX_POINTS_FOR_USING_BITSET
  //                              ? MAX_POINTS_FOR_USING_BITSET
  //                              : 2 * num_points;
  //       inserted_into_pool_bs.resize(resize_size);
  //     }
  //   }

  //   // Lambda to determine if a node has been visited
  //   auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs,
  //                          &inserted_into_pool_rs](const uint32_t id) {
  //     return fast_iterate ? inserted_into_pool_bs[id] == 0
  //                         : inserted_into_pool_rs.find(id) ==
  //                               inserted_into_pool_rs.end();
  //   };
  //   auto mark_visited = [this, fast_iterate, &inserted_into_pool_bs,
  //                        &inserted_into_pool_rs](uint32_t node_id) {
  //     if (fast_iterate) {
  //       inserted_into_pool_bs[node_id] = 1;
  //     } else {
  //       inserted_into_pool_rs.insert(node_id);
  //     }
  //   };

  //   // Lambda to batch compute query<-> node distances
  //   auto compute_dists = [this, query](
  //                            std::vector<std::shared_ptr<const data_type>> embs,
  //                            std::vector<float> &dists_out) {
  //     for (size_t i = 0; i < embs.size(); i++) {
  //       dists_out.push_back(dist_fn->compare(query->get_embedding_ptr(),
  //                                                    embs[i].get(), this->dim));
  //     }
  //   };
  //   std::vector<uint32_t> init_node_ids(query->get_candidate_queue_ptr(),
  //                                       query->get_candidate_queue_ptr() +
  //                                           query->get_candidate_queue_size());

  //   std::vector<std::shared_ptr<const data_type>> init_embs =
  //       retrieve_local_embeddings(init_node_ids, typed_ctxt);
  //   std::vector<float> init_distances;

  //   compute_dists(init_embs, init_distances);
  //   for (uint32_t i = 0; i < query->get_candidate_queue_size(); i++) {
  //     diskann::Neighbor nbr(query->get_candidate_queue_ptr()[i],
  //                           init_distances[i]);
  //     candidate_queue.insert(nbr);
  //   }

  //   uint32_t hops = 0;
  //   uint32_t cmps = 0;

  //   // used during search
  //   std::vector<uint32_t> primary_node_ids;
  //   std::vector<float> primary_dist;

  //   std::unordered_map<uint8_t, std::vector<uint32_t>> second_partition_nodes;

  //   while (candidate_queue.has_unexpanded_node()) {
  //     while (!compute_res_q->empty()) {
  //       const ComputeResult &res = compute_res_q->pop();
  //       if (res.get_cluster_sender_id() == 0 &&
  //           res.get_cluster_receiver_id() == 0 && res.get_node().id == 0 &&
  //           res.get_node().distance == 0 && res.get_query_id() == 0 &&
  //           res.get_num_neighbors() == 0 &&
  //           res.get_neighbors_ptr() ==
  //               nullptr) { // not popping an empty queue, check the
  //         // impl of concurrent queue
  //         continue;
  //       }
  //       candidate_queue.insert(res.get_node());
  //     }
  //     diskann::Neighbor node = candidate_queue.closest_unexpanded();
  //     uint32_t node_id = node.id;
  //     hops++;
  //     primary_node_ids.clear();
  //     primary_dist.clear();

  //     second_partition_nodes.clear();

  //     if (!is_in_cluster(node_id)) {
  //       throw std::runtime_error("Node id " + std::to_string(node_id) +
  //                                " not in cluster " +
  //                                std::to_string(this->cluster_id) + " " +
  //                                __PRETTY_FUNCTION__ + " " + __FILE__);
  //     }

  //     std::shared_ptr<const uint32_t> neighbors =
  //         get_neighbors(node_id, typed_ctxt);
  //     uint32_t num_nbrs = neighbors.get()[0];
  //     const uint32_t *neighbors_ptr = neighbors.get() + 1;
  //     for (size_t i = 0; i < num_nbrs; i++) {
  //       uint32_t nbr_node_id = neighbors_ptr[i];
  //       uint8_t nbr_cluster_id = cluster_assignment[nbr_node_id];
  //       if (nbr_cluster_id == this->cluster_id) {
  //         if (is_not_visited(nbr_node_id)) {
  //           primary_node_ids.push_back(nbr_node_id);
  //         }
  //       } else {
  //         if (second_partition_nodes.count(nbr_cluster_id) == 0) {
  //           second_partition_nodes[nbr_cluster_id] = std::vector<uint32_t>();
  //         }
  //         second_partition_nodes[nbr_cluster_id].push_back(nbr_node_id);
  //         mark_visited(nbr_node_id);
  //       }
  //     }

  //     // Mark nodes visited
  //     for (auto id : primary_node_ids) {
  //       mark_visited(id);
  //     }

  //     std::vector<std::shared_ptr<const data_type>> embs =
  //         retrieve_local_embeddings(primary_node_ids, typed_ctxt);
  //     primary_dist.reserve(embs.size());
  //     compute_dists(embs, primary_dist);
  //     cmps += primary_dist.size();
  //     for (size_t m = 0; m < primary_node_ids.size(); m++) {
  //       candidate_queue.insert(
  //           diskann::Neighbor(primary_node_ids[m], primary_dist[m]));
  //     }
  //     float min_distance = candidate_queue[candidate_queue.size() - 1].distance;

  //     for (auto &[cluster_id, node_ids] : second_partition_nodes) {
  //       for (uint32_t node_id : node_ids) {
  //         parent->batch_thread->push_compute_query(
  //             node_id, query->get_query_id(), min_distance, cluster_id,
  //             thread_id);
  //       }
  //     }
  //   }
  //   for (size_t i = 0; i < std::min(K, (uint32_t)candidate_queue.size()); i++) {
  //     indices[i] = candidate_queue[i].id;
  //     if (distances != nullptr)
  //       distances[i] = candidate_queue[i].distance;
  //   }
  //   // std::cout << "number of hops " << hops << std::endl;
  //   return std::make_pair(hops, cmps);
  // }
};

} // namespace cascade
} // namespace derecho
