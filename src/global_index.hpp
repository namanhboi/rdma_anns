// #include <cascade/object.hpp>
// #include <chrono>
// #include <immintrin.h> // needed to include this to make sure that the code compiles since in DiskANN/include/utils.h it uses this library.
// #include <cascade/service_types.hpp>
// #include <cascade/user_defined_logic_interface.hpp>
// #include <cascade/utils.hpp>
// #include <iostream>
// #include <memory>
// #include "ann_exception.h"
// #include "concurrent_queue.h"
// #include "neighbor.h"
// #include "scratch.h"
// #include "serialize_utils.hpp"
// #include "tsl/robin_set.h"
// #include <boost/dynamic_bitset.hpp>
// #include <stdexcept>
// #include <stdexcept>
// #define MAX_POINTS_FOR_USING_BITSET 10000000
// namespace derecho {
// namespace cascade {
// /**



// */
// template <typename data_type> class GlobalIndex {
//   uint32_t num_points;
//   // set of node ids that is in the current server.
//   std::unordered_set<uint32_t> local_nodes;
//   DefaultCascadeContextType *typed_ctxt;

//   std::unique_ptr<diskann::Distance<data_type>> dist_fn;
//   std::string cluster_prefix;
//   uint32_t dim;
//   uint32_t cluster_id;
// public:
//   GlobalIndex(DefaultCascadeContextType *typed_ctxt, int num_points, std::string cluster_prefix, uint32_t dim, uint32_t cluster_id) {
//     dist_fn.reset(
//         (diskann::Distance<data_type> *)
//             diskann::get_distance_function<data_type>(diskann::Metric::L2));
//     this->cluster_prefix = cluster_prefix;
//     this->num_points = num_points;
//     this->dim = dim;
//     this->cluster_id = cluster_id;
//     // retrieve data like all nodes in cluster
//   }

//   bool is_in_cluster(uint32_t node_id) {
//     if (local_nodes.count(node_id))
//       return true;
//     return false;
//   }

//   std::vector<uint32_t> get_neighbors(uint32_t node_id) {
//     if (!is_in_cluster(node_id))
//       throw std::runtime_error(
//           "Node id " + std::to_string(node_id) + " not in cluster " +
//           std::to_string(this->cluster_id) + " " + __PRETTY_FUNCTION__);
    
//     std::string nbr_key = this->cluster_prefix + "/nbr_" + std::to_string(node_id);
//     auto nbr = typed_ctxt->get_service_client_ref().get(nbr_key,
//                                                         CURRENT_VERSION, true);
//     auto &nbr_reply = nbr.get().begin()->second.get();
//     Blob nbr_blob = std::move(const_cast<Blob &>(nbr_reply.blob));
//     nbr_blob.memory_mode =
//       object_memory_mode_t::EMPLACED; // this transfer ownership of memory to
//     // this function?
//     uint32_t num_nbrs = nbr_blob.size / sizeof(uint32_t);
//     const uint32_t *start_nbr =
//       reinterpret_cast<const uint32_t *>(nbr_blob.bytes);
//     std::vector<uint32_t> nbrs(start_nbr, start_nbr + num_nbrs);
//     return nbrs;
//   }
  
//   std::shared_ptr<data_type[]>
//   retrieve_embeddings(const std::vector<uint32_t> &node_ids) {
//     std::shared_ptr<data_type[]> emb_data = std::make_shared<data_type[]>(node_ids.size() * this->dim);
//     for (size_t i = 0; i < node_ids.size(); i++) {
//       uint32_t node_id = node_ids[i];
//       if (!is_in_cluster(node_id))
//         throw std::runtime_error(
//             "Node id " + std::to_string(node_id) + " not in cluster " +
//             std::to_string(this->cluster_id) + " " + __PRETTY_FUNCTION__ + " " + __FILE__ + " "  +__LINE__);
            
//       std::string emb_key = this->cluster_prefix + "/emb_" + std::to_string(node_id);
//       auto emb = typed_ctxt->get_service_client_ref().get(emb_key,
//                                                           CURRENT_VERSION, true);
//       auto &emb_reply = emb.get().begin()->second.get();
//       Blob emb_blob = std::move(const_cast<Blob &>(emb_reply.blob));
//       emb_blob.memory_mode =
//         object_memory_mode_t::EMPLACED; // this transfer ownership of memory to
//       // this function?
    
//       std::memcpy(emb_data.get() + i * this->dim, emb_blob.bytes, this->dim * sizeof(data_type));
//     }
//     return emb_data;
//   }
  
      

//   /**
//      does greedy search on the global graph index.
//      the server that is doing the search is called the primary partition while
// the servers doing distance compute tasks are called secondary partitions.

//      If the current node in the candidate queue is stored in the local server's
// storage then explore their neighbors as usual. If the current node is not local
// then issue a distance computation request to the server with that node, which in
// turn will send back node_id and distance pairs to be added to the candidate
// queue.

// For this, we need to know which nodes are local and which are not. Also there
// should be a queue where the 2nd partitions can send back results which we will
// check upon each new iteration of the search loop.
//   */
//   std::pair<uint32_t, uint32_t>
//   search(std::shared_ptr<GreedySearchQuery<data_type>> query,
//          std::shared_ptr<diskann::ConcurrentQueue<diskann::Neighbor>>
//              compute_res_q,
//          const uint32_t L, const uint32_t R) {
//     // this can only contain nodes that are in the current cluster, while
//     // expanding the nodes in here, if a neighbor of the node being expanded is
//     // not in the cluster then send a compute request or a request to get its
//     // data. (pull/push mode)
    
//     diskann::NeighborPriorityQueue candidate_queue;
//     candidate_queue.reserve(L);
//     tsl::robin_set<uint32_t> inserted_into_pool_rs;
//     boost::dynamic_bitset<> inserted_into_pool_bs;



//     bool fast_iterate = this->num_points <= MAX_POINTS_FOR_USING_BITSET;

//     if (fast_iterate)
//     {
//         if (inserted_into_pool_bs.size() < num_points)
//         {
//             // hopefully using 2X will reduce the number of allocations.
//             auto resize_size =
//                 2 * num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 * num_points;
//             inserted_into_pool_bs.resize(resize_size);
//         }
//     }

//     // Lambda to determine if a node has been visited
//     auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
//         return fast_iterate ? inserted_into_pool_bs[id] == 0
//                             : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
//     };

//     // Lambda to batch compute query<-> node distances
//     auto compute_dists = [this, query](std::shared_ptr<data_type[]> embs,
//                                        std::vector<float> &dists_out) {
      
//       for (size_t i = 0; i < dists_out.size(); i++) {
//         dists_out[i] = this->dist_fn->compare(
// 					      query->get_embedding_ptr(), embs.get() + i * this->dim, this->dim);
//       }
//     };
//     std::vector<uint32_t> init_node_ids(query->get_candidate_queue_ptr(),
//                                         query->get_candidate_queue_ptr() +
//                                         query->get_candidate_queue_size());
    
//     std::shared_ptr<data_type[]> init_embs = retrieve_embeddings(init_node_ids);
//     std::vector<float> init_distances(query->get_candidate_queue_size());

//     compute_dists(init_embs, init_distances);
//     for (uint32_t i = 0; i < query->get_candidate_queue_size(); i++) {
//       diskann::Neighbor nbr(query->get_candidate_queue_ptr()[i], init_distances[i]);
//       candidate_queue.insert(nbr);
//     }

//     uint32_t hops = 0;
//     uint32_t cmps = 0;

//     //used during search
//     std::vector<uint32_t> node_ids;
//     std::vector<double> dist;

//     while (candidate_queue.has_unexpanded_node()) {
//       while (!compute_res_q->empty()) {
//         diskann::Neighbor n = compute_res_q->pop();
//         if (n.id != 0 && n.distance != -1) { // not popping an empty queue
//           candidate_queue.insert(n);
//         }
//       }
//       diskann::Neighbor node =  candidate_queue.closest_unexpanded();
//       uint32_t node_id = node.id;

//       node_ids.clear();
//       dist.clear();
      
//       if (!is_in_cluster(node_id)) {
//         throw std::runtime_error(
//             "Node id " + std::to_string(node_id) + " not in cluster " +
//             std::to_string(this->cluster_id) + " " + __PRETTY_FUNCTION__ + " " +
//             __FILE__ + " " + __LINE__);
//       }

//       std::vector<uint32_t> neighbors = get_neighbors(node_id);
//       for (size_t i = 0; i < neighbors.size(); i++) {
//         uint32_t nbr_id = neighbors[i];
//         if (is_in_cluster(nbr_id)) {
//           if (is_not_visited(nbr_id)) {
//             node_ids.push_back(nbr_id);
//           }
//         } else {
          
//         }
        
//       }

//               // Mark nodes visited
//         for (auto id : id_scratch)
//         {
//             if (fast_iterate)
//             {
//                 inserted_into_pool_bs[id] = 1;
//             }
//             else
//             {
//                 inserted_into_pool_rs.insert(id);
//             }
//         }

//     }
    
//   }

  




  


  

  
// };


// } // namespace cascade
// } // namespace derecho
