/*
  There will be 2 VCSS subgroups in total, each with an object pool:
"/anns/head_search", "/anns/global"

  The first subgroup is used for head index search and will have 1 big shard.
This is because head search requires the same data which is the head index. The
index for head search udl is stored in a file and will be loaded in once the the
udl is initialized. To issue a head index search : "/anns/head_search/query_i"

  The second subgroup is used for global search and will have M shards
corresponding to the M partitions. We don't need an affinity set regex for
"/anns/head_search" because we want to randomly distribute the search query.

  For "/anns/global", we want the regex "cluster_[0-9]+". Before we run any
query, we will load the appropriate vector embeddings and neighbors belonging to
each pre-partitioned partition of the whole graph. For example:
"/anns/global/data/cluster_{cluster_id}_emb_{node_id}"
"/anns/global/data/cluster_{cluster_id}_nbr_{node_id}" When we want to do search
at cluster i, we do a trigger put at "/anns/global/search/cluster_i_query_j"

Results of search will be put at "/anns/results/client_{client_id}". Each will
get its own object pool in subgroup 3. Each client's result object pool will
have the same number of shards as the number of servers. This is so that when we
notify, we don't have to do any replication in a shard i hope?
**/


#define UDL1_OBJ_POOL "/anns/head_search"
#define UDL1_SUBGROUP_INDEX 1
#define UDL1_PATHNAME "/anns/head_search"
// put queries here : /anns/head_search/query_i to trigger head index search

#define UDL2_SUBGROUP_INDEX 1
#define UDL2_OBJ_POOL "/anns/global"
#define UDL2_DATA_PREFIX "/anns/global/data"
#define UDL2_PATHNAME "/anns/global/search"
// both UDL2_DATA_PREFIX and UDL2_PATHNAME are a part of the same object pool /anns/global

// put greedy search queries (defined in serialize utils) here:
// /anns/global/search/cluste_/query_i to trigger global search

#define RESULTS_OBJ_POOL_SUBGROUP_INDEX 0
#define RESULTS_OBJ_POOL_PREFIX "/anns/results"
// each client will have a separate object pool: /anns/results/{client_id}


#define AFFINITY_SET_REGEX "cluster_[0-9]+"
