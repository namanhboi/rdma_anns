#include "omp.h"

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"
#include "disk_utils.h"

// template<typename T>
// bool build_index(const char *dataFilePath, const char *indexFilePath, const char *indexBuildParameters,
                 // pipeann::Metric m, bool singleFile) {
  // return pipeann::build_disk_index<T>(dataFilePath, indexFilePath, indexBuildParameters, m, singleFile);
// }



// build/tests/utils/gen_random_slice uint8 ${DATA_PATH} ${INDEX_PREFIX}_SAMPLE_RATE_0.01 0.01


template <typename T>
void create_indices(const std::string &base_file,
                    const std::string &index_path_prefix, uint32_t num_clusters,
                    const char *indexBuildParameters,
                    pipeann::Metric _compareMetric, bool single_file_index) {
  create_random_cluster_tag_files<T>(base_file, index_path_prefix,
                                     num_clusters);
  create_random_cluster_base_files<T>(base_file, index_path_prefix,
                                      num_clusters);

  create_random_cluster_disk_indices<T>(index_path_prefix, num_clusters,
                                        indexBuildParameters, _compareMetric,
                                        single_file_index);
  create_cluster_random_slices<T>(base_file, index_path_prefix,
                                        num_clusters);

  create_cluster_in_mem_indices<T>(base_file, index_path_prefix, num_clusters,
                                   indexBuildParameters, _compareMetric);
}

/**
   export DATA_PATH=/home/nam/big-ann-benchmarks/data/bigann/base.1B.u8bin.crop_nb_10000000
   export INDEX_PREFIX=/home/nam/big-ann-benchmarks/data/bigann/tes/pipeann_10M
   ./create_scatter_gather_indices uint8 ${DATA_PATH} ${INDEX_PREFIX} 32 64 0.15 20 16 l2 0 2
*/
int main(int argc, char **argv) {
  if (argc != 12) {
    std::cout << " number of args is " << argc << std::endl;
    for (auto i = 0; i < argc; i++) {
      std::cout << std::string(argv[i]) << std::endl;
    }
    std::cout << "Usage: " << argv[0]
              << " <data_type (float/int8/uint8)>  <data_file.bin>"
                 " <index_prefix_path> <R>  <L>  <B>  <M>  <T>"
                 " <similarity metric (cosine/l2) case sensitive>."
                 " <single_file_index (0/1)>"
		 " <num_clusters>"
                 " See README for more information on parameters."
              << std::endl;
  } else {
    std::string data_type(argv[1]);
    std::string base_file(argv[2]);
    std::string index_path_prefix(argv[3]);
    std::string R(argv[4]);
    std::string L(argv[5]);
    std::string B(argv[6]);
    std::string M(argv[7]);
    std::string T(argv[8]);
    std::string dist_metric(argv[9]);
    bool single_file_index = std::atoi(argv[10]) != 0;
    int num_clusters = std::stoull(argv[11]);

    std::string params = R + " " + L + " " + B + " " + M + " " + T;

    pipeann::Metric m = dist_metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
    if (dist_metric != "l2" && m == pipeann::Metric::L2) {
      std::cout << "Metric " << dist_metric << " is not supported. Using L2" << std::endl;
    }
    
    if (data_type == "float")
      create_indices<float>(base_file, index_path_prefix, num_clusters, params.c_str(), m, single_file_index);
    else if (data_type == "int8")
      create_indices<int8_t>(base_file, index_path_prefix, num_clusters,
                             params.c_str(), m, single_file_index);
    else if (data_type == "uint8")
      create_indices<uint8_t>(base_file, index_path_prefix, num_clusters,
                              params.c_str(), m, single_file_index);
    else
      std::cout << "Error. wrong file type" << std::endl;
  }
}
