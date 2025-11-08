#pragma once

#include <string>
#include <cstdint>
#include "utils.h"
#include "omp.h"


constexpr double MEM_INDEX_SAMPLING_RATE = 0.01;
constexpr float MEM_INDEX_ALPHA = 1.2;

/**
   from the basefile, randomly assign each embedding to a cluster from 0 -
   (num_clusters - 1) to create the tag files for each cluster. The indices in
   the tag files indexes into the base file to get embeddings for each cluster.

   The tag files are index_path_prefi + "_cluster{id}.tags". If the tag files
   already exists then don't create new ones

return true if new files are written, false if files already exists

 */
template<typename T, typename TagT=uint32_t>
void create_random_cluster_tag_files(const std::string &base_file,
                                     const std::string &index_path_prefix,
                                     uint32_t num_clusters);


template <typename T, typename TagT=uint32_t>
void create_base_from_tag(const std::string &base_file,
                          const std::string &tag_file,
                          const std::string &output_base_file);


/**
   check that num_clusters tag files exists . Then
   from each tag file and the basefile, create the base files for the clusters.
   base files name: index_path_prefix + "_cluster{id}.bin"
   If files already exists then don't need to recreate
   
*/
template<typename T, typename TagT=uint32_t>
void create_random_cluster_base_files(const std::string &base_file,
                                      const std::string &index_path_prefix,
                                      uint32_t num_clusters);

/**
   create disk index files from the base files, must check that they exist. If
   the disk index files arealdy exists then don't need to recreate
*/
template <typename T, typename TagT = uint32_t>
void create_random_cluster_disk_indices(const std::string &index_path_prefix,
                                        uint32_t num_clusters,
                                        const char *indexBuildParameters,
                                        pipeann::Metric _compareMetric,
                                        bool single_file_index);


template <typename T, typename TagT = uint32_t>
void create_cluster_random_slices(const std::string &base_file,
                                  const std::string &index_path_prefix,
                                  uint32_t num_clusters);

template <typename T, typename TagT = uint32_t>
void create_cluster_in_mem_indices(const std::string &base_file,
                                   const std::string &index_path_prefix,
                                   uint32_t num_clusters,
                                   const char *indexBuildParameters,
                                   pipeann::Metric metric);

/**
   create a mem index file from disk index file, won't work properly with frozen points
*/
template <typename T, typename TagT = uint32_t>
void write_graph_index_from_disk_index(const std::string &index_path_prefix,
                                     const std::string &graph_path);

template <typename T, typename TagT=uint32_t>
void dumb_way(const std::string &index_path_prefix,
              const std::string &graph_path);

std::vector<std::vector<int>> load_graph_file(const std::string& graph_path);

/**
   loc files are used to find out what order a node was writtin in a cluster
   disk index file for state send (global graph)
*/
void write_partitions_to_loc_files(const std::vector<std::vector<uint32_t>> &clusters,
                                 const std::string &cluster_path_prefix);


// void create_graph_from_tag(const std::string &graph_file,
                           // const std::string &tag_file);



template<typename T, typename TagT = uint32_t>
void create_base_files_from_tags(const std::string &base_file,
                                 const std::string &output_index_path_prefix,
                                 int num_partitions);

void create_graphs_from_tags(const std::string &source_graph_path,
                             const std::string &output_index_path_prefix,
                             int num_partitions);


template<typename T>
void create_and_write_partitions_to_loc_files(
    const std::string &graph_path, const std::string &output_index_path_prefix,
					      int num_partitions);


template <typename T, typename TagT = uint32_t>
void create_disk_indices(const std::string &output_index_path_prefix,
                         int num_partitions);


void create_partition_assignment_file(
				      const std::string &output_index_path_prefix, int num_partitions);


void write_partitions_to_txt_files(const std::string &output_index_path_prefix,
                                   int num_partitions);


template <typename T>
void create_pq_data(const std::string &base_path,
                    const std::string &index_path_prefix,
                    const size_t num_pq_chunks, pipeann::Metric metric);



void create_partition_assignment_symlinks(const std::string &index_path_prefix,
                                          int num_partitions);

void create_pq_data_symlink(const std::string &index_path_prefix,
                            const std::string &output_path_prefix,
                            int num_partitions);


void create_mem_index_symlink(const std::string &index_path_prefix,
                            const std::string &output_path_prefix,
                              int num_partitions);

template <typename T>
void create_mem_index_from_disk(const std::string &index_path_prefix, int R,
                                int L, int num_threads, pipeann::Metric metric);

template <typename T>
void create_slice_from_disk(const std::string &data_path,
                            const std::string &index_path_prefix);



template<typename T>
void create_and_write_overlap_partitions_to_loc_files(
    const std::string &base_file, int num_partitions, double overlap,
						      const std::string &output_index_path_prefix);



/**
   new format for partition assignment file:
   size_t num_points
   uint8_t num_partitions
   for each point:
   uint8_t num_home_partitions
   uint8_t home_partition[num_home_partitions]
*/
void create_overlap_partition_assignment_file(
					      const std::string &output_index_path_prefix, int num_partitions);


void load_partition_assignment_file(
    const std::string &partition_assignment_file,
    std::vector<std::vector<uint8_t>> &partition_assignment,
					    uint8_t &num_partitions);


void sort_and_rewrite_partition_loc_files(
					  const std::string &output_index_path_prefix, int num_partitions);


void load_parlayann_graph_file(const std::string &graph_file,
                               std::vector<std::vector<uint32_t>> &graph);


void write_graph_file_from_parlayann_graph_file(
    const std::string &parlayann_graph_file, const std::vector<uint32_t> &ids,
						const std::string &output_graph_file);



template <typename T>
int build_in_memory_index(const std::string &data_path,
                          const std::string &tags_file, const unsigned R,
                          const unsigned L, const float alpha,
                          const std::string &save_path,
                          const unsigned num_threads, bool dynamic_index,
                          bool single_file_index, pipeann::Metric distMetric);
