#include "disk_utils.h"
#include "utils.h"
#include "cached_io.h"
#include <memory>
#include <random>
#include <stdexcept>
#include "aux_utils.h"

template<typename T, typename TagT>
void create_random_cluster_tag_files(const std::string &base_file,
                                     const std::string &index_path_prefix,
                                     uint32_t num_clusters) {
  std::random_device rd; 
  std::mt19937 gen(rd()); 

  size_t base_num, base_dim;
  pipeann::get_bin_metadata(base_file, base_num, base_dim);

  std::vector<std::string> tag_files;

  for (auto i = 0; i < num_clusters; i++) {
    std::string tag_file =
      index_path_prefix + "_cluster" + std::to_string(i) + ".tags";
    tag_files.emplace_back(tag_file);
    
  }

  std::uniform_int_distribution<uint32_t> cluster_id_gen(0, num_clusters - 1);
  std::vector<std::vector<TagT>> cluster_ids;
  cluster_ids.resize(num_clusters);

  for (uint32_t i = 0; i < base_num; i++) {
    uint32_t cluster_id = cluster_id_gen(gen);
    cluster_ids[cluster_id].push_back(i);
  }

  for (auto i = 0; i < num_clusters; i++) {
    if (!file_exists(tag_files[i])) {
      pipeann::save_bin<TagT>(tag_files[i], cluster_ids[i].data(),
                                  cluster_ids[i].size(), 1);
    } else {
      LOG(INFO) << "File already exists: " << tag_files[i];
    }
  }
}


template<typename T, typename TagT>
void create_base_from_tag(const std::string &base_file,
                          const std::string &tag_file,
                          const std::string &output_base_file) {
  std::ifstream base_reader(base_file.c_str());
  base_reader.seekg(0, std::ios::beg);

  size_t tag_npts, tag_dim;
  std::vector<TagT> tag_ids;
  pipeann::load_bin<TagT>(tag_file, tag_ids, tag_npts, tag_dim);
    
  
  std::ofstream writer(output_base_file.c_str(), std::ios::binary);

  uint32_t npts_u32, nd_u32;

  base_reader.read((char *) &npts_u32, sizeof(uint32_t));
  base_reader.read((char *) &nd_u32, sizeof(uint32_t));
  LOG(INFO) << "Loading base " << base_file << ". #points: " << npts_u32 << ". #dim: " << nd_u32 << ".";
  writer.write((char *) &tag_npts, sizeof(uint32_t));
  writer.write((char *)&nd_u32, sizeof(uint32_t));

  std::unique_ptr<T[]> cur_row = std::make_unique<T[]>(nd_u32);


  size_t header_size = sizeof(uint32_t) * 2;
  size_t embedding_size = nd_u32 * sizeof(T);
  
  
  for (TagT id : tag_ids) {
    base_reader.seekg(header_size + embedding_size * id, base_reader.beg);
    base_reader.read((char *)cur_row.get(), embedding_size);
    writer.write((char *)cur_row.get(), embedding_size);
  }
  base_reader.close();
  writer.close();
}


template<typename T, typename TagT>
void create_random_cluster_base_files(const std::string &base_file,
                                      const std::string &index_path_prefix,
                                      uint32_t num_clusters) {
  std::vector<std::string> tag_files;
  for (uint32_t i = 0; i < num_clusters; i++) {
    std::string tag_file =
      index_path_prefix + "_cluster" + std::to_string(i) + ".tags";
    if (!file_exists(tag_file)) {
      throw std::runtime_error("Tag file doesn't exist " + tag_file);
    }
    tag_files.emplace_back(tag_file);
  }
  int i = 0;
  for (const auto &tag_file : tag_files) {
    std::string output_base_file =
      index_path_prefix + "_cluster" + std::to_string(i) + ".bin";
    if (!file_exists(output_base_file)) {
      create_base_from_tag<T, TagT>(base_file, tag_file, output_base_file);
    } else {
      LOG(INFO) << "Base file already exists " << output_base_file;
    }
    i++;
  }
}


template <typename T, typename TagT>
void create_random_cluster_disk_indices(const std::string &index_path_prefix,
                                        uint32_t num_clusters,
                                        const char *indexBuildParameters,
                                        pipeann::Metric _compareMetric,
                                        bool single_file_index) {
  std::vector<std::string> tag_files;
  std::vector<std::string> base_files;
  for (uint32_t i = 0; i < num_clusters; i++) {
    std::string cluster_base_file =
      index_path_prefix + "_cluster" + std::to_string(i) + ".bin";
    if (!file_exists(cluster_base_file)) {
      throw std::runtime_error("base file doesn't exist " + cluster_base_file);
    }
    base_files.push_back(cluster_base_file);


    std::string cluster_tag_file =
      index_path_prefix + "_cluster" + std::to_string(i) + ".tags";
    if (!file_exists(cluster_tag_file)) {
      throw std::runtime_error("tag file doesn't exist " + cluster_tag_file);
    }
    tag_files.push_back(cluster_tag_file);
  }
  for (uint32_t i = 0; i < num_clusters; i++) {
    std::string index_path = index_path_prefix + "_cluster" + std::to_string(i);
    pipeann::build_disk_index<T, TagT>(
        base_files[i].c_str(), index_path.c_str(), indexBuildParameters,
				       _compareMetric, single_file_index, nullptr, true);

  }
}


template
void create_random_cluster_tag_files<float>(
    const std::string &base_file, const std::string &index_path_prefix,
						uint32_t num_clusters);

template
void create_random_cluster_tag_files<uint8_t>(
    const std::string &base_file, const std::string &index_path_prefix,
						uint32_t num_clusters);

template
void create_random_cluster_tag_files<int8_t>(
    const std::string &base_file, const std::string &index_path_prefix,
						uint32_t num_clusters);

template void
create_random_cluster_base_files<float>(const std::string &base_file,
                                        const std::string &index_path_prefix,
                                        uint32_t num_clusters);

template void
create_random_cluster_base_files<uint8_t>(const std::string &base_file,
                                        const std::string &index_path_prefix,
                                        uint32_t num_clusters);

template void
create_random_cluster_base_files<int8_t>(const std::string &base_file,
                                        const std::string &index_path_prefix,
                                        uint32_t num_clusters);

template void create_random_cluster_disk_indices<float>(
    const std::string &index_path_prefix, uint32_t num_clusters,
    const char *indexBuildParameters, pipeann::Metric _compareMetric,
							bool single_file_index);
template void create_random_cluster_disk_indices<int8_t>(
    const std::string &index_path_prefix, uint32_t num_clusters,
    const char *indexBuildParameters, pipeann::Metric _compareMetric,
							 bool single_file_index);

template void create_random_cluster_disk_indices<uint8_t>(const std::string &index_path_prefix, uint32_t num_clusters, const char *indexBuildParameters, pipeann::Metric _compareMetric, bool single_file_index);


