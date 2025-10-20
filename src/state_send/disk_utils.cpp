#include "disk_utils.h"
#include "aux_utils.h"
#include "cached_io.h"
#include "index.h"
#include "log.h"
#include "query_buf.h"
#include "utils.h"
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include "graph_partitioning_utils.h"
#include "partition_and_pq.h"





template <typename T, typename TagT>
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

template <typename T, typename TagT>
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

  base_reader.read((char *)&npts_u32, sizeof(uint32_t));
  base_reader.read((char *)&nd_u32, sizeof(uint32_t));
  LOG(INFO) << "Loading base " << base_file << ". #points: " << npts_u32
            << ". #dim: " << nd_u32 << ".";
  writer.write((char *)&tag_npts, sizeof(uint32_t));
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

template <typename T, typename TagT>
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
    
    if (!file_exists(index_path + "_disk.index")) {
      pipeann::build_disk_index<T, TagT>(
          base_files[i].c_str(), index_path.c_str(), indexBuildParameters,
					 _compareMetric, single_file_index, nullptr, true);
    }
  }
}

template <typename T, typename TagT>
void create_cluster_random_slices(const std::string &base_file,
                                  const std::string &index_path_prefix,
                                  uint32_t num_clusters) {
  std::vector<std::string> slice_files_prefix;
  std::vector<std::string> cluster_base_files;
  for (auto i = 0; i < num_clusters; i++) {
    std::string cluster_base_file =
        index_path_prefix + "_cluster" + std::to_string(i) + ".bin";
    if (!file_exists(cluster_base_file)) {
      throw std::runtime_error("base file doesn't exist " + cluster_base_file);
    }
    cluster_base_files.push_back(cluster_base_file);

    std::stringstream slice;
    slice << index_path_prefix << "_cluster" << i << "_SAMPLE_RATE_"
    << MEM_INDEX_SAMPLING_RATE;
    slice_files_prefix.push_back(slice.str());
  }
  for (auto i = 0; i < num_clusters; i++) {
    std::string slice_file = slice_files_prefix[i] + "_data.bin";
    if (!file_exists(slice_file)) {
      gen_random_slice<T>(cluster_base_files[i], slice_files_prefix[i],
                          MEM_INDEX_SAMPLING_RATE);
    }
  }
  // for (auto &slice_file_prefix : slice_files_prefix) {
    // if (!file_exists(slice_file_prefix + "_data.bin")) {
      // gen_random_slice<T>(base_file, slice_file_prefix, MEM_INDEX_SAMPLING_RATE);
    // }
    // }
}


template<typename T>
int build_in_memory_index(const std::string &data_path, const std::string &tags_file, const unsigned R,
                          const unsigned L, const float alpha, const std::string &save_path, const unsigned num_threads,
                          bool dynamic_index, bool single_file_index, pipeann::Metric distMetric) {
  pipeann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("C", 750);  // maximum candidate set size during pruning procedure
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", 0);
  paras.Set<unsigned>("num_threads", num_threads);

  uint64_t data_num, data_dim;
  pipeann::get_bin_metadata(data_path, data_num, data_dim);
  std::cout << "Building in-memory index with parameters: data_file: " << data_path << "tags file: " << tags_file
            << " R: " << R << " L: " << L << " alpha: " << alpha << " index_path: " << save_path
            << " #threads: " << num_threads
            << ", using distance metric: " << (distMetric == pipeann::Metric::COSINE ? "cosine " : "l2 ");

  typedef uint32_t TagT;

  pipeann::Index<T, TagT> index(distMetric, data_dim, data_num, dynamic_index, single_file_index,
                                true);  // enable_tags forced to true!
  if (dynamic_index) {
    std::vector<TagT> tags(data_num);
    std::iota(tags.begin(), tags.end(), 0);

    auto s = std::chrono::high_resolution_clock::now();
    index.build(data_path.c_str(), data_num, paras, tags);
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    std::cout << "Indexing time: " << diff.count() << "\n";
  } else {
    std::ifstream reader;
    std::cout << "Opening bin file " << tags_file << "... " << std::endl;
    reader.open(tags_file, std::ios::binary);
    reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
    uint32_t tags_size = data_num * data_dim;
    std::vector<TagT> tags(data_num);
    reader.read((char *) tags.data(), tags_size * sizeof(uint32_t));
    reader.close();
    std::cout << "First tag is " << tags[0] << std::endl;

    auto s = std::chrono::high_resolution_clock::now();
    index.build(data_path.c_str(), data_num, paras, tags);
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    std::cout << "Indexing time: " << diff.count() << "\n";
  }
  index.save(save_path.c_str());

  return 0;
}



/**

build/tests/build_memory_index uint8 ${INDEX_PREFIX}_SAMPLE_RATE_0.01_data.bin
${INDEX_PREFIX}_SAMPLE_RATE_0.01_ids.bin ${INDEX_PREFIX}_mem.index 0 0 32 64 1.2
24 l2

*/

template <typename T, typename TagT>
void create_cluster_in_mem_indices(const std::string &base_file,
                                   const std::string &index_path_prefix,
                                   uint32_t num_clusters,
                                   const char *indexBuildParameters,
                                   pipeann::Metric metric) {
  // need to create the sample data
  std::vector<std::string> slice_files_bin;
  std::vector<std::string> slice_files_tag;
  for (auto i = 0; i < num_clusters; i++) {
    std::stringstream slice_prefix;
    slice_prefix << index_path_prefix << "_cluster" << i << "_SAMPLE_RATE_"
    << MEM_INDEX_SAMPLING_RATE;

    std::string slice_bin = slice_prefix.str() + "_data.bin";
    std::string slice_tag = slice_prefix.str() + "_ids.bin";
    if (!file_exists(slice_bin)) {
      throw std::invalid_argument("slice bin doesn't exist " + slice_bin);
    }
    if (!file_exists(slice_tag)) {
      throw std::invalid_argument("slice tag doesn't exist " + slice_bin);
    }
    slice_files_bin.push_back(slice_bin);
    slice_files_tag.push_back(slice_tag);
  }
  uint32_t R, L, num_threads;
  float B, M;
  std::istringstream parser(indexBuildParameters);
  parser >> R >> L >> B >> M >> num_threads;
  LOG(INFO) << "FOR IN MEM INDEX, R, L ARE " << R << " " << L;
  LOG(INFO) << "NUM THREAD " << num_threads;
  for (auto i = 0; i < num_clusters; i++) {
    std::string mem_index_path =
      index_path_prefix + "_cluster" + std::to_string(i) + "_mem.index";
    if (!file_exists(mem_index_path)) {
      std::string slice_bin = slice_files_bin[i];
      std::string slice_tag = slice_files_tag[i];
      build_in_memory_index<T>(slice_bin, slice_tag, R, L, MEM_INDEX_ALPHA,
                               mem_index_path, num_threads, false, false,
                               metric);
    }
  }
}






template <typename T, typename TagT>
void dumb_way(const std::string &index_path_prefix,
              const std::string &graph_path) {
  pipeann::Index<T, TagT> index(pipeann::Metric::L2, 128, 10'000'000, false,
                                false, false);
  index.load_from_disk_index(index_path_prefix);
  index.save_graph(graph_path);
  index.save_data(graph_path + ".data");
}


// need to check that the file doesn't already exists?
template <typename T, typename TagT>
void write_graph_index_from_disk_index(const std::string &index_path_prefix,
                                     const std::string &graph_path) {
  
  // only load V and E.
  std::ifstream in(index_path_prefix + "_disk.index", std::ios::binary);
  uint32_t nr, nc;
  uint64_t disk_nnodes, disk_ndims, medoid_id_on_file, max_node_len,
      nnodes_per_sector;

  in.read((char *)&nr, sizeof(uint32_t));
  in.read((char *)&nc, sizeof(uint32_t));

  in.read((char *)&disk_nnodes, sizeof(uint64_t));
  in.read((char *)&disk_ndims, sizeof(uint64_t));

  in.read((char *)&medoid_id_on_file, sizeof(uint64_t));
  in.read((char *)&max_node_len, sizeof(uint64_t));
  in.read((char *)&nnodes_per_sector, sizeof(uint64_t));

  LOG(INFO) << "Loading disk index from " << index_path_prefix << "_disk.index";
  LOG(INFO) << "Disk index has " << disk_nnodes << " nodes and " << disk_ndims
            << " dimensions.";
  LOG(INFO) << "Medoid id on file: " << medoid_id_on_file
            << " Max node len: " << max_node_len
            << " Nodes per sector: " << nnodes_per_sector;

  uint64_t data_dim = disk_ndims;


  std::ofstream mem_index_writer;
  mem_index_writer.open(graph_path, std::ios::binary);
  mem_index_writer.seekp(0, mem_index_writer.beg);
  uint64_t mem_index_size = 24;
  uint32_t max_degree = 0;
  uint32_t max_points = disk_nnodes;
  uint64_t num_frozen_points = 0;
  mem_index_writer.write((char *)&mem_index_size, sizeof(mem_index_size));
  mem_index_writer.write((char *)&max_degree, sizeof(max_degree));
  mem_index_writer.write((char *)&max_points, sizeof(max_points));
  mem_index_writer.write((char *)&num_frozen_points, sizeof(num_frozen_points));

  
  // std::vector<std::vector<uint32_t>> graph;
  // graph.resize(disk_nnodes);

  constexpr int kSectorsPerRead = 65536;
  constexpr int kSectorLen = 4096;
  char *buf;
  pipeann::alloc_aligned((void **)&buf, kSectorsPerRead * kSectorLen,
                         kSectorLen);
  uint64_t n_sectors =
      ROUND_UP(disk_nnodes, nnodes_per_sector) / nnodes_per_sector;
  in.seekg(4096, in.beg);
  for (uint64_t in_sector = 0; in_sector < n_sectors;
       in_sector += kSectorsPerRead) {
    uint64_t st_sector = in_sector,
             ed_sector = std::min(in_sector + kSectorsPerRead, n_sectors);
    uint64_t loc_st = st_sector * nnodes_per_sector,
             loc_ed = std::min(disk_nnodes, ed_sector * nnodes_per_sector);
    uint64_t n_sectors_to_read = ed_sector - st_sector;
    in.read(buf, n_sectors_to_read * kSectorLen);

    for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
      uint64_t id = loc;
      auto page_rbuf = buf + (loc / nnodes_per_sector - st_sector) * kSectorLen;
      auto node_rbuf = page_rbuf + (nnodes_per_sector == 0
                                        ? 0
                                        : ((uint64_t)loc % nnodes_per_sector) *
                                          max_node_len);
      pipeann::DiskNode<T> node(id, (T *)node_rbuf,
                                (unsigned *)(node_rbuf + data_dim * sizeof(T)));
      mem_index_writer.write((char *)&node.nnbrs, sizeof(node.nnbrs));
      mem_index_writer.write((char *)node.nbrs, node.nnbrs * sizeof(uint32_t));
      // LOG(INFO) << node.nnbrs << node.nbrs[0];
      // std::cout << node.nbrs[0] << std::endl;
      max_degree = std::max(max_degree, node.nnbrs);
      mem_index_size += sizeof(uint32_t) * (node.nnbrs + 1);
    }
  }
  mem_index_writer.seekp(0, mem_index_writer.beg);
  mem_index_writer.write((char *)&mem_index_size, sizeof(mem_index_size));
  mem_index_writer.write((char *)&max_degree, sizeof(max_degree));

  in.close();
  mem_index_writer.close();
}

std::vector<std::vector<int>> load_graph_file(const std::string &graph_path) {
  std::ifstream graph_reader(graph_path, std::ios::binary);
  graph_reader.seekg(0, graph_reader.beg);

  uint64_t file_size;
  uint32_t max_degree, num_points;
  uint64_t num_frozen_points;

  graph_reader.read((char*) &file_size, sizeof(file_size));
  graph_reader.read((char *)&max_degree, sizeof(max_degree));
  graph_reader.read((char *)&num_points, sizeof(num_points));
  graph_reader.read((char *)&num_frozen_points, sizeof(num_frozen_points));
  std::vector<std::vector<int>> graph;
  graph.reserve(num_points);

  for (int i = 0; i < num_points; i++) {
    uint32_t nnbrs;
    std::vector<uint32_t> nbrs;
    graph_reader.read((char *)&nnbrs, sizeof(nnbrs));

    nbrs.resize(nnbrs);
    graph_reader.read((char *)nbrs.data(), sizeof(uint32_t) * nnbrs);
    std::vector<int> nbrs_int;
    for (auto j = 0; j < nnbrs; j++) {
      nbrs_int.push_back(nbrs[j]);
    }
    graph.push_back(nbrs_int);
  }
  return graph;
}


void write_partitions_to_loc_files(const std::vector<std::vector<uint32_t>> &partitions,
                                 const std::string &output_index_path_prefix) {
  int partition_id  =0;
  for (const auto &partition : partitions) {
    std::string partition_loc_file = output_index_path_prefix + "_partition" +
                                     std::to_string(partition_id) +
                                   "_ids_uint32.bin";
    if (!file_exists(partition_loc_file)) {
      pipeann::save_bin<const uint32_t>(partition_loc_file, partition.data(),
                                        partition.size(), 1);
    } else {
      LOG(INFO) << "partition loc file already exists for partition"
      << partition_id;
    }
    partition_id++;
  }
}


std::vector<std::vector<uint32_t>>
parse_partition_loc_files(const std::string &output_index_path_prefix, int num_partitions) {
  std::vector<std::string> loc_files;
  for (auto i = 0; i < num_partitions; i++) {
    std::string partition_loc_file = output_index_path_prefix + "_partition" +
                                     std::to_string(i) + "_ids_uint32.bin";
    if (!file_exists(partition_loc_file)) {
      throw std::invalid_argument("loc file doesn't exist");
    }
    loc_files.push_back(partition_loc_file);
  }
  std::vector<std::vector<uint32_t>> partitions;
  partitions.resize(num_partitions);
  auto i = 0;
  for (auto loc_file : loc_files) {
    size_t num_pts, dim;
    pipeann::load_bin(loc_file, partitions[i], num_pts, dim);
    i++;
  }
  return partitions;
}


void write_partitions_to_txt_files(

				   const std::string &output_index_path_prefix, int num_partitions) {
  std::string partition_txt_file =
    output_index_path_prefix + "_partitions" + ".txt";
  if (file_exists(partition_txt_file)) {
    LOG(INFO) << "partition txt file already exists" << partition_txt_file;
    return;
  }
  auto partitions =
    parse_partition_loc_files(output_index_path_prefix, num_partitions);
  std::ofstream output(partition_txt_file);
  for (const auto &partition : partitions) {
    for (auto &i : partition) {
      output << i << ",";
    }
    output << std::endl;
  }  
}


void create_and_write_partitions_to_loc_files(
    const std::string &graph_path, const std::string &output_index_path_prefix,
					      int num_partitions) {
  if (!file_exists(graph_path)) {
    throw std::invalid_argument("graph doesn't exist " + graph_path);
  }

  bool should_partition = false;
  for (auto i = 0; i < num_partitions; i++) {
    std::string partition_loc_file = output_index_path_prefix + "_partition" +
                                     std::to_string(i) +
                                     "_ids_uint32.bin";
    if (!file_exists(partition_loc_file)) {
      should_partition = true;
      break;
    }
  }
  if (!should_partition)
    return;

  std::vector<std::vector<int>> graph = load_graph_file(graph_path);
  std::vector<std::vector<uint32_t>> partitions =
    get_partitions_from_adjgraph(graph, num_partitions);
  write_partitions_to_loc_files(partitions, output_index_path_prefix);
}


void create_graph_from_tag(const std::string &source_graph_path,
                           const std::string &tag_file,
                           const std::string &output_graph_path) {
  std::ifstream graph_reader(source_graph_path, std::ios::binary);
  graph_reader.seekg(0, graph_reader.beg);

  uint64_t file_size;
  uint32_t max_degree, num_points;
  uint64_t num_frozen_points;

  graph_reader.read((char*) &file_size, sizeof(file_size));
  graph_reader.read((char *)&max_degree, sizeof(max_degree));
  graph_reader.read((char *)&num_points, sizeof(num_points));
  graph_reader.read((char *)&num_frozen_points, sizeof(num_frozen_points));



  size_t num_pts, dim;
  std::vector<uint32_t> ids;
  pipeann::load_bin<uint32_t>(tag_file, ids, num_pts, dim);
  uint32_t ids_cnt = 0;


  std::ofstream graph_writer(output_graph_path, std::ios::binary);
  graph_writer.seekp(0, graph_writer.beg);

  uint64_t output_file_size = 24;
  uint32_t output_max_degree = 0, output_num_points = num_pts;
  uint64_t output_num_frozen_points = 0;
  graph_writer.write((char *)&output_file_size, sizeof(output_file_size));
  graph_writer.write((char *)&output_max_degree, sizeof(output_max_degree));
  graph_writer.write((char *)&output_num_points, sizeof(output_num_points));
  graph_writer.write((char *)&output_num_frozen_points,
                     sizeof(output_num_frozen_points));

  for (auto i = 0; i < num_points; i++) {
    if (ids_cnt == ids.size()) {
      break;
    }
    uint32_t nnbrs;
    graph_reader.read((char *)&nnbrs, sizeof(nnbrs));
    if ( i == ids[ids_cnt]) {
      uint32_t *nbrs = new uint32_t[nnbrs];
      graph_reader.read((char *)nbrs, sizeof(uint32_t) * nnbrs);


      graph_writer.write((char*) &nnbrs, sizeof(uint32_t));
      graph_writer.write((char*) nbrs, sizeof(uint32_t) * nnbrs);
      ids_cnt++;
      output_max_degree = std::max(output_max_degree, nnbrs);
      delete[] nbrs;
      output_file_size += sizeof(uint32_t) * (nnbrs + 1);
    } else {
      graph_reader.seekg(sizeof(uint32_t) * nnbrs, graph_reader.cur);
    }
  }
  graph_writer.seekp(0, graph_writer.beg);
  graph_writer.write((char *)&output_file_size, sizeof(output_file_size));
  graph_writer.write((char *)&output_max_degree, sizeof(output_max_degree));
}


void create_graphs_from_tags(const std::string &source_graph_path,
                             const std::string &output_index_path_prefix,
                             int num_partitions) {
  std::vector<std::string> tag_files;
  for (auto i = 0; i < num_partitions; i++) {
    std::string tag_file = output_index_path_prefix + "_partition" +
                           std::to_string(i) + "_ids_uint32.bin";
    if (!file_exists(tag_file)) {
      throw std::runtime_error(
			       "Tag file doesn't exist, can't create graph file " + tag_file);
    }
    tag_files.push_back(tag_file);
  }
  for (auto i = 0; i < num_partitions; i++) {
    std::string graph_file =
      output_index_path_prefix + "_partition" + std::to_string(i) + "_graph";
    if (!file_exists(graph_file)) {
      create_graph_from_tag(source_graph_path, tag_files[i], graph_file);
    } else {
      LOG(INFO)<< "graph file already exists " << graph_file;
    }
  }
}
template<typename T, typename TagT>
void create_base_files_from_tags(const std::string &base_file,
                                 const std::string &output_index_path_prefix,
                                 int num_partitions) {
  std::vector<std::string> loc_files;
  for (auto i = 0; i < num_partitions; i++) {
    std::string loc_file = output_index_path_prefix + "_partition" +
                           std::to_string(i) + "_ids_uint32.bin";
    if (!file_exists(loc_file)) {
      throw std::runtime_error("loc file doesn't exist : " + loc_file);
    }
    loc_files.push_back(loc_file);
  }

  for (auto i = 0; i < num_partitions; i++) {
    std::string partition_base_file =
      output_index_path_prefix + "_partition" + std::to_string(i) + ".bin";
    if (!file_exists(partition_base_file)) {
      create_base_from_tag<T, TagT>(base_file, loc_files[i], partition_base_file);
    } else {
      LOG(INFO) << "base file already exists " << partition_base_file;
    }
  }
}


template <typename T, typename TagT>
void create_disk_indices(const std::string &output_index_path_prefix,
                         int num_partitions) {
  std::vector<std::string> base_files;
  std::vector<std::string> graph_files;
  for (int i = 0; i < num_partitions; i++) {
    std::string partition_base_file =
      output_index_path_prefix + "_partition" + std::to_string(i) + ".bin";
    std::string partition_graph_file =
      output_index_path_prefix + "_partition" + std::to_string(i) + "_graph";
    if (!file_exists(partition_base_file)) {
      throw std::runtime_error("base file doesn't exist " +
                               partition_base_file);
    }
    if (!file_exists(partition_graph_file)) {
      throw std::runtime_error("graph file doesn't exist " +
                               partition_graph_file);
    }
    base_files.push_back(partition_base_file);
    graph_files.push_back(partition_graph_file);
  }
  for (int i = 0; i < num_partitions; i++) {
    std::string disk_index = output_index_path_prefix + "_partition" +
                             std::to_string(i) + "_disk.index";
    if (!file_exists(disk_index)) {
      pipeann::create_disk_layout<T>(graph_files[i], base_files[i], "", "", "",
                                     false, disk_index);
    } else {
      LOG(INFO) << "Disk index already exists " << disk_index;
    }

  }
}

void create_partition_assignment_file(const std::string &output_index_path_prefix, int num_partitions) {
  std::vector<std::string> loc_files;
  for (auto i = 0; i < num_partitions; i++) {
    std::string loc_file = output_index_path_prefix + "_partition" +
                           std::to_string(i) + "_ids_uint32.bin";
    if (!file_exists(loc_file)) {
      throw std::runtime_error("loc file doesn't exist : " + loc_file);
    }
    loc_files.push_back(loc_file);
  }

  std::string partition_assignment_file =
    output_index_path_prefix + "_partition_assignment.bin";
  if (file_exists(partition_assignment_file)) {
    LOG(INFO) << "partition_assignment_file already exists "
    << partition_assignment_file;
    return;
  }
  

  std::vector<std::vector<uint32_t>> locs;
  locs.resize(num_partitions);
  for (auto i = 0; i < num_partitions; i++) {
    std::string loc_file = loc_files[i];
    size_t num_pts, dim;
    pipeann::load_bin<uint32_t>(loc_file, locs[i], num_pts, dim);
  }

  size_t num_points = 0;
  for (const auto &loc : locs) {
    num_points += loc.size();
  }

  std::vector<uint8_t> partition_map;
  partition_map.resize(num_points);
  for (auto i = 0; i < num_partitions; i++) {
    uint8_t partition_i = static_cast<uint8_t>(i);
    for (const uint32_t j : locs[i]) {
      partition_map[j] = partition_i;
    }
  }


  pipeann::save_bin<uint8_t>(partition_assignment_file, partition_map.data(),
                             num_points, 1);
}


template void
create_random_cluster_tag_files<float>(const std::string &base_file,
                                       const std::string &index_path_prefix,
                                       uint32_t num_clusters);

template void
create_random_cluster_tag_files<uint8_t>(const std::string &base_file,
                                         const std::string &index_path_prefix,
                                         uint32_t num_clusters);

template void
create_random_cluster_tag_files<int8_t>(const std::string &base_file,
                                        const std::string &index_path_prefix,
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

template void create_random_cluster_disk_indices<uint8_t>(
    const std::string &index_path_prefix, uint32_t num_clusters,
    const char *indexBuildParameters, pipeann::Metric _compareMetric,
    bool single_file_index);


template void
write_graph_index_from_disk_index<float>(const std::string &index_path_prefix,
                                         const std::string &mem_index_path);
template void
write_graph_index_from_disk_index<uint8_t>(const std::string &index_path_prefix,
                                         const std::string &mem_index_path);

template void
write_graph_index_from_disk_index<int8_t>(const std::string &index_path_prefix,
                                          const std::string &mem_index_path);


template void dumb_way<float>(const std::string &index_path_prefix,
                              const std::string &graph_path);

template void dumb_way<uint8_t>(const std::string &index_path_prefix,
                                const std::string &graph_path);

template void dumb_way<int8_t>(const std::string &index_path_prefix,
                              const std::string &graph_path);


template void create_base_files_from_tags<float>(const std::string &base_file,
                                 const std::string &output_index_path_prefix,
                                 int num_partitions);


template void create_base_files_from_tags<uint8_t>(const std::string &base_file,
                                 const std::string &output_index_path_prefix,
                                 int num_partitions);


template void create_base_files_from_tags<int8_t>(const std::string &base_file,
                                 const std::string &output_index_path_prefix,
                                 int num_partitions);


template void
create_disk_indices<float>(const std::string &output_index_path_prefix,
                           int num_partitions);

template void
create_disk_indices<uint8_t>(const std::string &output_index_path_prefix,
                             int num_partitions);

template void
create_disk_indices<int8_t>(const std::string &output_index_path_prefix,
                            int num_partitions);


template
void create_cluster_random_slices<float>(const std::string &base_file,
                                         const std::string &index_path_prefix,
                                         uint32_t num_clusters);

template void
create_cluster_random_slices<uint8_t>(const std::string &base_file,
                                      const std::string &index_path_prefix,
                                      uint32_t num_clusters);

template void
create_cluster_random_slices<int8_t>(const std::string &base_file,
                                     const std::string &index_path_prefix,
                                     uint32_t num_clusters);


template
void create_cluster_in_mem_indices<float>(const std::string &base_file,
                                   const std::string &index_path_prefix,
                                   uint32_t num_clusters,
                                   const char *indexBuildParameters,
                                   pipeann::Metric metric);

template
void create_cluster_in_mem_indices<uint8_t>(const std::string &base_file,
                                   const std::string &index_path_prefix,
                                   uint32_t num_clusters,
                                   const char *indexBuildParameters,
                                   pipeann::Metric metric);

template
void create_cluster_in_mem_indices<int8_t>(const std::string &base_file,
                                   const std::string &index_path_prefix,
                                   uint32_t num_clusters,
                                   const char *indexBuildParameters,
                                   pipeann::Metric metric);



