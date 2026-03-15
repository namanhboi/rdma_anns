#include "disk_utils.h"
#include "parlay/primitives.h"
#include "utils.h"
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace po = boost::program_options;
namespace fs = std::filesystem;

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

template <typename T>
void load_bin_mmap(const std::string &filename, T *&data, size_t &num_pts,
                   size_t &dim) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    throw std::runtime_error("Could not open file for mmap");

  // 1. Read the header (Assuming 2 x 32-bit or 64-bit unsigned ints for n and
  // d) Adjust types (uint32_t vs size_t) based on your specific file format
  uint32_t header[2];
  if (read(fd, header, sizeof(uint32_t) * 2) != sizeof(uint32_t) * 2) {
    close(fd);
    throw std::runtime_error("Error reading file header");
  }
  num_pts = header[0];
  dim = header[1];

  // 2. Map the data
  size_t header_size = sizeof(uint32_t) * 2;
  size_t data_size = (size_t)num_pts * dim * sizeof(T);

  // We map the whole file but offset the pointer to the start of the data
  void *mapped_ptr =
      mmap(NULL, data_size + header_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped_ptr == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("mmap failed - check address space limits");
  }

  // data points to the first element after the header
  data = reinterpret_cast<T *>(static_cast<char *>(mapped_ptr) + header_size);

  // Close the file descriptor; the mapping remains active
  close(fd);
}

template <typename T>
void create_medoid_file(const std::string &base_file,
                        const std::vector<std::string> &ids_files,
                        const std::string &medoid_file,
                        const std::string &index_path_prefix) {

  size_t num_pts, dim;
  T *full_data = nullptr;

  // Use the mmap version for the 1B point file
  std::cout << "Memory mapping 1B point file: " << base_file << "..."
            << std::endl;
  load_bin_mmap<T>(base_file, full_data, num_pts, dim);

  std::vector<T> all_medoids_flat;
  std::vector<uint32_t> medoid_global_ids;

  for (size_t i = 0; i < ids_files.size(); i++) {
    // IDs files are usually small enough to load with your standard load_bin
    uint32_t *part_ids_raw;
    size_t n_part, d_part;
    pipeann::load_bin(ids_files[i], part_ids_raw, n_part, d_part);

    parlay::sequence<uint32_t> partition_indices(part_ids_raw,
                                                 part_ids_raw + n_part);

    T *medoid_vec = nullptr;
    uint32_t best_global_id = 0;

    calculate_medoid<T>(full_data, dim, partition_indices, medoid_vec,
                        best_global_id);

    for (size_t j = 0; j < dim; j++)
      all_medoids_flat.push_back(medoid_vec[j]);
    medoid_global_ids.push_back(best_global_id);

    delete[] medoid_vec;
    // If your load_bin allocates with new[], clean up here:
    // delete[] part_ids_raw;

    std::cout << "Partition " << i << " done. Medoid ID: " << best_global_id
              << std::endl;
  }

  // Save final results
  std::string out_fn = medoid_file;
  pipeann::save_bin<T>(out_fn, all_medoids_flat.data(), ids_files.size(), dim);


  parlay::sequence<uint32_t> medoid_local_ids = parlay::tabulate(ids_files.size(), [&](size_t i) {
    uint32_t medoid_global_id = medoid_global_ids[i];
    std::vector<uint32_t> partition_ids;
    size_t num_partition_ids, dim;
    pipeann::load_bin(ids_files[i], partition_ids, num_partition_ids, dim);
    for (uint32_t j = 0; j < num_partition_ids; j++) {
      if (partition_ids[j] == medoid_global_id) {
	return j;
      }
    }
    return std::numeric_limits<uint32_t>::max();
  });
  
  for (size_t i = 0; i < medoid_global_ids.size(); i++) {
    std::string disk_index_path =
      index_path_prefix + "_cluster" + std::to_string(i) + "_disk.index";
    std::string medoid_path = disk_index_path + "_medoids.bin";
    pipeann::save_bin<uint32_t>(medoid_path, &medoid_local_ids[i], 1, 1);
  }

  // Unmap full_data before finishing (optional but clean)
  size_t header_size = sizeof(uint32_t) * 2;
  size_t data_size = (size_t)num_pts * dim * sizeof(T);
  munmap(static_cast<char *>(static_cast<void *>(full_data)) - header_size,
         data_size + header_size);
}

int main(int argc, char **argv) {
  po::options_description desc(
      "create the medoids file containing medoids from "
      "each partition from the base_file and ids "
      "file for each partition");

  std::string base_file, medoid_file, data_type, index_path_prefix;
  std::vector<std::string> ids_files;
  desc.add_options()("help,h", "show help")(
      "base_file", po::value<std::string>(&base_file)->required(),
      "base file containing the whole dataset")(
      "data_type", po::value<std::string>(&data_type)->required(),
      "data type of base file")(
      "ids_files",
      po::value<std::vector<std::string>>(&ids_files)->multitoken()->required(),
      "ids files for each partition")(
      "medoid_file", po::value<std::string>(&medoid_file)->required(),
      "output medoid file")(
      "index_path_prefix",
      po::value<std::string>(&index_path_prefix)->required(),
      "index path prefix, should have no 'partition' in path', for 100M its "
      "/pipeann_100M");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);
  if (!file_exists(base_file)) {
    throw std::invalid_argument(base_file + " doesn't exist");
  }
  if (data_type != "uint8" && data_type != "int8" && data_type != "float") {
    throw std::invalid_argument("data type has weird value");
  }

  size_t base_file_num_pts, base_file_dim;
  pipeann::get_bin_metadata(base_file, base_file_num_pts, base_file_dim);

  size_t num_ids = 0;
  for (const std::string &id_file : ids_files) {
    if (!file_exists(id_file)) {
      throw std::invalid_argument(id_file + " doesn't exist");
    }
    size_t num_pts, dim;
    pipeann::get_bin_metadata(id_file, num_pts, dim);
    if (dim != 1) {
      throw std::invalid_argument(id_file + " has weird dim value " +
                                  std::to_string(dim));
    }
    std::cout << "num_pts from " << id_file << " " << num_pts << std::endl;
    num_ids += num_pts;
  }
  if (num_ids != base_file_num_pts) {
    throw std::invalid_argument("number of ids " + std::to_string(num_ids) +
                                " dif from num pts from base file " +
                                std::to_string(base_file_num_pts));
  }
  if (data_type == "uint8") {
    create_medoid_file<uint8_t>(base_file, ids_files, medoid_file,
                                index_path_prefix);
  } else if (data_type == "int8") {
    create_medoid_file<int8_t>(base_file, ids_files, medoid_file,
                               index_path_prefix);
  } else if (data_type == "float") {
    create_medoid_file<float>(base_file, ids_files, medoid_file,
                              index_path_prefix);
  }

  return 0;
}
