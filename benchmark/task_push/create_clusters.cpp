#include "data_loading_utils.hpp"
#include <boost/program_options.hpp>
#include <stdexcept>


namespace po = boost::program_options;

int main(int argc, char **argv) {
  po::options_description desc(
      "Given a big disk index file + the cluster assignment bin file + the "
      "data_file, create the indices for each file.");
  std::string cluster_assignment_file, clusters_folder, data_file, index_path_prefix, data_type;

  desc.add_options()("help,h", "show help message")(
      "data_type,T", po::value<std::string>(&data_type)->required(),
      "data type of index and data_file")(
      "index_path_prefix,P",
      po::value<std::string>(&index_path_prefix)->required(),
      "Path to disk index file.")(
      "data_file,D", po::value<std::string>(&data_file)->required(),
      "Path to in data file, like sift learn")(
      "cluster_assignment_file,F",
      po::value<std::string>(&cluster_assignment_file)->required(),
      "path to cluster assignment bin file")(
      "clusters_folder,O", po::value<std::string>(&clusters_folder)->required(),
					     "path to output all the indices for each cluster");
  po::variables_map vm;

  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  po::notify(vm);

  Clusters clusters =
    parse_cluster_assignment_bin_file(cluster_assignment_file);
  if (data_type == "uint8") {
    create_cluster_index_files<uint8_t>(clusters, data_file, index_path_prefix,
                              clusters_folder);
  } else if (data_type == "int8") {
    create_cluster_index_files<int8_t>(clusters, data_file, index_path_prefix,
                                       clusters_folder);
  } else if (data_type == "float") {
    create_cluster_index_files<float>(clusters, data_file, index_path_prefix,
                                      clusters_folder);
  } else {
    throw std::invalid_argument("data type not uint8, int8 or float " +
                                data_type);
  }
}
