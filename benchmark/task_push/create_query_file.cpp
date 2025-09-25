#include <fstream>
#include <random>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/program_options/value_semantic.hpp>

namespace po = boost::program_options;

template <typename data_type>
void write_query_file(const std::string &data_file,
                      const std::string &query_file, int num_points) {
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 gen(rd()); // seed the generator

  std::ifstream data_file_in(data_file, std::ios::binary);
  int npts_i32, dim_i32;
  data_file_in.seekg(0, data_file_in.beg);
  data_file_in.read((char *)&npts_i32, sizeof(int));
  data_file_in.read((char *)&dim_i32, sizeof(int));
  std::uniform_int_distribution<uint32_t> uint32_t_gen(
						       0, npts_i32 - 1);
  std::set<uint32_t> random_indices;
  while (random_indices.size() < num_points) {
    uint32_t node_id = uint32_t_gen(gen);
    if (random_indices.count(node_id) == 0) {
      random_indices.emplace(node_id);
    }
  }
  size_t emb_start_pos = sizeof(int) * 2;
  size_t emb_size = sizeof(data_type) * dim_i32;

  std::ofstream query_file_out(query_file, std::ios::binary);
  query_file_out.write((char *)&num_points, sizeof(int));
  query_file_out.write((char *)&dim_i32, sizeof(int));
  for (uint32_t index : random_indices) {
    data_type *emb = new data_type[dim_i32];
    data_file_in.seekg(emb_start_pos + emb_size * index);
    data_file_in.read((char *)emb, emb_size);
    query_file_out.write((char *)emb, emb_size);
    delete[] emb;
  }

}


int main(int argc, char **argv) {
  po::options_description desc("Create a query file from a data file");
  std::string data_file;
  std::string data_type;
  std::string query_file;
  int num_points;
  desc.add_options()("help,h", "show help message")(
      "data_type,T", po::value<std::string>(&data_type)->required(),
      "data type of index and data_file")(
      "data_file,D", po::value<std::string>(&data_file)->required(),
      "Path to in data file, like sift learn")(
      "num_points,N", po::value<int>(&num_points)->required(),
      "Number of points you want to take from the data file to create the "
      "query file")("query_file,Q",
                    po::value<std::string>(&query_file)->required(),
                    "Path to in data file, like sift learn");
  

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  po::notify(vm);

  if (data_type == "uint8") {
    write_query_file<uint8_t>(data_file ,query_file, num_points);
  } else if (data_type == "int8") {
    write_query_file<int8_t>(data_file ,query_file, num_points);
  } else if (data_type == "float") {
    write_query_file<float>(data_file ,query_file, num_points);
  }
}
