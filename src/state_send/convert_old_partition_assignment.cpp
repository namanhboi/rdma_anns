#include <stdexcept>
#include <string>
#include "disk_utils.h"
#include "utils.h"


void convert(const std::string &partition_assignment_file) {
  if (!file_exists(partition_assignment_file)) {
    throw std::invalid_argument(partition_assignment_file + " doesn't exist");
  }
  std::vector<uint8_t> partition_assignment;
  size_t num_pts, dim;
  pipeann::load_bin<uint8_t>(partition_assignment_file, partition_assignment,
                             num_pts, dim);
  LOG(INFO) << "number of points " << num_pts;
  LOG(INFO) << "number of dim " << dim;
  LOG(INFO) << "partittion assignment size " << partition_assignment.size();
  uint8_t max_partition_id = 0;
  for (const auto &partition_id : partition_assignment) {
    max_partition_id = std::max(partition_id, max_partition_id);
  }
  uint8_t num_partitions = max_partition_id + 1;
  
  std::ofstream partition_out(partition_assignment_file, std::ios::binary);
  
  partition_out.write(reinterpret_cast<char *>(&num_pts), sizeof(num_pts));
  partition_out.write(reinterpret_cast<char *>(&num_partitions),
                      sizeof(num_partitions));
  for (const auto &home_partitions : partition_assignment) {
    uint8_t num_home_partition_u8 = static_cast<uint8_t>(1);
    partition_out.write(reinterpret_cast<char *>(&num_home_partition_u8),
                        sizeof(num_home_partition_u8));
    partition_out.write(reinterpret_cast<const char *>(&home_partitions),
                        sizeof(uint8_t));
  }    
}


int main(int argc, char** argv) {
  std::string partition_assignment_file = argv[1];
  convert(partition_assignment_file);
  return 0;
}

