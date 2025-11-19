#include "disk_utils.h"


// Source - https://stackoverflow.com/a
// Posted by Joseph, modified by community. See post 'Timeline' for change history
// Retrieved 2025-11-19, License - CC BY-SA 3.0
inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


int main(int argc, char **argv) {
  std::string source_graph_file(argv[1]);
  std::string partition_file(argv[2]);
  std::string output_partition_graph_file(argv[3]);

  // diskann layout
  if (ends_with(source_graph_file, "_graph")) {
    create_graph_from_tag(source_graph_file, partition_file,
                          output_partition_graph_file);
  } else {
    // parlayann layout
    size_t num_pts, dim;
    std::vector<uint32_t> ids;
    pipeann::load_bin<uint32_t>(partition_file, ids, num_pts, dim);
    if (dim != 1) {
      throw std::invalid_argument(
				  "dim for loc file should be 1, weird dim value " + std::to_string(dim));
    }
    LOG(INFO) << "num pts" << num_pts;
    write_graph_file_from_parlayann_graph_file(source_graph_file, ids,
                                               output_partition_graph_file);

  }
  return 0;
}
