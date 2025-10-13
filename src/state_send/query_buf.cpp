#include "query_buf.h"
namespace pipeann {
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
    this->coords = coords;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }
  // structs for DiskNode
  template struct DiskNode<float>;
  template struct DiskNode<uint8_t>;
  template struct DiskNode<int8_t>;
}
