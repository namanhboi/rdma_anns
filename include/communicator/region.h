#pragma once
#include "../state_send/types.h"
#include "../state_send/utils.h"

struct Region {
  static constexpr size_t MAX_BYTES_REGION = 80000 * 4;
  static constexpr size_t MAX_PRE_ALLOC_ELEMENTS_RDMA = 12000;
  static constexpr size_t MAX_PRE_ALLOC_ELEMENTS = 100000;

  char *addr; // address to whatever is sent must be allocated with new []
  uint32_t length;

  // for rdma, specifically this : https://arxiv.org/pdf/2212.09134
  uint64_t context;
  uint32_t lkey;

  // passed into delete addr, used to detemine whether to allow zmq to delete
  // data or not
  // bool self_manage_data = false;

  void *prealloc_queue;

  // by default, we don't manage our memeory
  Region() : prealloc_queue(nullptr) {}
  Region(uint64_t context, char *addr, uint32_t length, uint32_t lkey)
      : context(context), addr(addr), length(length), lkey(lkey) {}

  static void reset(Region *r) { r->length = 0; }

  // hint used to pass in the address to prealloc queue
static void assign_addr(Region *r, char *prealloacted_addr, void *hint, uint32_t lkey) {
    r->addr = prealloacted_addr;
    r->prealloc_queue = hint;
    r->lkey = lkey;
}


/* // hint used to pass in the address to prealloc queue, added the lkey for mr */
/* static void assign_addr_mr(Region *r, char *prealloacted_addr, void *hint, uint32_t lkey) { */
/*   r->addr = prealloacted_addr; */
/*   r->prealloc_queue = hint; */
/*   r->lkey = lkey; */
/* } */


  /**
   * used for zmq zmq_msg_init_data, which will use this function to free addr,
   so no need to manually free it if use zmq. Hint here represents the pointer
   to the Region r you're freeing.
   If the region is self managed, then should pass in pointer to it. Else pass
   in nullptr, which will trigger a delete[] call to free addr
   */
  static void delete_addr(void *data, void *hint) {
    // reinterpretting the hint as a bool representing self_managed_data
    if (hint == nullptr) {
      delete[] reinterpret_cast<char *>(data);
    } else {
      Region *r = reinterpret_cast<Region *>(hint);
      if (unlikely(r->prealloc_queue == nullptr)) {
        throw std::runtime_error(
            "prealloc queue is nullptr even tho ptr to Region is not nullptr");
      }
      PreallocatedQueue<Region> *q =
          reinterpret_cast<PreallocatedQueue<Region> *>(r->prealloc_queue);
      q->free(r);
    }
  }
};
