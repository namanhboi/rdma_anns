#pragma once

#include "VerbsEP.hpp"
#include "ring.hpp"
#include <stdio.h>
#include <stdexcept>
#include "../region.h"

#define MAGIC_BYTE_T uint8_t
#define LEN_BYTE_T uint32_t

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define PREFIX_SIZE (64)
#define SUFFIX_SIZE (64)


enum class msg_opcode_t:uint8_t {
  USER_DATA, FREED_BYTES
};

// the first sge is for the 0 + opcode, the second is the user provided payload,
// the third is data len + 1
class CircularConnectionReverse {
  VerbsEP *const ep;
  RemoteBuffer *const remote_buffer;

  uint64_t last_wrid = 0;
  uint64_t next_id_ = 1;

  struct ibv_send_wr wr;
  struct ibv_sge sges[3];

  const bool with_sges;
  const uint64_t local_mem;
  const uint32_t local_mem_lkey; // ADDED: Must save this to reuse in SendAsync!
public:
  CircularConnectionReverse(VerbsEP *ep, RemoteBuffer *remote_buffer,
                            uint64_t local_mem, uint32_t local_mem_lkey)
  : ep(ep), remote_buffer(remote_buffer),
    with_sges(ep->GetMaxSendSge() >= 3), local_mem(local_mem),
    local_mem_lkey(local_mem_lkey) {
    if (!with_sges) {
      throw std::runtime_error(
                               "Expected SGE to be used. Hardware does not support 3 SGEs.");
    }
    printf(">>>>>>>>>>>>>>>>>> I use sges implementation\n");

    // Initialize the static parts of the Work Request
    wr.sg_list = &sges[0];
    wr.num_sge = 3;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = 0; // Updated dynamically per message
    wr.wr.rdma.rkey = remote_buffer->GetKey();
    wr.next = NULL;

    // Initialize the SGE keys (addresses and lengths are set dynamically in
    // SendAsync)
    sges[0].lkey = local_mem_lkey;
    sges[1].lkey = 0; // User payload key set dynamically
    sges[2].lkey = local_mem_lkey;

    // Prime the remote buffer for the initial state of the Reverse Ring Buffer.
    // Our prefix is now 2 bytes: Magic 0 + Opcode.
    remote_buffer->GetWriteAddr(sizeof(MAGIC_BYTE_T) + sizeof(msg_opcode_t));
  }

  ~CircularConnectionReverse() { delete remote_buffer; };

  void PrintInfo() {
    printf("-----------------------------------------\n");
    printf("Name: Circular Reverse Sender\n");
    printf("Role: Sender\n");
    printf("Type: P2P.\n");
    printf("Used OPs: RDMA Write\n");
    printf("Way of informing the receiver: MagicByte\n");
    printf("Prefix: Yes - zero magic byte \n");
    printf("Suffix: Yes - and length magic byte.\n");
    if (with_sges)
      printf("Sges: Yes - 3 to send prefix and suffix\n");
    printf("Downsides: no\n");
    printf("-----------------------------------------\n");
  }

  uint32_t ReqSize(Region *region) {
    return region->length + sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T);
  }

  // ADDED: uint8_t opcode parameter
  uint64_t SendAsync(msg_opcode_t opcode, Region *region) {
    struct ibv_send_wr *bad_wr;

    // 1. Sliding Window Slot Calculation (Fixes the race condition)
    // 128 outstanding messages, max 16 bytes of metadata per message
    uint32_t slot_index = next_id_ % 128;
    uint64_t slot_base_addr = local_mem + (slot_index * 16);
    uint64_t suffix_addr =
      slot_base_addr + 8; // Safe offset within the 16-byte slot

    // 2. Calculate Total Wire Footprint
    // Prefix (Magic 0 + Opcode) + Payload + Suffix (Length + Magic 1)
    uint32_t prefix_len = sizeof(MAGIC_BYTE_T) + sizeof(msg_opcode_t);
    uint32_t suffix_len = sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T);
    uint32_t wire_length = prefix_len + region->length + suffix_len;

    // THE OVERLAP HACK: Only allocate (L + 6) from the Ring Buffer!
    uint32_t ring_allocation = wire_length - sizeof(MAGIC_BYTE_T);
    uint64_t rem_addr = remote_buffer->GetWriteAddr(ring_allocation);
    // std::cout << "Sender targeting remote address: " << std::hex << rem_addr
    // << std::dec << std::endl;

    if (rem_addr == 0) {
      // Return -1 to indicate memory is full right now
      return (uint64_t)-1;
    }
    static bool printed_once = false;
    if (!printed_once) {
      std::cout << "Sender targeting remote address: 0x"
      << std::hex << rem_addr << std::dec
      << std::endl;
      printed_once = true;
    }
    // 3. Write Metadata into local pinned RAM
    // --- PREFIX ---
    *(volatile MAGIC_BYTE_T *)(void *)(slot_base_addr) =
      (MAGIC_BYTE_T)0; // Magic 0 (Reset Bell)
    *(volatile msg_opcode_t *)(void *)(slot_base_addr + sizeof(MAGIC_BYTE_T)) =
      opcode; // Application Opcode

    // --- SUFFIX ---
    *(volatile LEN_BYTE_T *)(void *)(suffix_addr) =
      (LEN_BYTE_T)region->length; // Payload Length
    *(volatile MAGIC_BYTE_T *)(void *)(suffix_addr + sizeof(LEN_BYTE_T)) =
      (MAGIC_BYTE_T)1; // Magic 1 (Trigger Bell)

    // 4. Populate SGEs
    sges[0].addr = slot_base_addr;
    sges[0].length = prefix_len;

    sges[1].addr = (uint64_t)(void *)region->addr;
    sges[1].length = region->length;
    sges[1].lkey = region->lkey;

    sges[2].addr = suffix_addr;
    sges[2].length = suffix_len;

    // 5. Hardware Handoff
    wr.wr_id = next_id_;
    wr.wr.rdma.remote_addr = rem_addr + 1;

    if (ibv_post_send(ep->qp, &wr, &bad_wr)) {
      printf("Failed to send %d\n", errno);
      exit(1);
    }

    return next_id_++;
  }

  // For User Data
  inline uint64_t SendDataAsync(Region* user_region) {
    return SendAsync(msg_opcode_t::USER_DATA, user_region);
  }

uint64_t SendAckAsync(uint32_t freed_bytes) {
    struct ibv_send_wr *bad_wr;

    uint32_t slot_index = next_id_ % 128;
    uint64_t slot_base_addr = local_mem + (slot_index * 16);

    // =================================================================
    // 1. PACK ALL 11 BYTES CONTIGUOUSLY
    // =================================================================
    // Prefix (Magic 0 + Opcode) -> 2 bytes
    *(volatile MAGIC_BYTE_T*)(void*)(slot_base_addr) = 0;
    *(volatile msg_opcode_t*)(void*)(slot_base_addr + 1) = msg_opcode_t::FREED_BYTES;

    // Payload (Freed Bytes) -> 4 bytes
    *(volatile uint32_t*)(void*)(slot_base_addr + 2) = freed_bytes;

    // Suffix (Length 4 + Magic 1) -> 5 bytes
    *(volatile LEN_BYTE_T*)(void*)(slot_base_addr + 6) = 4;
    *(volatile MAGIC_BYTE_T*)(void*)(slot_base_addr + 10) = 1;

    // =================================================================
    // 2. SETUP A SINGLE SGE
    // =================================================================
    sges[0].addr   = slot_base_addr;
    sges[0].length = 11; // 2 + 4 + 5 = Exactly 11 bytes footprint
    sges[0].lkey   = local_mem_lkey; // Safely locked in!

    wr.num_sge = 1; // THE FIX: Tell the NIC exactly how many SGEs to read

    // =================================================================
    // 3. HARDWARE HANDOFF
    // =================================================================
    uint32_t ring_allocation = 11 - sizeof(MAGIC_BYTE_T);
    uint64_t rem_addr = remote_buffer->GetWriteAddr(ring_allocation);

    if (rem_addr == 0) {
        // Return -1 so send_loop can gracefully try again next time
        return (uint64_t)-1;
    }

    wr.wr.rdma.remote_addr = rem_addr + 1; // Keep your brilliant 1-byte shift
    wr.wr_id = next_id_;

    // Capture the REAL error code just in case!
    int ret = ibv_post_send(ep->qp, &wr, &bad_wr);
    if(ret) {
      printf("CRITICAL: Failed to send ACK. Real error code: %d\n", ret);
      exit(1);
    }

    return next_id_++;
  }
  bool AckSentBytes(uint32_t bytes) { return remote_buffer->FreeBytes(bytes); }

  void WaitSend(uint64_t id) {
    while (last_wrid < id) {
      TestSend(id);
    }
  }

  // check if nic is done sending id
  bool TestSend(uint64_t id) {
    if (last_wrid >= id)
      return true;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq, 16, wcs);
    for (int i = 0; i < ret; i++) {
      if (wcs[i].status != IBV_WC_SUCCESS) {
        printf("Failed request %d \n", wcs[i].status);
        exit(1);
      }
      last_wrid = wcs[i].wr_id;
    }
    if (last_wrid >= id)
      return true;
    return false;
  }
};

class CircularReverseReceiver  {
  Buffer *const local_buffer;

public:
  CircularReverseReceiver(Buffer *local_buffer) : local_buffer(local_buffer) {
    *(volatile MAGIC_BYTE_T *)local_buffer->Read(sizeof(MAGIC_BYTE_T)) =
      (MAGIC_BYTE_T)0;
  };
  ~CircularReverseReceiver() {
    delete local_buffer;
  };

  void PrintInfo() {
    printf("-----------------------------------------\n");
    printf("Name: Circular Reverse Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: P2P.\n");
    printf("Used OPs: No - memory checking\n");
    printf("Passive: No\n");
    printf("-----------------------------------------\n");
  }
  /**
     this is non blocking, all good to go. Receive multiple messages at once;
     that's why there is a loop.
     The receiver polls the magic byte at the "start" of the message, if it is
     not 0 then there is a message. It then reads the length of the message and
     then the rest. Push back to client the region with the address and length
     of message.
   */
  int Receive(std::vector<Region> &v) {
    static bool printed_once = false;
    if (!printed_once) {
      std::cout << "Receiver polling local address: 0x"
      << std::hex << (uint64_t)local_buffer->GetReadPtr() << std::dec
      << std::endl;
      printed_once = true;
    }
    MAGIC_BYTE_T completion = *(volatile MAGIC_BYTE_T *)local_buffer->GetReadPtr();
    int c = 0;

    while (completion) {
      // 1. Read the payload length
      LEN_BYTE_T length = *(volatile LEN_BYTE_T *)local_buffer->Read(sizeof(LEN_BYTE_T));

      // 2. Step back over the payload AND the 2-byte prefix (Magic + Opcode)
      char *whole_message = local_buffer->Read((uint32_t)length + sizeof(MAGIC_BYTE_T) + sizeof(msg_opcode_t));

      // 3. Push the pointer to the absolute start of the prefix!
      v.push_back({0, whole_message, (uint32_t)length, 0});
      c++;

      completion = *(volatile MAGIC_BYTE_T *)local_buffer->GetReadPtr();
    }
    return c;
  }

  // void FreeReceive(Region &region) {
  //   local_buffer->Free(region.length + sizeof(LEN_BYTE_T) +
  //                      sizeof(MAGIC_BYTE_T));
  // }
  void FreeReceive(Region &region) {
    // 2 (Prefix) + Payload + 5 (Suffix)
    local_buffer->Free(region.length +sizeof(msg_opcode_t) + sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T));
  }
  uint32_t GetFreedReceiveBytes() { return local_buffer->GetFreedBytes(); }


};
