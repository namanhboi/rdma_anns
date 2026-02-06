#pragma once

#include <cstddef>
#include <vector>
#include <limits>
#include "utils.h"
#include <array>

namespace pipeann {

  struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {
    }

    inline bool operator<(const Neighbor &other) const {
      return (distance < other.distance) || (distance == other.distance && id < other.id);
    }
    inline bool operator==(const Neighbor &other) const {
      return (id == other.id);
    }
    inline bool operator>(const Neighbor &other) const {
      return (distance > other.distance) || (distance == other.distance && id > other.id);
    }
  };

  template<typename TagT = int>
  struct NeighborTag {
    TagT tag;
    float dist;
    NeighborTag() = default;

    NeighborTag(TagT tag, float dist) : tag{tag}, dist{dist} {
    }
    inline bool operator<(const NeighborTag &other) const {
      return (dist < other.dist) || (dist == other.dist && tag < other.tag);
    }
    inline bool operator==(const NeighborTag &other) const {
      return (tag == other.tag);
    }
  };


  static inline unsigned InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    unsigned left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }
    while (right > 1 && left < right - 1) {
      unsigned mid = (left + right) / 2;
      if (addr[mid].distance > nn.distance)
        right = mid;
      else
        left = mid;
    }
    // check equal ID

    while (left > 0) {
      if (addr[left].distance < nn.distance)
        break;
      if (addr[left].id == nn.id)
        return K + 1;
      left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
      return K + 1;
    memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
  }
}  // namespace pipeann


static inline std::pair<unsigned, unsigned> BatchInsertIntoPool(
    pipeann::Neighbor *addr, unsigned K, unsigned L,
    std::pair<uint32_t, float> *sorted_list,
    uint32_t len_sorted_list) {

    // fast path: nothing to insert
    if (len_sorted_list == 0) return {K, K};

    // We rely on L <= 512 for stack buffer; assert to catch misuse.
    assert(L <= 512);

    if (K > L) K = L; // defensive clamp

    // Temp buffer on stack (max 512)
    std::array<pipeann::Neighbor, 512> temp;
    unsigned temp_count = 0;

    // Small open-addressing hash for seen IDs.
    // Power-of-two size for cheap masking. 2048 gives low load for L<=512.
    constexpr unsigned HSIZE = 2048u;
    constexpr unsigned HMASK = HSIZE - 1u;
    static_assert((HSIZE & HMASK) == 0, "HSIZE must be power of two");

    const uint32_t SENTINEL = std::numeric_limits<uint32_t>::max();
    std::array<uint32_t, HSIZE> seen;
    // initialize seen to sentinel
    for (unsigned i = 0; i < HSIZE; ++i) seen[i] = SENTINEL;

    auto seen_insert = [&](uint32_t id) -> bool {
        // return true if id was newly inserted into seen; false if already present
        unsigned idx = id & HMASK;
        for (unsigned probe = 0; probe < HSIZE; ++probe) {
            unsigned s = (idx + probe) & HMASK;
            uint32_t cur = seen[s];
            if (cur == SENTINEL) {
                seen[s] = id;
                return true;
            }
            if (cur == id) return false;
        }
        // table full -- extremely unlikely; treat as duplicate (conservative)
        return false;
    };

    unsigned read_pos = 0;           // index into addr[0..K)
    unsigned insert_pos = 0;         // index into sorted_list[0..len_sorted_list)
    const float INF = std::numeric_limits<float>::infinity();

    // Hot merge loop
    while (temp_count < L && (read_pos < K || insert_pos < len_sorted_list)) {
        // Branch-minimized selection of distances
        float d0 = (read_pos < K) ? addr[read_pos].distance : INF;
        float d1 = (insert_pos < len_sorted_list) ? sorted_list[insert_pos].second : INF;
        bool take_from_addr = (d0 <= d1);

        if (take_from_addr) {
            // take from addr
            uint32_t id = addr[read_pos].id;
            ++read_pos;

            // skip duplicates already placed into temp
            if (!seen_insert(id)) continue;

            // fast copy existing neighbor (source index is read_pos-1)
            temp[temp_count] = addr[read_pos - 1];
            ++temp_count;
        } else {
            // take from sorted_list
            uint32_t id = sorted_list[insert_pos].first;
            float dist  = sorted_list[insert_pos].second;
            ++insert_pos;

            // skip duplicates
            if (!seen_insert(id)) continue;

            // construct neighbor for new item (mark flag true as before)
            temp[temp_count].id = id;
            temp[temp_count].distance = dist;
            temp[temp_count].flag = true;
            ++temp_count;
        }
    }

    unsigned new_size = temp_count;

    // Compute lowest_insert_idx by comparing temp vs original addr
    unsigned lowest_insert_idx = K; // default: no change
    unsigned compare_limit = (new_size < K) ? new_size : K;
    bool found_diff = false;
    for (unsigned i = 0; i < compare_limit; ++i) {
        if (temp[i].id != addr[i].id || temp[i].distance != addr[i].distance) {
            lowest_insert_idx = i;
            found_diff = true;
            break;
        }
    }
    if (!found_diff) {
        if (new_size != K) {
            lowest_insert_idx = std::min<unsigned>(K, new_size);
        } else {
            lowest_insert_idx = K;
        }
    }

    // Copy temp back to addr
    if (new_size > 0) {
        std::memcpy(addr, temp.data(), new_size * sizeof(pipeann::Neighbor));
    }

    return {lowest_insert_idx, new_size};
}

