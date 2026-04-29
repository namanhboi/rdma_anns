#include "communicator.h"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

namespace po = boost::program_options;

struct TestMsgHeader {
  uint64_t msg_id;
  uint64_t src_peer_id;
  uint64_t dst_peer_id;
};

static constexpr size_t TEST_MSG_HEADER_BYTES = sizeof(TestMsgHeader);
static constexpr size_t TEST_MSG_TRAILER_BYTES = sizeof(uint64_t);
static constexpr size_t TEST_MSG_MIN_BYTES =
    TEST_MSG_HEADER_BYTES + TEST_MSG_TRAILER_BYTES;

static void fill_test_message(Region *r,
                              uint32_t msg_size,
                              uint64_t src_peer_id,
                              uint64_t dst_peer_id,
                              uint64_t msg_id) {
  r->length = msg_size;

  TestMsgHeader header;
  header.msg_id = msg_id;
  header.src_peer_id = src_peer_id;
  header.dst_peer_id = dst_peer_id;

  // Put the message metadata at the start.
  std::memcpy(r->addr, &header, sizeof(header));

  // Fill the middle payload with a deterministic pattern.
  if (msg_size > TEST_MSG_MIN_BYTES) {
    std::memset(r->addr + TEST_MSG_HEADER_BYTES,
                static_cast<int>(msg_id & 0xff),
                msg_size - TEST_MSG_MIN_BYTES);
  }

  // Put msg_id again at the end.
  std::memcpy(r->addr + msg_size - sizeof(uint64_t),
              &msg_id,
              sizeof(uint64_t));
}

int main(int argc, char **argv) {
  po::options_description desc("Options here mate");

  uint64_t server_peer_id;
  std::vector<std::string> address_list;
  uint32_t num_msg;
  uint32_t msg_size;

  desc.add_options()
      ("server_peer_id",
       po::value<uint64_t>(&server_peer_id)->required(),
       "Server peer ID")
      ("address_list",
       po::value<std::vector<std::string>>(&address_list)
           ->multitoken()
           ->required(),
       "Address list")
      ("num_msg",
       po::value<uint32_t>(&num_msg)->required(),
       "num msg to send between the servers")
      ("msg_size",
       po::value<uint32_t>(&msg_size)->required(),
       "msg_size");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  po::notify(vm);

  if (msg_size > Region::MAX_BYTES_REGION) {
    std::cerr << "Error: msg_size exceeds MAX_BYTES_REGION!" << std::endl;
    return 1;
  }

  if (msg_size < TEST_MSG_MIN_BYTES) {
    std::cerr << "Error: msg_size must be at least "
              << TEST_MSG_MIN_BYTES
              << " bytes for test header/trailer checks!"
              << std::endl;
    return 1;
  }

  if (server_peer_id >= address_list.size()) {
    std::cerr << "Error: server_peer_id "
              << server_peer_id
              << " is out of range for address_list size "
              << address_list.size()
              << std::endl;
    return 1;
  }

  std::cout << "server_peer_id:" << server_peer_id << std::endl;
  std::cout << "address list of " << address_list.size() << " servers"
            << std::endl;

  for (size_t i = 0; i < address_list.size(); i++) {
    std::cout << address_list[i] << ",";
  }
  std::cout << std::endl;

  // RDMAP2PCommunicator communicator(server_peer_id, address_list);
  std::unique_ptr<P2PCommunicator> communicator =
    P2PCommunicator::create_communicator(true, server_peer_id, address_list);


  const uint64_t num_peers = communicator->get_num_peers();
  const uint64_t my_id = communicator->get_my_id();

  std::atomic<uint64_t> num_received{0};
  std::atomic<uint64_t> num_errors{0};

  // For each source peer, track the next expected msg_id.
  // Messages from different peers can interleave, so this must be per source.
  std::vector<uint64_t> expected_next_from_peer(num_peers, 0);

  communicator->register_receive_handler(
      [&num_received,
       &num_errors,
       &expected_next_from_peer,
       my_id,
       num_peers](const char *data, size_t len) {
        if (len < TEST_MSG_MIN_BYTES) {
          std::cerr << "ERROR: received too-small message, len="
                    << len << std::endl;
          num_errors.fetch_add(1);
          std::abort();
        }

        TestMsgHeader header;
        std::memcpy(&header, data, sizeof(header));

        uint64_t trailer_msg_id = 0;
        std::memcpy(&trailer_msg_id,
                    data + len - sizeof(uint64_t),
                    sizeof(uint64_t));

        if (header.src_peer_id >= num_peers) {
          std::cerr << "ERROR: invalid src_peer_id="
                    << header.src_peer_id << std::endl;
          num_errors.fetch_add(1);
          std::abort();
        }

        if (header.dst_peer_id != my_id) {
          std::cerr << "ERROR: message for wrong dst. expected dst="
                    << my_id
                    << " got dst=" << header.dst_peer_id
                    << " src=" << header.src_peer_id
                    << " msg_id=" << header.msg_id
                    << std::endl;
          num_errors.fetch_add(1);
          std::abort();
        }

        if (header.msg_id != trailer_msg_id) {
          std::cerr << "ERROR: start/end msg_id mismatch. start="
                    << header.msg_id
                    << " end=" << trailer_msg_id
                    << " src=" << header.src_peer_id
                    << " dst=" << header.dst_peer_id
                    << std::endl;
          num_errors.fetch_add(1);
          std::abort();
        }

        uint64_t expected = expected_next_from_peer[header.src_peer_id];

        if (header.msg_id != expected) {
          std::cerr << "ERROR: out-of-order or corrupted message from src="
                    << header.src_peer_id
                    << ". expected msg_id=" << expected
                    << " got msg_id=" << header.msg_id
                    << " dst=" << header.dst_peer_id
                    << std::endl;
          num_errors.fetch_add(1);
          std::abort();
        }

        expected_next_from_peer[header.src_peer_id]++;

        uint64_t received_now =
            num_received.fetch_add(1, std::memory_order_release) + 1;

        if (received_now % 100000 == 0) {
          std::cout << "received: " << received_now << std::endl;
        }
      });

  PreallocatedQueue<Region> prealloc_region_queue(12000, Region::reset);

  std::pair<char *, uint32_t> ptr_lkey =
      communicator->get_preallocated_region_ptr_lkey(Region::MAX_BYTES_REGION,
                                                    12000);

  char *region_addr = ptr_lkey.first;
  uint32_t lkey = ptr_lkey.second;

  prealloc_region_queue.assign_additional_block_mr(region_addr,
                                                   lkey,
                                                   Region::MAX_BYTES_REGION,
                                                   Region::assign_addr);

  communicator->start_recv_thread();

  std::cout << "Starting all-to-all send. num_msg per peer="
            << num_msg
            << ", msg_size="
            << msg_size
            << std::endl;

  for (uint64_t msg_id = 0; msg_id < num_msg; msg_id++) {
    for (uint64_t peer_id = 0; peer_id < num_peers; peer_id++) {
      if (peer_id == my_id) continue;

      Region *r = nullptr;
      prealloc_region_queue.dequeue_exact(1, &r);

      fill_test_message(r,
                        msg_size,
                        my_id,
                        peer_id,
                        msg_id);

      communicator->send_to_peer(peer_id, r);
    }
  }

  uint64_t expected_received =
      static_cast<uint64_t>(num_msg) *
      static_cast<uint64_t>(num_peers - 1);

  std::cout << "Finished enqueueing sends. Waiting for "
            << expected_received
            << " received messages..."
            << std::endl;

  while (num_received.load(std::memory_order_acquire) < expected_received) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cout << "All expected messages received: "
            << num_received.load(std::memory_order_acquire)
            << std::endl;

  if (num_errors.load() != 0) {
    std::cerr << "Test failed with "
              << num_errors.load()
              << " errors"
              << std::endl;
    return 1;
  }

  for (uint64_t peer_id = 0; peer_id < num_peers; peer_id++) {
    if (peer_id == my_id) continue;

    std::cout << "from peer " << peer_id
              << ": received "
              << expected_next_from_peer[peer_id]
              << " messages"
              << std::endl;

    if (expected_next_from_peer[peer_id] != num_msg) {
      std::cerr << "ERROR: expected "
                << num_msg
                << " messages from peer "
                << peer_id
                << ", got "
                << expected_next_from_peer[peer_id]
                << std::endl;
      return 1;
    }
  }

  std::cout << "Correctness check passed." << std::endl;
  std::cout << "Press q to exit." << std::endl;

  std::string shutdown;
  while (shutdown != "q") {
    std::cin >> shutdown;
  }

  return 0;
}
