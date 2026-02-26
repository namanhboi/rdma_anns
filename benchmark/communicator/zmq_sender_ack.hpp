#pragma once
#include "communicator.h"
#include <chrono>
#include <limits>
#include <thread>

class AckNode {
private:
  ZMQP2PCommunicator server;
  void ack(const char *data, size_t size) {
    char *tmp = new char[size];
    std::memcpy(tmp, data, size);
    uint64_t peer_id = *reinterpret_cast<uint64_t *>(tmp);
    Region r;  r.addr = tmp;
    r.length = static_cast<uint32_t>(size);
    r.context = 0;
    r.lkey = 0;
    server.send_to_peer(peer_id, &r);
  }
public :
  AckNode(uint64_t my_id, const std::string &config_file)
  : server(my_id, config_file) {
    server.register_receive_handler([this](const char *data, size_t size) {
      this->ack(data, size);
    });
  }
  void start() { server.start_recv_thread(); }
};

class SenderNode {
private:
  ZMQP2PCommunicator server;
  uint32_t num_msg, msg_size;
  std::unordered_map<uint64_t,std::chrono::steady_clock::time_point> send_time;
  std::unordered_map<uint64_t, std::chrono::steady_clock::time_point>
      receive_time;
  std::atomic<uint64_t> num_ack_received {0};
  void receive_ack(const char *data, size_t size) {
    const uint64_t *header = reinterpret_cast<const uint64_t *>(data);
    uint64_t msg_id = header[1];
    //warmup 
    if (msg_id == std::numeric_limits<uint64_t>::max())
      return;
    
    receive_time[msg_id] = std::chrono::steady_clock::now();
    num_ack_received.fetch_add(1);
  }
  std::atomic<uint64_t> msg_id{0};
public:
  SenderNode(uint64_t my_id, const std::string &config_file, uint32_t num_msg,
             uint32_t msg_size)
  : server(my_id, config_file), num_msg(num_msg), msg_size(msg_size) {
    server.register_receive_handler([this](const char *data, size_t size) {
      this->receive_ack(data, size);
    });
  }

  void blocking_send_msgs(int num_warmup) {
    std::vector<uint64_t> other_peers;
    for (uint64_t i = 0; i < server.get_num_peers(); i++) {
      if (i != server.get_my_id()) 
	other_peers.push_back(i);
    }
    for (size_t i = 0; i < num_warmup; i++) {
      for (const uint64_t peer_id : other_peers) {
        char *tmp = new char[msg_size];
        uint64_t *region_header = reinterpret_cast<uint64_t *>(tmp);
        region_header[0] = server.get_my_id();
        region_header[1] = std::numeric_limits<uint64_t>::max();
        Region r;
        r.addr = tmp;
        r.length = msg_size;
        server.send_to_peer(peer_id, &r);
      }
    }

    for (size_t i = 0; i < num_msg; i++) {
      for (const uint64_t peer_id : other_peers) {
        char *tmp = new char[msg_size];
        uint64_t *region_header = reinterpret_cast<uint64_t *>(tmp);
        region_header[0] = server.get_my_id();
        region_header[1] = msg_id.fetch_add(1);
        Region r;
        r.addr = tmp;
        r.length = msg_size;
        send_time[region_header[1]] = std::chrono::steady_clock::now();
        server.send_to_peer(peer_id, &r);
      }
    }
    std::cout << "sent all msgs" << std::endl;
  }
  void start_recv_thread() { server.start_recv_thread(); }
  void stop_recv_thread() { server.stop_recv_thread(); }

  void blocking_wait_all_acks() {
    std::cout << "starting to wait for acks" << std::endl;
    while (num_ack_received != num_msg) {
      std::cout << "received " << num_ack_received << "/" << num_msg
      << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }
  ~SenderNode() {
    std::vector<double> latencies;
    std::chrono::steady_clock::time_point first = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last;
    double sum = 0.0;
    for(uint64_t i=0; i<num_msg; i++){
      auto msg_id = i;
      auto &sent = send_time[msg_id];
      auto &received = receive_time[msg_id];
      std::chrono::microseconds elapsed =
          std::chrono::duration_cast<std::chrono::microseconds>(received -
                                                                sent);
      double lat = static_cast<double>(elapsed.count()) / 1000.0;
      latencies.push_back(lat);
      sum += lat;
      first = std::min(first,sent);
      last = std::max(last,received);
    }
    std::chrono::microseconds total_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(last - first);
    double total_time = static_cast<double>(total_elapsed.count()) / 1000000.0;
    double throughput = (num_msg) / total_time;
    std::sort(latencies.begin(),latencies.end());
    double avg = sum / latencies.size();
    double min = latencies.front();
    double max = latencies.back();
    auto median = latencies[latencies.size()/2];
    auto p95 = latencies[(uint64_t)(latencies.size()*0.95)];

    std::cout << "Throughput: " << throughput << " queries/s" << " (" << num_msg << " queries in " << total_time << " seconds)" << std::endl;
    std::cout << "E2E latency:" << std::endl;
    std::cout << "  avg: " << avg << std::endl;
    std::cout << "  median: " << median << std::endl;
    std::cout << "  min: " << min << std::endl;
    std::cout << "  max: " << max << std::endl;
    std::cout << "  p95: " << p95 << std::endl;
  }
};
