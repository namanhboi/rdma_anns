#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <cascade/object.hpp>

/// USED FOR DUMMY UDL

/**
   this is sent from the client to the udl, with the expectation that it will
   forward this request to another udl.

precondition: the udl receiving this request must have the same cluster id as
cluster_sender_id. This udl will then send a send_object_t to the expectant udl
with the bytes created. The recipeient udl will then send an acknowledgement
back to the client.

   
 */

struct send_request_t {
  uint64_t message_id;
  uint64_t num_bytes;
  uint64_t client_node_id;
  uint8_t cluster_sender_id;
  uint8_t cluster_receiver_id;


  size_t get_serialize_size() {
    return sizeof(message_id) + sizeof(num_bytes) + sizeof(cluster_sender_id) + sizeof(cluster_receiver_id) + sizeof(client_node_id);
  }

  void write_serialize(uint8_t *buffer) {
    size_t offset = 0;
    std::memcpy(buffer + offset, &message_id, sizeof(message_id));
    offset += sizeof(message_id);

    std::memcpy(buffer + offset, &num_bytes, sizeof(num_bytes));
    offset += sizeof(num_bytes);

    std::memcpy(buffer + offset, &client_node_id, sizeof(client_node_id));
    offset += sizeof(client_node_id);

    std::memcpy(buffer + offset, &cluster_sender_id, sizeof(cluster_sender_id));
    offset += sizeof(cluster_sender_id);

    std::memcpy(buffer + offset, &cluster_receiver_id, sizeof(cluster_receiver_id));
    offset += sizeof(cluster_receiver_id);
  }
  
  static std::shared_ptr<send_request_t> deserialize(const uint8_t *buffer) {
    size_t offset = 0;
    auto req = std::make_shared<send_request_t>();
    req->message_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(req->message_id);

    req->num_bytes = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(req->num_bytes);

    req->client_node_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(req->client_node_id);
    
    req->cluster_sender_id = *reinterpret_cast<const uint8_t *>(buffer + offset);
    offset += sizeof(req->cluster_sender_id);

    req->cluster_receiver_id = *reinterpret_cast<const uint8_t *>(buffer + offset);
    offset += sizeof(req->cluster_receiver_id);

    return req;
  }

  static std::vector<std::shared_ptr<send_request_t>> deserialize_send_requests(const uint8_t *buffer) {
    std::vector<std::shared_ptr<send_request_t>> send_reqs;
    size_t offset = 0;
    uint64_t num_requests = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(num_requests);
    for (auto i = 0; i < num_requests; i++) {
      auto req = send_request_t::deserialize(buffer + offset);
      send_reqs.push_back(req);
      offset += req->get_serialize_size();
    }
    return send_reqs;
  }

  static size_t get_serialize_size_send_requests(
						 const std::vector<std::shared_ptr<send_request_t>> &send_requests) {
    size_t total_size = 0;
    total_size += sizeof(uint64_t); // header which is just number of objects
    for (auto &send_req : send_requests) {
      total_size += send_req->get_serialize_size();
    }
    return total_size;
  }

  static void write_serialize_send_requests(
					    const std::vector<std::shared_ptr<send_request_t>> &send_requests,
					    uint8_t *buffer) {
    size_t offset = 0;
    uint64_t num_requests = send_requests.size();

    std::memcpy(buffer + offset, &num_requests, sizeof(num_requests));
    offset += sizeof(num_requests);

    for (auto &req : send_requests) {
      req->write_serialize(buffer + offset);
      offset += req->get_serialize_size();
    }
  }

  static 
  std::shared_ptr<derecho::cascade::Blob> get_send_requests_blob(
								 std::vector<std::shared_ptr<send_request_t>> send_requests) {
    size_t total_size = send_request_t::get_serialize_size_send_requests(send_requests);
    // can revert to [&] once we find out when a blob object's bytes are actually written
    return std::make_shared<derecho::cascade::Blob>(
						    [send_requests = std::move(send_requests)](uint8_t *buffer, size_t size) {
						      send_request_t::write_serialize_send_requests(send_requests, buffer);
						      return size;
						    },
						    total_size);
  }  
};


struct send_object_t {
  uint64_t message_id;
  uint64_t num_bytes;
  uint64_t client_node_id;
  uint8_t cluster_sender_id;
  uint8_t cluster_receiver_id;
  std::shared_ptr<uint8_t[]> bytes;

  size_t get_serialize_size()const {
    return sizeof(message_id) + sizeof(num_bytes) + num_bytes +
           sizeof(cluster_sender_id) + sizeof(cluster_receiver_id) + sizeof(client_node_id);
  }
  void write_serialize(uint8_t *buffer) const {
    size_t offset = 0;
    std::memcpy(buffer + offset, &message_id, sizeof(message_id));
    offset += sizeof(message_id);

    std::memcpy(buffer + offset, &num_bytes, sizeof(num_bytes));
    offset += sizeof(num_bytes);

    std::memcpy(buffer + offset, &client_node_id, sizeof(client_node_id));
    offset += sizeof(client_node_id);

    std::memcpy(buffer + offset, &cluster_sender_id, sizeof(cluster_sender_id));
    offset += sizeof(cluster_sender_id);

    std::memcpy(buffer + offset, &cluster_receiver_id, sizeof(cluster_receiver_id));
    offset += sizeof(cluster_receiver_id);
    
    std::memcpy(buffer + offset, bytes.get(), num_bytes);
    offset += num_bytes;
  }

  static std::shared_ptr<send_object_t> deserialize(const uint8_t *buffer) {
    size_t offset = 0;
    auto obj = std::make_shared<send_object_t>();
    obj->message_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(obj->message_id);

    obj->num_bytes = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(obj->num_bytes);

    obj->client_node_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(obj->client_node_id);
    
    obj->cluster_sender_id = *reinterpret_cast<const uint8_t *>(buffer + offset);
    offset += sizeof(obj->cluster_sender_id);

    obj->cluster_receiver_id = *reinterpret_cast<const uint8_t *>(buffer + offset);
    offset += sizeof(obj->cluster_receiver_id);

    std::shared_ptr<uint8_t[]> tmp(new uint8_t[obj->num_bytes]);
    obj->bytes = tmp;
    std::memcpy(obj->bytes.get(), buffer + offset, obj->num_bytes);

    return obj;
  }

  static std::vector<std::shared_ptr<send_object_t>>
  deserialize_send_objects(const uint8_t *buffer) {
    std::vector<std::shared_ptr<send_object_t>> send_objs;
    size_t offset = 0;
    uint64_t num_objs = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(num_objs);
    for (auto i = 0; i < num_objs; i++) {
      auto obj = send_object_t::deserialize(buffer + offset);
      send_objs.push_back(obj);
      offset += obj->get_serialize_size();
    }
    return send_objs;
  }
  
  static size_t get_serialize_size_send_objects(
						const std::vector<std::shared_ptr<send_object_t>> &send_objects) {

    size_t total_size = 0;
    total_size += sizeof(uint64_t); // header which is just number of objects
    for (auto &send_obj : send_objects) {
      total_size += send_obj->get_serialize_size();
    }
    return total_size;
  }
  

  static void write_serialize_send_objects(
					   const std::vector<std::shared_ptr<send_object_t>> &send_objects,
					   uint8_t *buffer) {
    size_t offset = 0;
    size_t num_objects = send_objects.size();

    std::memcpy(buffer + offset, &num_objects, sizeof(num_objects));
    offset += sizeof(num_objects);

    for (auto &obj : send_objects) {
      obj->write_serialize(buffer + offset);
      offset += obj->get_serialize_size();
    }
  }
  static 
  std::shared_ptr<derecho::cascade::Blob> get_send_objects_blob(
								std::vector<std::shared_ptr<send_object_t>> send_objects) {
    size_t total_size = send_object_t::get_serialize_size_send_objects(send_objects);
    // can revert to [&] once we find out when a blob object's bytes are actually written
    return std::make_shared<derecho::cascade::Blob>(
						    [send_objects = std::move(send_objects)](uint8_t *buffer, size_t size) {
						      send_object_t::write_serialize_send_objects(send_objects, buffer);
						      return size;
						    },
						    total_size);
  }
};

/**
   the receiver udl once it has received a send_object_t will send back ack_t to
   the client to acknowledge that that send request has been completed. Client will
   wait for all send request to complete.

 */
struct ack_t {
  uint64_t message_id;
  uint64_t num_bytes;
  uint64_t client_node_id;

  size_t get_serialize_size() { return sizeof(message_id) + sizeof(num_bytes) + sizeof(client_node_id); }

  void write_serialize(uint8_t *buffer) const {
    size_t offset = 0;
    std::memcpy(buffer + offset, &message_id, sizeof(message_id));
    offset += sizeof(message_id);

    std::memcpy(buffer + offset, &num_bytes, sizeof(num_bytes));
    offset += sizeof(num_bytes);

    std::memcpy(buffer + offset, &client_node_id, sizeof(client_node_id));
    offset += sizeof(client_node_id);
  }
  
  static std::shared_ptr<ack_t> deserialize(const uint8_t *buffer) {
    size_t offset = 0;
    auto ack = std::make_shared<ack_t>();
    ack->message_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(ack->message_id);

    ack->num_bytes = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(ack->num_bytes);

    ack->client_node_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(ack->client_node_id);    
    return ack;
  }

  static std::vector<std::shared_ptr<ack_t>> deserialize_acks(const uint8_t *buffer) {
    std::vector<std::shared_ptr<ack_t>> acks;
    size_t offset = 0;
    uint64_t num_acks = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(num_acks);
    for (auto i = 0; i < num_acks; i++) {
      auto ack = deserialize(buffer + offset);
      acks.push_back(ack);
      offset += ack->get_serialize_size();
    }
    return acks;
  }

  static size_t get_serialize_size_acks(
					const std::vector<std::shared_ptr<ack_t>> &acks) {
    size_t total_size = 0;
    total_size += sizeof(uint64_t); // header which is just number of objects
    for (auto &ack : acks) {
      total_size += ack->get_serialize_size();
    }
    return total_size;
  }


  static void
  write_serialize_acks(const std::vector<std::shared_ptr<ack_t>> &acks,
                       uint8_t *buffer) {
    size_t offset = 0;
    size_t num_acks = acks.size();

    std::memcpy(buffer + offset, &num_acks, sizeof(num_acks));
    offset += sizeof(num_acks);

    for (auto &ack : acks) {
      ack->write_serialize(buffer + offset);
      offset += ack->get_serialize_size();
    }
  }
  static std::shared_ptr<derecho::cascade::Blob>
  get_acks_blob(std::vector<std::shared_ptr<ack_t>> acks) {
    size_t total_size = ack_t::get_serialize_size_acks(acks);
    // can revert to [&] once we find out when a blob object's bytes are actually written
    return std::make_shared<derecho::cascade::Blob>(
						    [acks = std::move(acks)](uint8_t *buffer, size_t size) {
						      ack_t::write_serialize_acks(acks, buffer);
						      return size;
						    },
						    total_size);
  }
};


/**
   Used to test the dummy udl where there is no serialization of send objects. There is also no acks, the udl flushes the logs automatically. The send object will then be send round robin style to each other udls. 
*/
struct send_query_t {
  uint64_t client_node_id;
  uint64_t num_bytes;
  uint64_t num_msg;
  uint8_t cluster_sender_id; // id of the udl that this send query is supposed to go to
  uint8_t num_clusters;

  size_t get_serialize_size() {
    return sizeof(num_bytes) + sizeof(num_msg) + sizeof(num_clusters) +
           sizeof(cluster_sender_id) + sizeof(client_node_id);
  }

  void write_serialize(uint8_t *buffer) const {
    size_t offset = 0;
    std::memcpy(buffer + offset, &client_node_id, sizeof(client_node_id));
    offset += sizeof(client_node_id);    

    std::memcpy(buffer + offset, &num_bytes, sizeof(num_bytes));
    offset += sizeof(num_bytes);

    std::memcpy(buffer + offset, &num_msg, sizeof(num_msg));
    offset += sizeof(num_msg);

    std::memcpy(buffer + offset, &cluster_sender_id, sizeof(cluster_sender_id));
    offset += sizeof(cluster_sender_id);

    std::memcpy(buffer + offset, &num_clusters, sizeof(num_clusters));
    offset += sizeof(num_clusters);
  }
  static std::shared_ptr<send_query_t> deserialize(const uint8_t *buffer) {
    size_t offset = 0;
    auto client_send_query = std::make_shared<send_query_t>();
    client_send_query->client_node_id = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(client_send_query->client_node_id);

    client_send_query->num_bytes = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(client_send_query->num_bytes);

    client_send_query->num_msg = *reinterpret_cast<const uint64_t *>(buffer + offset);
    offset += sizeof(client_send_query->num_msg);

    client_send_query->cluster_sender_id = *reinterpret_cast<const uint8_t *>(buffer + offset);
    offset += sizeof(client_send_query->cluster_sender_id);

    client_send_query->num_clusters = *reinterpret_cast<const uint8_t *>(buffer + offset);
    offset += sizeof(client_send_query->num_clusters);
    return client_send_query;
  }
  std::shared_ptr<derecho::cascade::Blob> get_blob() {
    size_t total_size = get_serialize_size();
    return std::make_shared<derecho::cascade::Blob>(
        [this](uint8_t *buffer, size_t size) {
          this->write_serialize(buffer);
          return size;
        },
						    total_size);
  }
};



