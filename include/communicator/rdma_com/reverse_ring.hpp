#pragma once

#include "rdma_com/utils.hpp"
#include "ring.hpp"
#include <infiniband/verbs.h>

class ReverseRingBuffer : public Buffer {
public:
  ReverseRingBuffer() = delete;
  ReverseRingBuffer(ReverseRingBuffer const&) = delete;
  ReverseRingBuffer& operator=(ReverseRingBuffer const&) = delete;


  ReverseRingBuffer(struct ibv_mr * mr, uint32_t length_bit)
  :mr_(mr), addr_((char*)mr->addr), length_(1<<length_bit), length_mask_(length_-1),
  head_(std::numeric_limits<uint64_t>::max() ),freed_bytes(0), read_ptr_(0)
  {
    // nothing
  }
  ~ReverseRingBuffer() {
    ibv_dereg_mr(mr_);
    FreeMagicBuffer(addr_, length_);
  }

  uint32_t GetLength() const{
    return length_;
  }

  uint32_t GetKey() const{
    return mr_->rkey;
  }

  BufferContext GetContext() const{
    return BufferContext{(uint64_t)(void*)addr_,mr_->rkey,length_};
  }

  char *GetReadPtr() const{
    return this->addr_ + this->read_ptr_;
  }

  uint32_t GetReadOff() const{
    return this->read_ptr_;
  }

  char *Read(uint32_t len) {
    //uint32_t read_offset = this->read_ptr;
    this->read_ptr_ = (this->read_ptr_ + this->length_ - len) & this->length_mask_;  // it is -len in the field
    return this->addr_ + this->read_ptr_ ;
  }

  uint64_t Free(uint32_t len){
    freed_bytes+=len;
    this->head_-=len;
    return this->head_;
  }

  // can be implemented but does not make sense for me.
  uint64_t FreeOrdered(char *addr, uint32_t len){
    return Free(len);
  }

  uint32_t GetFreedBytes(){
    uint32_t ret = freed_bytes;
    this->freed_bytes = 0;
    return ret;
  }

private:

  struct ibv_mr * const mr_;
  char * const addr_;
  const uint32_t length_;
  const uint32_t length_mask_;

  // it is safe to not have "full" check as it is managed by the client
  uint64_t head_; // it is a fake head. real head is +1.
  uint32_t freed_bytes;
  uint32_t read_ptr_;

};


class ReverseRemoteBuffer : public RemoteBuffer {
public:
  ReverseRemoteBuffer() = delete;
  ReverseRemoteBuffer(ReverseRemoteBuffer const&) = delete;
  ReverseRemoteBuffer& operator=(ReverseRemoteBuffer const&) = delete;


  ReverseRemoteBuffer(BufferContext &bc):
    addr_(bc.addr), rkey_(bc.rkey), length_(bc.length), length_mask_(bc.length-1),
    head_( std::numeric_limits<uint64_t>::max() ),reverse_tail_(0),free_(bc.length)
  {

  }

  uint32_t GetKey() const {
    return this->rkey_;
  }

  uint32_t GetTail() const {
    return this->reverse_tail_;
  }

  // get tail for writing len bytes
  uint32_t GetNextTail(uint32_t len) const {
    return (this->reverse_tail_ + length_ - len ) & length_mask_;
  }

  uint64_t GetWriteAddr(uint32_t len){
    if(len<=this->free_){ // can write
      this->reverse_tail_ = (this->reverse_tail_ + length_ - len ) & length_mask_;
      free_-=len;
      return this->addr_ + this->reverse_tail_;
    }else{
      return 0;
    }
  }


  bool UpdateHead(uint64_t new_head){
    uint32_t freed_bytes = (uint32_t)(this->head_ - new_head);
    free_ += freed_bytes;
    this->head_ = new_head;
    return freed_bytes;
  }

  bool FreeBytes(uint32_t freed_bytes){
    free_ += freed_bytes;
    this->head_ -= freed_bytes;
    return freed_bytes;
  }


private:
  const uint64_t addr_;
  const uint32_t rkey_;
  const uint32_t length_;
  const uint32_t length_mask_;


  uint64_t head_;
  uint32_t reverse_tail_;
  uint32_t free_;
};
