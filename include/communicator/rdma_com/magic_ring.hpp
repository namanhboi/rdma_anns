#pragma once
#include <set>
#include "ring.hpp"

class MagicRingBuffer : public Buffer {
  public:
    MagicRingBuffer() = delete;
    MagicRingBuffer(MagicRingBuffer const&) = delete;
    MagicRingBuffer& operator=(MagicRingBuffer const&) = delete;


    MagicRingBuffer(struct ibv_mr * mr, uint32_t length_bit, bool with_zero)
    :mr_(mr), addr_((char*)mr->addr), length_(1<<length_bit), length_mask_(length_-1), with_zero_(with_zero), 
    head_(length_), freed_bytes(0), read_ptr_(0)
    {
        // nothing
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

    char *GetReadPtr(uint32_t offset) const{
      return this->addr_ + offset;
    }

    uint32_t GetReadOff() const{
      return this->read_ptr_;
    }

    uint32_t GetOffset(char* ptr) const{
      return ((uint32_t)(ptr - this->addr_)& this->length_mask_);
    }

    char *Read(uint32_t len) {
      	uint32_t read_offset = this->read_ptr_;
      	this->read_ptr_ = (this->read_ptr_ + len) & this->length_mask_;    
      	return this->addr_ + read_offset;  
    }

    uint64_t Free(uint32_t len){
      if(with_zero_){
      	memset(this->addr_+(this->head_ & this->length_mask_),0,len);
      }
      this->freed_bytes+=len;
      this->head_ += len; 
      return this->head_;
    }

    uint64_t FreeOrdered(char *addr, uint32_t len){
      uint32_t start = ((uint32_t)(addr - this->addr_) & this->length_mask_);
      if(((this->head_ - start) & this->length_mask_)  == 0 ){ // we free current head. i.e., start  == head_ (mod length)
      	start =  (start + len) & this->length_mask_; 
      	auto it = frees.lower_bound({start,0});
      	while(it != frees.end()){
      		if(it->first != start) break;
      		len += it->second;
      		start =  (start + it->second) & this->length_mask_; 
      		it = frees.erase(it);
          if(it == frees.end()){
            it = frees.begin();
          }
      	}
 
      	return Free(len);
      } 
      // else 
      frees.insert({start,len});
      return this->head_;
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
    const bool with_zero_;

    // it is safe to not have "full" check as it is managed by the client
    uint64_t head_;
    uint32_t freed_bytes;
    uint32_t read_ptr_;   

    std::set<std::pair<uint32_t,uint32_t>>  frees;
};




class MagicRemoteBuffer : public RemoteBuffer {
  public:
    MagicRemoteBuffer() = delete;
    MagicRemoteBuffer(MagicRemoteBuffer const&) = delete;
    MagicRemoteBuffer& operator=(MagicRemoteBuffer const&) = delete;

    MagicRemoteBuffer(BufferContext &bc): 
    addr_(bc.addr), rkey_(bc.rkey), length_(bc.length), length_mask_(bc.length-1),
    head_(0),tail_(0),free_(bc.length) 
    {

    }
    
    uint32_t GetKey() const {
      return this->rkey_;
    }

    uint32_t GetTail() const {
      return this->tail_;
    }

    // get tail for writing len bytes
    uint32_t GetNextTail(uint32_t len) const {
      return this->tail_;
    }


    uint64_t GetWriteAddr(uint32_t len){
      if(len<=this->free_){ // can write
      	uint32_t tail = this->tail_;
        this->tail_ = (this->tail_+len) & length_mask_;
        free_-=len;
        return this->addr_ + tail;
      }else{
        return 0;
      }
    }

    uint32_t GetOffset(uint64_t addr){
        return (uint32_t)( (addr - addr_) & length_mask_ );
    }

    uint64_t GetBaseAddr(){
        return  addr_;
    }


    bool UpdateHead(uint64_t new_head){
      uint32_t freed_bytes = (uint32_t)(new_head - this->head_) ;
      free_ += freed_bytes;
      this->head_ = new_head;
      return freed_bytes;
    }

    bool FreeBytes(uint32_t freed_bytes){
      free_ += freed_bytes;
      this->head_ += freed_bytes;
      return freed_bytes;
    }

  private:
    const uint64_t addr_;
    const uint32_t rkey_;
    const uint32_t length_;
    const uint32_t length_mask_;
    

    uint64_t head_;
    uint32_t tail_;
    uint32_t free_;
};


