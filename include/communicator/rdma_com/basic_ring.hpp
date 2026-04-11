#pragma once
#include "ring.hpp"

class BasicRingBuffer : public Buffer {
  public:

    BasicRingBuffer() = delete;
    BasicRingBuffer(BasicRingBuffer const&) = delete;
    BasicRingBuffer& operator=(BasicRingBuffer const&) = delete;


    BasicRingBuffer(struct ibv_mr * mr, uint32_t length, bool with_zero)
    :mr_(mr), addr_((char*)mr->addr), length_(length), with_zero_(with_zero), 
    head_(0), read_ptr_(0), freed_bytes(0)
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

    uint32_t GetReadOff() const{
      return this->read_ptr_;
    }

    char *Read(uint32_t len) {
      // We either read at the read_ptr or a the beginning of the buffer if there is not enough space
      if(this->read_ptr_ <= this->length_ - len){
            uint32_t read_offset = this->read_ptr_;
            this->read_ptr_ += len;
            return this->addr_ + read_offset;  
      } else {
       // printf("[local] wrap around\n");
        // wrap around
        this->read_ptr_ = len;  
        return this->addr_;  
      }      
    }

    // i assume in order free calls. 
    uint64_t Free(uint32_t len){
      if(with_zero_){
        uint32_t real_head = (uint32_t)(head_ % length_);
        if(real_head <= this->length_ - len){
            memset(this->addr_+real_head,0,len);
        }else{
         //   printf("[rem] free around\n");
            // wrap around
            memset(this->addr_+real_head,0,this->length_ - real_head);
            memset(this->addr_          ,0,len);
        }
      }
      this->freed_bytes+=len;
      this->head_ += len;
      return this->head_;
    }

    uint64_t FreeOrdered(char *addr, uint32_t len){
        //TODO. 
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
    const bool with_zero_;

    // it is safe to not have "full" check as it is managed by the client
    uint64_t head_;
    uint32_t read_ptr_; 
    uint32_t freed_bytes;   
};



class BasicRemoteBuffer : public RemoteBuffer {
  public:

    BasicRemoteBuffer() = delete;
    BasicRemoteBuffer(BasicRemoteBuffer const&) = delete;
    BasicRemoteBuffer& operator=(BasicRemoteBuffer const&) = delete;

    BasicRemoteBuffer(BufferContext &bc): 
    addr_(bc.addr), rkey_(bc.rkey), length_(bc.length),
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
      return (this->tail_  <= this->length_ - len) ? this->tail_ : 0;
    }

    uint64_t GetWriteAddr(uint32_t len){
      uint32_t tail = 0;
      uint32_t len_with_wrap = len;

      if(this->tail_  <= this->length_ - len){
        tail = this->tail_;
      } else {
     //   printf("[rem] wrap around\n");
        len_with_wrap+=(this->length_ - this->tail_); 
      }

      if(len_with_wrap<=free_){ // can write
        this->tail_ = tail + len;
        free_-=len_with_wrap;
        return this->addr_ + tail;
      }else{
        return 0;
      }
    }

    bool UpdateHead(uint64_t new_head){
      uint32_t freed_bytes = (uint32_t)(new_head - this->head_);
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
    
    uint64_t head_;
    uint32_t tail_;
    uint32_t free_;
};


