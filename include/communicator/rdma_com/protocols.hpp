#pragma once
#include "communicator.hpp"
#include "ring.hpp"
#include "magic_ring.hpp"
#include <list>
#include <unordered_map>

#define MAGIC_BYTE_T uint64_t
#define LEN_BYTE_T   uint64_t

#define MIN(a,b) (((a)<(b))?(a):(b))


#define PREFIX_SIZE (64)
#define SUFFIX_SIZE (64)


class SendConnection:  public SendCommunicator {
  VerbsEP* const ep; 
    
  uint64_t last_wrid = 0;
  uint64_t next_id_ = 1;

public:
  SendConnection(VerbsEP* ep): ep(ep) {
    // empty
  };
  ~SendConnection( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Send Sender\n");
    printf("Role: Sender\n");
    printf("Type: P2P.\n");
    printf("Used OPs: RDMA send\n");
    printf("Way of informing the receiver: recv completion\n");
    printf("Prefix: No.\n");
    printf("Suffix: No.\n");
    printf("Downsides: polling on receiver\n");
    printf("-----------------------------------------\n");
  }
    

  uint32_t ReqSize(Region& region) { return region.length; }

  uint64_t SendAsync(Region& region){
    uint64_t send_addr = (uint64_t)(void*)region.addr;
    uint32_t real_length = region.length;
 
    struct ibv_sge sge = {send_addr, real_length, region.lkey };
    ep->send(&sge, 1, next_id_);  

    return next_id_++;    
  }

  bool AckSentBytes(uint32_t bytes){
    return true;
  }

  void WaitSend(uint64_t id){ 
    while(last_wrid < id){
      TestSend(id);
    }
  }

  bool TestSend(uint64_t id){ 
    if(last_wrid >= id) return true;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS){
        printf("Failed request %d \n",wcs[i].status);
        exit(1);  
      } 
      last_wrid = wcs[i].wr_id;
    }
    if(last_wrid >= id) return true;
    return false;
  }
};


class ReceiveReceiver:  public ReceiveCommunicator {
  VerbsEP* const ep;
  Buffer* const local_buffer;
  const uint32_t lkey;
  const uint32_t size;
 
public:
  ReceiveReceiver(VerbsEP* ep, Buffer* local_buffer,  uint32_t size, uint32_t max_recv_size): 
    ep(ep), local_buffer(local_buffer), lkey(local_buffer->GetKey()), size(size)
  {   
    uint32_t i =0;
    char* addr = local_buffer->GetReadPtr();
    char* end = addr + local_buffer->GetLength();
    while(addr+size<end && i < max_recv_size){
      struct ibv_sge sge = {(uint64_t)(void*)addr, size, lkey };
      ep->post_recv(sge);
      addr+=size;
      i++;
    }
    ep->trigget_post_recv();
  };
  ~ReceiveReceiver( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Receive Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: P2P.\n");
    printf("Used OPs: Yes - post recv\n");
    printf("Passive: Yes\n");
    printf("-----------------------------------------\n");
  }

  int Receive(std::vector<Region> &v){
    int c = 0;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->recv_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].opcode == IBV_WC_RECV){
        uint32_t length = wcs[i].byte_len;
        char* whole_message = (char*)(void*)wcs[i].wr_id;
        v.push_back({0, whole_message, (uint32_t)length, lkey }); 
        c++;
      }
    }
    return c;  
  }

  void FreeReceive(Region& region){
    local_buffer->Free(size);
    struct ibv_sge sge = {(uint64_t)(void*)region.addr, size, lkey };
    ep->post_recv(sge);
  }

  uint32_t GetFreedReceiveBytes(){
    ep->trigget_post_recv();
    return local_buffer->GetFreedBytes();
  }
};



class SharedSendConnection:  public SendCommunicator {
  VerbsEP* const ep; 
    
  uint64_t last_wrid = 0;
  uint64_t next_id_ = 1;

public:
  SharedSendConnection(VerbsEP* ep): ep(ep) {
    // empty
  };
  ~SharedSendConnection( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Send Sender\n");
    printf("Role: Sender\n");
    printf("Type: N to 1.\n");
    printf("Used OPs: RDMA send\n");
    printf("Way of informing the receiver: recv completion\n");
    printf("Prefix: No.\n");
    printf("Suffix: No.\n");
    printf("Downsides: polling on receiver\n");
    printf("-----------------------------------------\n");
  }
    

  uint32_t ReqSize(Region& region) { return region.length; }

  uint64_t SendAsync(Region& region){
    uint64_t send_addr = (uint64_t)(void*)region.addr;
    uint32_t real_length = region.length;
 
    struct ibv_sge sge = {send_addr, real_length, region.lkey };
    ep->send(&sge, 1, next_id_);  

    return next_id_++;    
  }

  bool AckSentBytes(uint32_t bytes){
    return true;
  }

  void WaitSend(uint64_t id){ 
    while(last_wrid < id){
      TestSend(id);
    }
  }

  bool TestSend(uint64_t id){ 
    if(last_wrid >= id) return true;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS) printf("Failed send request %d \n",wcs[i].status);
    }
    ret = ibv_poll_cq(ep->qp->recv_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS) printf("Failed recv request %d \n",wcs[i].status);
      last_wrid += wcs[i].imm_data;
    }
    if(ret > 0) ep->post_empty_recvs((uint32_t)ret);

    if(last_wrid >= id) return true;
    return false;
  }
};

class SharedReceiveReceiver:  public ReceiveCommunicator {
  std::vector<VerbsEP*> &eps;
  Buffer* const local_buffer;
  const uint32_t lkey;
  const uint32_t size;
  const uint32_t qpn_offset;
  std::vector<uint32_t> acks;
 
public:
  SharedReceiveReceiver(std::vector<VerbsEP*> &eps, Buffer* local_buffer,  uint32_t size, uint32_t max_recv_size): 
    eps(eps), local_buffer(local_buffer), lkey(local_buffer->GetKey()), size(size), qpn_offset(eps[0]->qp->qp_num)
  {   

    //printf("qpns %u %u\n", eps[0]->qp->qp_num, eps[1]->qp->qp_num);
    uint32_t i =0;
    char* addr = local_buffer->GetReadPtr();
    char* end = addr + local_buffer->GetLength();
    while(addr+size<end && i < max_recv_size){
      struct ibv_sge sge = {(uint64_t)(void*)addr, size, lkey };
      eps[0]->post_recv(sge);
      addr+=size;
      i++;
    }
    eps[0]->trigget_post_recv();

    acks.resize(eps.size(), 0);

  };
  ~SharedReceiveReceiver( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Receive Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: N to 1.\n");
    printf("Used OPs: Yes - post recv\n");
    printf("Passive: Yes\n");
    printf("-----------------------------------------\n");
  }

  int Receive(std::vector<Region> &v){
    int c = 0;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(eps[0]->qp->recv_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].opcode == IBV_WC_RECV){
        uint32_t length = wcs[i].byte_len;
        uint32_t conid = (wcs[i].qp_num - qpn_offset)/2; // connection id.todo. /2 for debugging on one node

        char* whole_message = (char*)(void*)wcs[i].wr_id;
        v.push_back({conid, whole_message, (uint32_t)length, lkey }); 
        c++;
      }
    }
    return c;  
  }

  void FreeReceive(Region& region){
    local_buffer->Free(size);
    acks[region.context]++;
    struct ibv_sge sge = {(uint64_t)(void*)region.addr, size, lkey };
    eps[0]->post_recv(sge);
  }

  uint32_t GetFreedReceiveBytes(){
    eps[0]->trigget_post_recv();
    for(uint32_t i=0; i<eps.size(); i++){
      if(acks[i]>0){

        eps[i]->write_imm(NULL, 0, 0, 0, 0, acks[i]); 
        acks[i]=0;
      }
    }
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(eps[0]->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS){
        printf("Failed send request %d \n",wcs[i].status);
        exit(1);  
      } 
    }
    return local_buffer->GetFreedBytes();
  }
};



class CircularConnectionMailBox:  public SendCommunicator {
  VerbsEP* const ep;
  RemoteBuffer* const remote_buffer;
  const uint64_t rem_mailbox;
  const uint32_t rem_mailbox_rkey;

  uint64_t last_wrid = 0;
  uint64_t next_id_ = 1;

  struct ibv_send_wr wrs[2];
  struct ibv_sge sges[3];

  const bool with_sges;
  const uint64_t local_mem;
public:
  CircularConnectionMailBox(VerbsEP* ep, RemoteBuffer* remote_buffer, uint64_t rem_mailbox, uint32_t rem_mailbox_rkey,
			    uint64_t local_mem, uint32_t local_mem_lkey
			    ): 
			      ep(ep), remote_buffer(remote_buffer), rem_mailbox(rem_mailbox), rem_mailbox_rkey(rem_mailbox_rkey), with_sges(ep->GetMaxSendSge()>=2), local_mem(local_mem)
  {
    sges[0] = {local_mem,  sizeof(LEN_BYTE_T), local_mem_lkey }; // TODO. In theory can be subject to race conditions. Should use outstanding count
    sges[1] = {0, 0, 0 };
    sges[2] = {0, sizeof(MAGIC_BYTE_T), 0 };

    if(with_sges){
      wrs[0].sg_list = &sges[0];
      wrs[0].num_sge = 2;
      printf(">>>>>>>>>>>>>>>>>> I use sges implementation\n");
    }else{
      wrs[0].sg_list = &sges[1];
      wrs[0].num_sge = 1;
    }
    wrs[1].sg_list = &sges[2];
    wrs[1].num_sge = 1;

    wrs[0].opcode = IBV_WR_RDMA_WRITE;
    wrs[1].opcode = IBV_WR_RDMA_WRITE;

    wrs[0].send_flags = IBV_SEND_SIGNALED;   
    wrs[1].send_flags = IBV_SEND_INLINE;   
    wrs[0].wr.rdma.remote_addr = 0;
    wrs[1].wr.rdma.remote_addr = rem_mailbox;

    wrs[0].wr.rdma.rkey        = remote_buffer->GetKey();
    wrs[1].wr.rdma.rkey        = rem_mailbox_rkey;

    wrs[0].next = &wrs[1];
    wrs[1].next = NULL;
  };

  ~CircularConnectionMailBox( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Circular MailBox Sender\n");
    printf("Role: Sender\n");
    printf("Type: P2P.\n");
    printf("Used OPs: RDMA Write\n");
    printf("Way of informing the receiver: RDMA Write to remote mailbox message ID.\n");
    printf("Prefix: Yes - Message Length.\n");
    printf("Suffix: No.\n");
    if(with_sges) printf("Sges: Yes - 2 to send prefix\n");
    printf("Downsides: 2 writes per send message\n");
    printf("-----------------------------------------\n");
  }

  uint32_t ReqSize(Region& region) { return region.length + sizeof(LEN_BYTE_T); }

  uint64_t SendAsync(Region& region){
    struct ibv_send_wr *bad_wr;

    uint64_t send_addr = (uint64_t)(void*)region.addr;
    uint32_t real_length = region.length + sizeof(LEN_BYTE_T);
        
    uint64_t rem_addr = remote_buffer->GetWriteAddr(real_length);
    if(rem_addr == 0){
      printf("Failed to get mem\n");
      exit(1);
    }

    if(with_sges){
      *(volatile LEN_BYTE_T*)(void*)(local_mem) = (LEN_BYTE_T)region.length; 
      real_length-=sizeof(LEN_BYTE_T);
    }else{
      // we prepend length
      *(LEN_BYTE_T*)(region.addr - sizeof(LEN_BYTE_T)) = (LEN_BYTE_T)region.length; // write len at the begin
      send_addr-=sizeof(LEN_BYTE_T);
    }


    sges[1] = {send_addr, real_length, region.lkey };
    MAGIC_BYTE_T mail_box_val = (MAGIC_BYTE_T)next_id_; // we send message id
    sges[2] = { (uint64_t)(void*)&mail_box_val, sizeof(MAGIC_BYTE_T), 0 }; // 0 as send inline
    wrs[0].wr_id = next_id_;
    wrs[0].wr.rdma.remote_addr = rem_addr; 


    /*   printf("Send Async to %lx %u and to mailbox %lx %u\n", 
         wrs[0].wr.rdma.remote_addr, wrs[0].wr.rdma.rkey, wrs[1].wr.rdma.remote_addr, wrs[1].wr.rdma.rkey );

        printf("Send from %lx %u %u  %lx %u %u and to %lx %u %u\n", 
        wrs[0].sg_list[0].addr,wrs[0].sg_list[0].lkey, wrs[0].sg_list[0].length, 
        wrs[0].sg_list[1].addr,wrs[1].sg_list[1].lkey, wrs[0].sg_list[1].length, 
        wrs[1].sg_list->addr,wrs[1].sg_list->lkey, wrs[1].sg_list->length ); */

    if(ibv_post_send(ep->qp, wrs, &bad_wr )){
      printf("Failed to send %d\n",errno);
      exit(1);
    }

    return next_id_++;    
  }

  bool AckSentBytes(uint32_t bytes){
    //printf("akk %u bytes\n",bytes);
    return remote_buffer->FreeBytes(bytes);
  }

  void WaitSend(uint64_t id){ 
    while(last_wrid < id){
      TestSend(id);
    }
  }

  bool TestSend(uint64_t id){ 
    if(last_wrid >= id) return true;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS){
        printf("Failed request %d \n",wcs[i].status);
        exit(1);  
      }  
      last_wrid = wcs[i].wr_id;
    }
    if(last_wrid >= id) return true;
    return false;
  }
};



class CircularConnectionMagicbyte:  public SendCommunicator {
  VerbsEP* const ep;
  RemoteBuffer* const remote_buffer;
    
  uint64_t last_wrid = 0;
  uint64_t next_id_ = 1;


  struct ibv_send_wr wr;
  struct ibv_sge sges[3];

  const bool with_sges;
  const uint64_t local_mem;

public:
  CircularConnectionMagicbyte(VerbsEP* ep, RemoteBuffer* remote_buffer, uint64_t local_mem, uint32_t local_mem_lkey)
  : 
    ep(ep), remote_buffer(remote_buffer), with_sges(ep->GetMaxSendSge()>=3), local_mem(local_mem)
  {
    sges[0] = {local_mem,  sizeof(LEN_BYTE_T), local_mem_lkey }; // TODO. In theory can be subject to race conditions. Should use outstanding count
    sges[1] = {0, 0, 0 };
    sges[2] = {local_mem+8, sizeof(MAGIC_BYTE_T), local_mem_lkey };

    *(volatile MAGIC_BYTE_T*)(void*)(local_mem+8) = (MAGIC_BYTE_T)1; 

    if(with_sges){
      wr.sg_list = &sges[0];
      wr.num_sge = 3;
      printf(">>>>>>>>>>>>>>>>>> I use sges implementation\n");
    }else{
      wr.sg_list = &sges[1];
      wr.num_sge = 1;
    }
      
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;   
    
    wr.wr.rdma.remote_addr = 0;
    wr.wr.rdma.rkey        = remote_buffer->GetKey();
    wr.next = NULL;
  };
  ~CircularConnectionMagicbyte( ) {};


  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Circular MagicByte Sender\n");
    printf("Role: Sender\n");
    printf("Type: P2P.\n");
    printf("Used OPs: RDMA Write\n");
    printf("Way of informing the receiver: the receiver polls len and magic byte.\n");
    printf("Prefix: Yes - Message Length.\n");
    printf("Suffix: Yes - Magic byte.\n");
    if(with_sges) printf("Sges: Yes - 3 to send prefix and suffix\n");
    printf("Downsides: Requires zeroing on remote machine\n");
    printf("-----------------------------------------\n");
  }
    

  uint32_t ReqSize(Region& region) { return region.length + sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T); }

  uint64_t SendAsync(Region& region){
    struct ibv_send_wr *bad_wr;

    uint64_t send_addr = (uint64_t)(void*)region.addr;
    uint32_t real_length = region.length + sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T);
        
    uint64_t rem_addr = remote_buffer->GetWriteAddr(real_length);
    if(rem_addr == 0){
      printf("Failed to get mem\n");
      exit(1);
    } 

    if(with_sges){
      *(volatile LEN_BYTE_T*)(void*)(local_mem) = (LEN_BYTE_T)region.length; 
      real_length = region.length;
    }else{
      *(LEN_BYTE_T*)(region.addr - sizeof(LEN_BYTE_T)) = (LEN_BYTE_T)region.length; // write len at the begin
      *(MAGIC_BYTE_T*)(region.addr + region.length) = (MAGIC_BYTE_T)1; // write 1 at the end   
      send_addr-=sizeof(LEN_BYTE_T);
    }

    sges[1] = {send_addr, real_length, region.lkey };

    wr.wr_id = next_id_;
    wr.wr.rdma.remote_addr = rem_addr; 
 
    if(ibv_post_send(ep->qp, &wr, &bad_wr )){
      printf("Failed to send %d\n",errno);
      exit(1);
    }

    return next_id_++;    
  }

  bool AckSentBytes(uint32_t bytes){
    return remote_buffer->FreeBytes(bytes);
  }

  void WaitSend(uint64_t id){ 
    while(last_wrid < id){
      TestSend(id);
    }
  }

  bool TestSend(uint64_t id){ 
    if(last_wrid >= id) return true;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS){
        printf("Failed request %d \n",wcs[i].status);
        exit(1);  
      }  
      last_wrid = wcs[i].wr_id;
    }
    if(last_wrid >= id) return true;
    return false;
  }
};



class CircularConnectionNotify:  public SendCommunicator {
  VerbsEP* const ep;
  RemoteBuffer* const remote_buffer;
    
  uint64_t last_wrid = 0;
  uint64_t next_id_ = 1;

public:
  CircularConnectionNotify(VerbsEP* ep, RemoteBuffer* remote_buffer): ep(ep), remote_buffer(remote_buffer) {
    // empty
  };
  ~CircularConnectionNotify( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Circular Notify Sender\n");
    printf("Role: Sender\n");
    printf("Type: P2P.\n");
    printf("Used OPs: RDMA Write with IMM\n");
    printf("Way of informing the receiver: Immediate data\n");
    printf("Prefix: No.\n");
    printf("Suffix: No.\n");
    printf("Downsides: polling on receiver\n");
    printf("-----------------------------------------\n");
  }
    

  uint32_t ReqSize(Region& region) { return region.length; }

  uint64_t SendAsync(Region& region){
    uint64_t send_addr = (uint64_t)(void*)region.addr;
    uint32_t real_length = region.length;
        
    uint64_t rem_addr = remote_buffer->GetWriteAddr(real_length);
    if(rem_addr == 0){
      printf("Faield to get mem\n");
      exit(1);
    }
 
    struct ibv_sge sge = {send_addr, real_length, region.lkey };
    ep->write_imm(&sge, 1, remote_buffer->GetKey(), rem_addr, next_id_, 0);  

    return next_id_++;    
  }

  bool AckSentBytes(uint32_t bytes){
    return remote_buffer->FreeBytes(bytes);
  }

  void WaitSend(uint64_t id){ 
    while(last_wrid < id){
      TestSend(id);
    }
  }

  bool TestSend(uint64_t id){ 
    if(last_wrid >= id) return true;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS){
        printf("Failed request %d \n",wcs[i].status);
        exit(1);  
      }  
      last_wrid = wcs[i].wr_id;
    }
    if(last_wrid >= id) return true;
    return false;
  }
};


// TODO. I do not  need 1. i can reuse length!
class CircularConnectionReverse:  public SendCommunicator {
  VerbsEP* const ep;
  RemoteBuffer* const remote_buffer;
    
  uint64_t last_wrid = 0;
  uint64_t next_id_ = 1;


  struct ibv_send_wr wr;
  struct ibv_sge sges[3];

  const bool with_sges;
  const uint64_t local_mem;

public:
  CircularConnectionReverse(VerbsEP* ep, RemoteBuffer* remote_buffer, uint64_t local_mem, uint32_t local_mem_lkey)
  : ep(ep), remote_buffer(remote_buffer) , with_sges(ep->GetMaxSendSge()>=3), local_mem(local_mem)
  {
    sges[0] = {local_mem,  sizeof(MAGIC_BYTE_T), local_mem_lkey }; // TODO. In theory can be subject to race conditions. Should use outstanding count
    sges[1] = {0, 0, 0 };
    sges[2] = {local_mem+8, sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T), local_mem_lkey };

    *(volatile MAGIC_BYTE_T*)(void*)(local_mem) = (MAGIC_BYTE_T)0; 
    *(volatile MAGIC_BYTE_T*)(void*)(local_mem+8) = (LEN_BYTE_T)0; 
    *(volatile MAGIC_BYTE_T*)(void*)(local_mem+16) = (MAGIC_BYTE_T)1; 

    if(with_sges){
      wr.sg_list = &sges[0];
      wr.num_sge = 3;
      printf(">>>>>>>>>>>>>>>>>> I use sges implementation\n");
    }else{
      wr.sg_list = &sges[1];
      wr.num_sge = 1;
    }
      
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;   
    
    wr.wr.rdma.remote_addr = 0;
    wr.wr.rdma.rkey        = remote_buffer->GetKey();
    wr.next = NULL;
    


    remote_buffer->GetWriteAddr(sizeof(MAGIC_BYTE_T)); // it is for the first magicbyte
  };

  ~CircularConnectionReverse( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Circular Reverse Sender\n");
    printf("Role: Sender\n");
    printf("Type: P2P.\n");
    printf("Used OPs: RDMA Write\n");
    printf("Way of informing the receiver: MagicByte\n");
    printf("Prefix: Yes - zero magic byte \n");
    printf("Suffix: Yes - and length magic byte.\n");
    if(with_sges) printf("Sges: Yes - 3 to send prefix and suffix\n");
    printf("Downsides: no\n");
    printf("-----------------------------------------\n");
  }
    

  uint32_t ReqSize(Region& region) { return region.length + sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T); }

  uint64_t SendAsync(Region& region){
    struct ibv_send_wr *bad_wr;

    uint64_t send_addr = (uint64_t)(void*)region.addr;
    uint32_t real_length = region.length + sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T); // that is how many bytes we send
        
    uint64_t rem_addr = remote_buffer->GetWriteAddr(real_length);
    if(rem_addr == 0){
      printf("Failed to get mem\n");
      exit(1);
    }

    if(with_sges){
      *(volatile LEN_BYTE_T*)(void*)(local_mem+8) = (LEN_BYTE_T)region.length; 
      real_length = region.length;
    }else{
      *(MAGIC_BYTE_T*)(region.addr - sizeof(MAGIC_BYTE_T)) = (MAGIC_BYTE_T)0;    
      *(LEN_BYTE_T*)(region.addr + region.length) = (LEN_BYTE_T)region.length; // write len at the end
      *(MAGIC_BYTE_T*)(region.addr + region.length + sizeof(LEN_BYTE_T)) = (MAGIC_BYTE_T)1; // write 1 after the end  

      send_addr-=sizeof(MAGIC_BYTE_T);
      real_length+=sizeof(MAGIC_BYTE_T); // we do not count one magic byte as a message.
    }

    //rem_addr points to the magicbyte with 0.

    // printf("Zero written to %lx. One written to %lx\n", rem_addr, rem_addr + real_length - sizeof(MAGIC_BYTE_T));


    sges[1] = {send_addr, real_length, region.lkey };

    wr.wr_id = next_id_;
    wr.wr.rdma.remote_addr = rem_addr; 
 
    if(ibv_post_send(ep->qp, &wr, &bad_wr )){
      printf("Failed to send %d\n",errno);
      exit(1);
    }

 

    return next_id_++;    
  }

  bool AckSentBytes(uint32_t bytes){
    return remote_buffer->FreeBytes(bytes);
  }

  void WaitSend(uint64_t id){ 
    while(last_wrid < id){
      TestSend(id);
    }
  }

  bool TestSend(uint64_t id){ 
    if(last_wrid >= id) return true;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS){
        printf("Failed request %d \n",wcs[i].status);
        exit(1);  
      }  
      last_wrid = wcs[i].wr_id;
    }
    if(last_wrid >= id) return true;
    return false;
  }
};

 

class CircularMailBoxReceiver:  public ReceiveCommunicator {
  Buffer* const local_buffer;
  const uint64_t mailbox_addr;
  const uint32_t mailbox_rkey;


  MAGIC_BYTE_T processed = 0;
  volatile MAGIC_BYTE_T* const pending;
 
public:
  CircularMailBoxReceiver(Buffer* local_buffer, uint64_t mailbox_addr, uint32_t mailbox_rkey): 
    local_buffer(local_buffer), mailbox_addr(mailbox_addr), mailbox_rkey(mailbox_rkey), pending((volatile MAGIC_BYTE_T*)(void*)mailbox_addr)
  {
    *pending = (MAGIC_BYTE_T)0;
  };
  ~CircularMailBoxReceiver( ) {};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Circular MailBox Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: P2P.\n");
    printf("Used OPs: No - memory checking\n");
    printf("Passive: No\n");
    printf("-----------------------------------------\n");
  }
    

  int Receive(std::vector<Region> &v){
    MAGIC_BYTE_T pending_copy = *pending;
    int c = 0;
    while(processed<pending_copy){
      //    printf("Get a new mailbox %lu %lu \n",processed,pending_copy);
      LEN_BYTE_T length = *(LEN_BYTE_T*)local_buffer->GetReadPtr();
      //     printf("length %lu\n",length);
      char* whole_message = local_buffer->Read(((uint32_t)length) + sizeof(LEN_BYTE_T));
      uint32_t lkey = 0; // I do not need this
      v.push_back({0, whole_message+sizeof(LEN_BYTE_T), (uint32_t)length, lkey });
      processed++;
      c++;
    }

    return c;
  }

  void FreeReceive(Region& region){
    local_buffer->Free(region.length + sizeof(LEN_BYTE_T));
  }

  uint32_t GetFreedReceiveBytes(){
    return local_buffer->GetFreedBytes();
  }
};

class CircularMagicbyteReceiver:  public ReceiveCommunicator {
  Buffer* const local_buffer;
 
public:
  CircularMagicbyteReceiver(Buffer* local_buffer): 
    local_buffer(local_buffer)
  {
        
  };
  ~CircularMagicbyteReceiver( ) {};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Circular Magicbyte Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: P2P.\n");
    printf("Used OPs: No - memory checking\n");
    printf("Passive: No\n");
    printf("-----------------------------------------\n");
  }

  int Receive(std::vector<Region> &v){
    LEN_BYTE_T length = *(volatile LEN_BYTE_T*)local_buffer->GetReadPtr();
    int c = 0;
    while(length != 0){ // we received a message
      //   printf("Get a new byte %lu \n",length);
      *(volatile LEN_BYTE_T*)local_buffer->GetReadPtr() = (LEN_BYTE_T)0; // zero the length. otherwise we can read length of an old message.
      char* whole_message = local_buffer->Read((uint32_t)length + sizeof(LEN_BYTE_T)+ sizeof(MAGIC_BYTE_T));
      uint32_t lkey = 0; // I do not need this
      v.push_back({0, whole_message+sizeof(LEN_BYTE_T), (uint32_t)length, lkey }); 
      MAGIC_BYTE_T completion = 0;
      do{
        completion = *(volatile MAGIC_BYTE_T*)(whole_message+ sizeof(LEN_BYTE_T)+length );    
      }while(completion ==0);
      c++;
      //    printf("done %lu \n",length);
      length = *(volatile LEN_BYTE_T*)local_buffer->GetReadPtr();
    }
    return c;
  }

  void FreeReceive(Region& region){
    local_buffer->Free(region.length + sizeof(LEN_BYTE_T)+ sizeof(MAGIC_BYTE_T));
  }

  uint32_t GetFreedReceiveBytes(){
    return local_buffer->GetFreedBytes();
  }
};

class CircularNotifyReceiver:  public ReceiveCommunicator {
  VerbsEP* const ep;
  Buffer* const local_buffer;

 
 
public:
  CircularNotifyReceiver(VerbsEP* ep, Buffer* local_buffer): 
    ep(ep), local_buffer(local_buffer) 
  {
        
  };
  ~CircularNotifyReceiver( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Circular Magicbyte Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: P2P.\n");
    printf("Used OPs: Yes - post recv\n");
    printf("Passive: Yes\n");
    printf("-----------------------------------------\n");
  }

  int Receive(std::vector<Region> &v){
    int c = 0;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->recv_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM){
        uint32_t length = wcs[i].byte_len;
        char* whole_message = local_buffer->Read(length);
        uint32_t lkey = 0; // I do not need this
        v.push_back({0, whole_message, (uint32_t)length, lkey }); 
        c++;
      }
    }
    if(ret > 0) ep->post_empty_recvs((uint32_t)ret);
    return c;  
  }

  void FreeReceive(Region& region){
    local_buffer->Free(region.length);
  }

  uint32_t GetFreedReceiveBytes(){
    return local_buffer->GetFreedBytes();
  }
};



class CircularReverseReceiver:  public ReceiveCommunicator {
  Buffer* const local_buffer;
public:
  CircularReverseReceiver(Buffer* local_buffer): 
    local_buffer(local_buffer)
  {
    *(volatile MAGIC_BYTE_T*)local_buffer->Read(sizeof(MAGIC_BYTE_T)) = (MAGIC_BYTE_T)0;
  };
  ~CircularReverseReceiver() {
    
  };
  


  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Circular Reverse Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: P2P.\n");
    printf("Used OPs: No - memory checking\n");
    printf("Passive: No\n");
    printf("-----------------------------------------\n");
  }

  int Receive(std::vector<Region> &v){
    MAGIC_BYTE_T completion = *(volatile MAGIC_BYTE_T*)local_buffer->GetReadPtr();
    int c = 0;

    while(completion){ // we received a message
      LEN_BYTE_T length = *(volatile LEN_BYTE_T*)local_buffer->Read(sizeof(LEN_BYTE_T));
      //printf("Get a new len %lu \n",length);
 
      char* whole_message = local_buffer->Read((uint32_t)length + sizeof(MAGIC_BYTE_T)); //we read the byte with zero. 
      uint32_t lkey = 0; // I do not need this
      v.push_back({0, whole_message+sizeof(MAGIC_BYTE_T), (uint32_t)length, lkey }); 
      c++;
      //   printf("done %lu \n",length);
      completion = *(volatile MAGIC_BYTE_T*)local_buffer->GetReadPtr();
    }
    return c;
  }

  void FreeReceive(Region& region){
    local_buffer->Free(region.length + sizeof(LEN_BYTE_T)+ sizeof(MAGIC_BYTE_T));
  }

  uint32_t GetFreedReceiveBytes(){
    return local_buffer->GetFreedBytes();
  }
};

 


/////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////
/* shared connections with atomics*/

// it uses read to fetch offset. It is called magic byte with zeroing.
class SharedCircularConnectionMailbox:  public SendCommunicator {
  const uint8_t total_faa_slots = 16; // usually we cannot have more than 16 outstanding atomics
 
  VerbsEP* const ep;
  MagicRemoteBuffer* const remote_buffer;

  const uint64_t rem_head;
  const uint32_t rem_head_rkey;

  const uint64_t rem_win;
  const uint32_t rem_win_rkey;
 
  uint64_t last_wrid = 0;
  uint64_t next_id_ = 1;

  struct ibv_send_wr wrs[3];
  struct ibv_sge sges[5];


  const uint64_t local_mem;  // read here and faa here. should be at least 16 bytes
  const uint32_t local_mem_lkey;

  std::vector<uint8_t> free_local_slots; // for faas.


  volatile uint64_t* const pending_val;
  bool has_pending_read = false; // we have at most one outstanding read
  std::list<std::pair<uint64_t,Region*>> delayed;

  const bool with_sges;
public:
  SharedCircularConnectionMailbox(VerbsEP* ep, MagicRemoteBuffer* remote_buffer, uint64_t rem_head, uint32_t rem_head_rkey, 
				  uint64_t rem_win, uint32_t rem_win_rkey, uint64_t local_mem, uint32_t local_mem_lkey): 
				    ep(ep), remote_buffer(remote_buffer), rem_head(rem_head), rem_head_rkey(rem_head_rkey), rem_win(rem_win), rem_win_rkey(rem_win_rkey), 
				    local_mem(local_mem), local_mem_lkey(local_mem_lkey),pending_val( (volatile uint64_t*)(void*)(local_mem + (total_faa_slots+1)*8 )), with_sges(ep->GetMaxSendSge()>=3)
  {
      
    for(uint8_t i=0; i< total_faa_slots; i++){ // local faa slots to have parallelization
      free_local_slots.push_back(i);
    } 

    sges[0] = {local_mem, 8, local_mem_lkey }; // fetch here
    sges[1] = {local_mem + (total_faa_slots+1)*8, 8, local_mem_lkey }; // always read here

    wrs[0].sg_list = &sges[0];
    wrs[1].sg_list = &sges[1];

    wrs[0].num_sge = 1;
    wrs[1].num_sge = 1;

    wrs[0].opcode = IBV_WR_ATOMIC_FETCH_AND_ADD; // it may target device memory!
    wrs[1].opcode = IBV_WR_RDMA_READ; // To fetch head.  // it may target device memory!

    wrs[0].send_flags = IBV_SEND_SIGNALED;   
    wrs[1].send_flags = IBV_SEND_SIGNALED;   

    wrs[0].wr.atomic.remote_addr = rem_win;
    wrs[1].wr.rdma.remote_addr = rem_head;

    wrs[0].wr.atomic.rkey        = rem_win_rkey;
    wrs[1].wr.rdma.rkey        = rem_head_rkey;

    wrs[0].next = &wrs[1];
    wrs[1].next = NULL;

    printf("We fetch from %lx %u\n", rem_win,rem_win_rkey);
    printf("We read head from %lx\n", rem_head);

    if(with_sges){
      wrs[2].sg_list = &sges[2];
      wrs[2].num_sge = 3;
      printf(">>>>>>>>>>>>>>>>>> I use sges implementation\n");
    }else{
      wrs[2].sg_list = &sges[3];
      wrs[2].num_sge = 1;
    }

    sges[2] = {local_mem + (total_faa_slots+2)*8, 8, local_mem_lkey }; // fetch here
    sges[4] = {local_mem + (total_faa_slots+3)*8, 8, local_mem_lkey }; // always read here
      
    *(volatile MAGIC_BYTE_T*)(void*)(local_mem+ (total_faa_slots+3)*8) = (MAGIC_BYTE_T)1; 

    wrs[2].opcode = IBV_WR_RDMA_WRITE; // TODO. SHould we try target dev memory here?   
    wrs[2].send_flags = IBV_SEND_SIGNALED;   
    wrs[2].wr.rdma.remote_addr = 0; 
    wrs[2].wr.rdma.rkey        = remote_buffer->GetKey();
    wrs[2].next = NULL;

  };

  ~SharedCircularConnectionMailbox( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Circular Mailbox Sender\n");
    printf("Role: Sender\n");
    printf("Type: Many to One.\n");
    printf("Used OPs: RDMA Write, RDMA Atomic, RDMA Read\n");
    printf("Way of informing the receiver: MagicByte\n");
    printf("Prefix: Yes - length\n");
    printf("Suffix: Yes - magic byte.\n");
    if(with_sges) printf("Sges: Yes - 3 to send prefix and suffix\n");
    printf("Downsides: One sender can block others\n");
    printf("Comment: One should call test send to fetch head and trigger send\n");
    printf("-----------------------------------------\n");

    PrintDebug();
  }


  void PrintDebug(){
    printf("delayed: %lu\n",delayed.size());
    printf("free_local_slots: %lu\n",free_local_slots.size());
    printf("has_pending_read: %s\n",has_pending_read ? "yes" : "no");
    printf("Head %lu\n", (*(pending_val)));
  }

  uint32_t ReqSize(Region& region) { return region.length + sizeof(LEN_BYTE_T)+ sizeof(MAGIC_BYTE_T); }

  uint64_t SendAsync(Region& region){
    struct ibv_send_wr *bad_wr;
    region.context = next_id_;

    if(free_local_slots.size() == 0){
      printf("No slots\n");
      exit(1);
    }
    uint64_t slot = free_local_slots.back(); // it is 8 bit
    free_local_slots.pop_back();

    void* ptr = (void*)&region; // get pointer to the original object

    uint64_t wrid = (slot << 56) + (uint64_t)ptr;

    sges[0].addr = local_mem + slot*8;

    wrs[0].wr_id = wrid;
    wrs[0].wr.atomic.compare_add = region.length + sizeof(LEN_BYTE_T)+ sizeof(MAGIC_BYTE_T);
    wrs[1].wr_id = 0; // zero for reads

    if(has_pending_read) { // then we only do FAA
      wrs[0].next = NULL;
    } 

    has_pending_read = true;
    //   printf("post _send async with add %u and id %lu\n", wrs[0].opcode , wrs[0].wr_id );
    if(ibv_post_send(ep->qp, &wrs[0], &bad_wr )){
      printf("Failed to send\n");
      exit(1);
    }

    wrs[0].next = &wrs[1];
 
    return next_id_++;    
  }

  bool AckSentBytes(uint32_t bytes){ // does nothing
    return true;
  }

  void WaitSend(uint64_t id){ 
    while(last_wrid < id){
      TestSend(id);
    }
  }
private:
  inline void send(uint64_t fetched, Region* region){
    struct ibv_send_wr *bad_wr;
    uint64_t next_id_ = region->context;

    uint32_t offset = remote_buffer->GetOffset(remote_buffer->GetBaseAddr() + fetched);

    //   printf("Write to base %lx and offset %u, %lu \n",remote_buffer->GetBaseAddr(), offset, fetched);
    uint64_t rem_addr = remote_buffer->GetBaseAddr() + offset;

    uint64_t send_addr = ((uint64_t)(void*)region->addr)- sizeof(LEN_BYTE_T);
    uint32_t real_length = region->length + sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T);

       
    if(with_sges){
      *(volatile LEN_BYTE_T*)(void*)(local_mem + ((total_faa_slots+2)*8) ) = (LEN_BYTE_T)region->length; 
      real_length = region->length;
      send_addr = ((uint64_t)(void*)region->addr);
    }else{
      *(volatile LEN_BYTE_T*)(region->addr - sizeof(LEN_BYTE_T)) = (LEN_BYTE_T)region->length; // write len at the begin
      *(volatile MAGIC_BYTE_T*)(region->addr + region->length) = (MAGIC_BYTE_T)1; // write 1 at the end   
    }
 
    sges[3] = {send_addr, real_length, region->lkey };

    wrs[2].wr_id = next_id_;
    wrs[2].wr.rdma.remote_addr = rem_addr; 
    //    printf("Write %u\n",real_length);
    //    printf("Write message to %lx\n",rem_addr);

    if(ibv_post_send(ep->qp, &wrs[2], &bad_wr )){
      printf("Failed to send\n");
      exit(1);
    }

  }

  inline  bool try_send_delayed(){
    if(delayed.empty()) return true;

    uint64_t head = (*(pending_val)); 
    while(!delayed.empty()){

      auto val = delayed.front();
      uint32_t real_length = val.second->length + sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T);
      if(val.first + real_length > head){
        break;
      }
            
      send(val.first,val.second);
      delayed.pop_front();
    }
    return delayed.empty();
  }
public:
  bool TestSend(uint64_t id){ 
    if(last_wrid >= id) return true;
        
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS){
        printf("Failed request %d %d\n",wcs[i].status,wcs[i].opcode);
        exit(1);
      }
      switch(wcs[i].opcode){
      case IBV_WC_RDMA_READ:
        //  printf("completed read\n");
        has_pending_read = false;
        break;
      case IBV_WC_FETCH_ADD: {
        //  printf("completed fetch\n");
        uint8_t slot = (uint8_t)(wcs[i].wr_id >> 56);
        free_local_slots.push_back(slot);
        Region* region = (Region*)(void*)(wcs[i].wr_id & 0xFFFFFFFFFFFFFF);
        uint64_t fetched = *(volatile uint64_t *)(void*)(local_mem+slot*8);
        //    printf("fetched %lx \n",fetched);
        uint64_t head = (*(pending_val)); 
        uint32_t real_length = region->length + sizeof(LEN_BYTE_T) + sizeof(MAGIC_BYTE_T);
        if(try_send_delayed() && fetched + real_length <=head){
          //can send.
          send(fetched, region);
        }else{
          //cannot send.
          //           printf("cannot send because of head\n");
          delayed.push_back({fetched,region});

        }
      }
        break;
      default:
        //      printf("completed op %lu\n",wcs[i].wr_id);
        // here is completed write
        last_wrid = wcs[i].wr_id;
      }         
    }
    if(!delayed.empty() && !has_pending_read){
      has_pending_read = true;
      struct ibv_send_wr *bad_wr;
      if(ibv_post_send(ep->qp, &wrs[1], &bad_wr )){
        printf("Failed to send\n");
        exit(1);
      }
    }

    try_send_delayed();

    if(last_wrid >= id) return true;
    return false;
  }
};




// for exp 6
class SharedCircularConnectionNotify:  public SendCommunicator {
  const uint8_t total_faa_slots = 16; // usually we cannot have more than 16 outstanding atomics


  VerbsEP* const ep;
  MagicRemoteBuffer* const remote_buffer;

  const uint64_t rem_head;
  const uint32_t rem_head_rkey;

  const uint64_t rem_win;
  const uint32_t rem_win_rkey;



  uint64_t last_wrid = 0;
  uint64_t next_id_ = 1;

  struct ibv_send_wr wrs[2];
  struct ibv_sge sges[2];


  const uint64_t local_mem;  // read here and faa here
  const uint32_t local_mem_lkey;

  std::vector<uint8_t> free_local_slots; // for faas.


  volatile uint64_t* const pending_val;
  bool has_pending_read = false; // we have at most one outstanding read

  std::list<std::pair<uint64_t,Region*>> delayed;

public:
  SharedCircularConnectionNotify(VerbsEP* ep, MagicRemoteBuffer* remote_buffer, uint64_t rem_head, uint32_t rem_head_rkey, 
				 uint64_t rem_win, uint32_t rem_win_rkey, uint64_t local_mem, uint32_t local_mem_lkey): 
				   ep(ep), remote_buffer(remote_buffer), rem_head(rem_head), rem_head_rkey(rem_head_rkey), rem_win(rem_win), rem_win_rkey(rem_win_rkey), 
				   local_mem(local_mem), local_mem_lkey(local_mem_lkey),pending_val( (volatile uint64_t*)(void*)(local_mem + (total_faa_slots+1)*8 ))
  {
      
    for(uint8_t i=0; i< total_faa_slots; i++){
      free_local_slots.push_back(i);
    } 

    sges[0] = {local_mem, 8, local_mem_lkey }; // fetch here
    sges[1] = {local_mem + (total_faa_slots+1)*8, 8, local_mem_lkey }; // always read here

    wrs[0].sg_list = &sges[0];
    wrs[1].sg_list = &sges[1];

    wrs[0].num_sge = 1;
    wrs[1].num_sge = 1;

    wrs[0].opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;         // can targer dev memory
    wrs[1].opcode = IBV_WR_RDMA_READ;                  // To fetch head. TODO. SHould we try target dev memory here?
 
    wrs[0].send_flags = IBV_SEND_SIGNALED;   
    wrs[1].send_flags = IBV_SEND_SIGNALED;   

    wrs[0].wr.atomic.remote_addr = rem_win;
    wrs[1].wr.rdma.remote_addr = rem_head;

    wrs[0].wr.atomic.rkey        = rem_win_rkey;
    wrs[1].wr.rdma.rkey        = rem_head_rkey;

    wrs[0].next = &wrs[1];
    wrs[1].next = NULL;
  };

  ~SharedCircularConnectionNotify( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Circular Notify Sender\n");
    printf("Role: Sender\n");
    printf("Type: Many to One.\n");
    printf("Used OPs: RDMA Write with IMM, RDMA Atomic, RDMA Read to fetch head\n");
    printf("Way of informing the receiver: offset in the IMM data\n");
    printf("Prefix: No\n");
    printf("Suffix: No\n");
    printf("Downsides: Use of IMM\n");
    printf("Comment: I can fetch multiple faas to have many outstanding. Head is shared.\n");
    printf("-----------------------------------------\n");

    PrintDebug();
  }

  void PrintDebug(){
    printf("delayed: %lu\n",delayed.size());
    printf("free_local_slots: %lu\n",free_local_slots.size());
    printf("has_pending_read: %s\n",has_pending_read ? "yes" : "no");
    printf("Head %lu\n", (*(pending_val)));
  }


  uint32_t ReqSize(Region& region) { return region.length; }

  uint64_t SendAsync(Region& region){
    struct ibv_send_wr *bad_wr;
    region.context = next_id_;

    if(free_local_slots.size() == 0){
      printf("No slots\n");
      exit(1);
    }

    uint64_t slot = free_local_slots.back(); // it is 8 bit
    free_local_slots.pop_back();

    void* ptr = (void*)&region; // get pointer to the original object

    uint64_t wrid = (slot << 56) + (uint64_t)ptr;

    sges[0].addr = local_mem + slot*8;

    wrs[0].wr_id = wrid;
    wrs[0].wr.atomic.compare_add = region.length  ;
    wrs[1].wr_id = 0; // zero for reads

    if(has_pending_read) {
      wrs[0].next = NULL;
    }
    has_pending_read = true;
 
    if(ibv_post_send(ep->qp, wrs, &bad_wr )){
      printf("Failed to send\n");
      exit(1);
    }

    wrs[0].next = &wrs[1];
 
    return next_id_++;    
  }

  bool AckSentBytes(uint32_t bytes){
    return true;
  }

  void WaitSend(uint64_t id){ 
    while(last_wrid < id){
      TestSend(id);
    }
  }
private:
  inline void send(uint64_t fetched, Region* region){
    uint64_t next_id_ = region->context;
    uint32_t offset = remote_buffer->GetOffset(remote_buffer->GetBaseAddr() + fetched);
    uint64_t rem_addr = remote_buffer->GetBaseAddr() + offset;

    uint64_t send_addr = ((uint64_t)(void*)region->addr);
    uint32_t real_length = region->length ;
    struct ibv_sge sge = {send_addr, real_length, region->lkey };
    if(ep->write_imm(&sge, 1, remote_buffer->GetKey(), rem_addr, next_id_, offset)){
      printf("Failed to send\n");
      exit(1);
    } 
  }

  inline  bool try_send_delayed(){
    if(delayed.empty()) return true;

    uint64_t head = (*(pending_val)); 
    while(!delayed.empty()){

      auto val = delayed.front();
      uint32_t real_length = val.second->length;
      if(val.first + real_length > head){
        break;
      }
            
      send(val.first,val.second);
      delayed.pop_front();
    }
    return delayed.empty();
  }
public:

  bool TestSend(uint64_t id){ 
    if(last_wrid >= id) return true;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS){
        printf("Failed request %d %d\n",wcs[i].status,wcs[i].opcode);
        exit(1);
      }
      switch(wcs[i].opcode){
      case IBV_WC_RDMA_READ:
        has_pending_read = false;
        break;
      case IBV_WC_FETCH_ADD: {
        uint8_t slot = (uint8_t)(wcs[i].wr_id >> 56);
        free_local_slots.push_back(slot);
        Region* region = (Region*)(void*)(wcs[i].wr_id & 0xFFFFFFFFFFFFFF);
        uint64_t fetched = *(volatile uint64_t *)(void*)(local_mem+slot*8);
        //printf("fetched %lx \n",fetched);
        uint64_t head = (*(pending_val)); 
        uint32_t real_length = region->length ;
        if(try_send_delayed() && fetched + real_length <=head){
          //can send.
          send(fetched,region);
        }else{
          //cannot send.
          //     printf("cannot send because of head\n");
          delayed.push_back({fetched,region});
        }
      }
        break;
      default:
        // here is completed write
        last_wrid = wcs[i].wr_id;
      }         
    }
    if(!delayed.empty() && !has_pending_read){
      has_pending_read = true;
      struct ibv_send_wr *bad_wr;
      if(ibv_post_send(ep->qp, &wrs[1], &bad_wr)){
        printf("Failed to send\n");
        exit(1);
      }
    }
    try_send_delayed();

    if(last_wrid >= id) return true;
    return false;
  }
};



class SharedCircularMagicbyteReceiver:  public ReceiveCommunicator {
  MagicRingBuffer* const local_buffer;

  volatile uint64_t* const head;
  volatile uint64_t* const faa;
    
  struct ibv_dm * const dm;
  uint64_t lhead=0;
public:
  SharedCircularMagicbyteReceiver(MagicRingBuffer* local_buffer,void* mailbox, void* faaaddr,struct ibv_dm *dm): 
    local_buffer(local_buffer), head((volatile uint64_t* )mailbox), faa((volatile uint64_t* )faaaddr), dm(dm)
  {   
    lhead = local_buffer->Free(0); // it will set the value of buffer length.
    if(lhead == 0){
      printf("head cannot be zero at the beginning!\n");
      exit(1);
    } 
    if(head!=NULL){
      *head = lhead;
    }else{
      if(ibv_memcpy_to_dm(dm, 0, &lhead, sizeof(lhead))){
        printf("failed to zero dev memory\n");
      }
    }
        
    if(faa!=NULL){
      *faa = 0;  
    } 

    printf("Head value: %lu\n", lhead);
    printf("Clients fetch from %p\n", faa);
    printf("I write my head progress here: %p\n", head);
  };  
  ~SharedCircularMagicbyteReceiver( ) {};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Circular MagicByte Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: Many to One.\n");
    printf("Used OPs: No - memory checking\n");
    printf("Passive: No\n");
    printf("-----------------------------------------\n");
  }

  int Receive(std::vector<Region> &v){
    LEN_BYTE_T length = *(volatile LEN_BYTE_T*)local_buffer->GetReadPtr();
    int c = 0;
    while(length != 0){ // we received a message
      //  printf("Get a new byte %lu \n",length);
      *(volatile LEN_BYTE_T*)local_buffer->GetReadPtr() = (LEN_BYTE_T)0; // zero the length. otherwise we can read length of an old message.
      char* whole_message = local_buffer->Read((uint32_t)length + sizeof(LEN_BYTE_T)+ sizeof(MAGIC_BYTE_T));
      uint32_t lkey = 0; // I do not need this
      v.push_back({0, whole_message+sizeof(LEN_BYTE_T), (uint32_t)length, lkey }); 
      MAGIC_BYTE_T completion = 0;
      do{ 
        completion = *(volatile MAGIC_BYTE_T*)(whole_message+ sizeof(LEN_BYTE_T)+length );    
      }while(completion ==0);
      c++;
      // printf("done %lu \n",length);
      length = *(volatile LEN_BYTE_T*)local_buffer->GetReadPtr();
    }
    return c;
  }

  void FreeReceive(Region& region){
    // printf("free receive \n");
    lhead = local_buffer->Free(region.length + sizeof(LEN_BYTE_T)+ sizeof(MAGIC_BYTE_T));
    if(head!=NULL){
      //     printf("write %lu \n",lhead);
      *head = lhead;
    }else{
      if(ibv_memcpy_to_dm(dm, 0, &lhead, sizeof(lhead))){
        printf("failed to zero dev memory\n");
        exit(1);
      }
    }
  }

  uint32_t GetFreedReceiveBytes(){
         
    return local_buffer->GetFreedBytes();; 
  }
};

class SharedCircularNotifyReceiver:  public ReceiveCommunicator {
  struct ibv_cq* const recv_cq; // shared for all EPS
  std::vector<VerbsEP*> &eps;
  MagicRingBuffer* const local_buffer;

  volatile uint64_t* const head; // is rdma accessible. 
  volatile uint64_t* const faa;
    
  struct ibv_dm * const dm;
  uint64_t lhead=0;
public:
  SharedCircularNotifyReceiver(std::vector<VerbsEP*> &eps, MagicRingBuffer* local_buffer, void* mailbox, void* faaaddr, struct ibv_dm *dm): 
    recv_cq(eps[0]->qp->recv_cq),eps(eps), local_buffer(local_buffer), head((volatile uint64_t* )mailbox), faa((volatile uint64_t* )faaaddr), dm(dm)
  {
    lhead = local_buffer->Free(0); // it will set the value of buffer length.
    if(lhead == 0){
      printf("head cannot be zero at the beginning!\n");
      exit(1);
    } 
    if(head!=NULL){
      *head = lhead;
    }else{
      if(ibv_memcpy_to_dm(dm, 0, &lhead, sizeof(lhead))){
        printf("failed to zero dev memory\n");
        exit(1);
      }
    }
    if(faa!=NULL) *faa = 0;
        
    printf("Head value: %lu\n", lhead);
    printf("Clients fetch from %p\n", faa);
    printf("I write my head progress here: %p\n", head);
  };
    
  ~SharedCircularNotifyReceiver( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Circular Notify Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: Many to One.\n");
    printf("Used OPs: Yes - post recv\n");
    printf("Passive: yes\n");
    printf("My progress is shared via writing to local head\n");
    printf("-----------------------------------------\n");
  }

  int Receive(std::vector<Region> &v){
    int c = 0;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(recv_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM){
        //   printf("Got imm from %lu\n",wcs[i].wr_id);
        uint32_t length = wcs[i].byte_len;
        uint32_t offset = wcs[i].imm_data;
        char* whole_message = local_buffer->GetReadPtr(offset); 
        uint32_t lkey = 0; // I do not need this
        v.push_back({0, whole_message, (uint32_t)length, lkey }); 
        c++;
        eps[wcs[i].wr_id]->post_empty_recvs(1); // if shared it will use the first eps.
      }
    }
    return c;  
  }

  void FreeReceive(Region& region){
    uint64_t new_head = local_buffer->FreeOrdered(region.addr,region.length); 
    if(new_head == lhead) return;

    lhead = new_head;
    if(head!=NULL){
      *head = lhead;
    }else{
      if(ibv_memcpy_to_dm(dm, 0, &lhead, sizeof(lhead))){
        printf("failed to zero dev memory\n");
        exit(1);
      }
    }
  }

  uint32_t GetFreedReceiveBytes(){
    return local_buffer->GetFreedBytes();
  }
};


 

/////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////

/* read based things. sender can have many clients */

class ReadCircularConnectionNotify:  public SendCommunicator {
  std::vector<VerbsEP*> &eps;
  MagicRemoteBuffer* const local_buffer;

    
  uint64_t local_tail = 0;
  uint64_t replicated_tail = 0;

  volatile uint64_t* const tails;
  const uint32_t num_of_clients;

  std::list<Region*> delayed;

  uint32_t can_send;

public:
  ReadCircularConnectionNotify(std::vector<VerbsEP*> &eps, MagicRemoteBuffer* local_buffer, uint64_t tail_ptr): 
    eps(eps), local_buffer(local_buffer),tails((volatile uint64_t*)(void*)tail_ptr),num_of_clients(eps.size())
  {   
    for(uint32_t i=0;i<num_of_clients;i++){
      tails[i] =0 ;
    }
    can_send = 16 * num_of_clients;
  };
  ~ReadCircularConnectionNotify( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Read Circular Notify Sender\n");
    printf("Role: Sender\n");
    printf("Type: One to Many.\n");
    printf("Used OPs: RDMA empty (IMM data via Send or write)\n");
    printf("Way of informing the receiver: Imm data\n");
    printf("Prefix: No.\n");
    printf("Suffix: No.\n");
    printf("Downsides:  Long Latency cost of broadcast\n");
    printf("-----------------------------------------\n");
  }

  uint32_t ReqSize(Region& region) { return region.length; }  
 
  uint64_t SendAsync(Region& region){
    uint32_t real_length = region.length;

    try_send_delayed();

    if(!delayed.empty()){
      delayed.push_back(&region);
      local_tail+=real_length;
      return local_tail; 
    }
 
    char* whole_message = (char*)(void*)(local_buffer->GetWriteAddr(real_length));
    if(whole_message == NULL){
      //printf("Failed to write, update tails\n");
      update_tail();
      whole_message = (char*)(void*)(local_buffer->GetWriteAddr(real_length));
      if(whole_message == NULL){
        delayed.push_back(&region);
        local_tail+=real_length;
        return local_tail;   
      }
    }  

    // copy to my memory
    memcpy(whole_message,region.addr,real_length);
       
    while(can_send < num_of_clients){
      all_poll();
    }
    // notify all
    for(auto ep:eps){
      ep->write_imm(NULL, 0, 0, 0, 0, real_length);  
    }
    can_send-=num_of_clients;

        
    local_tail+=real_length;
    return local_tail;   
  }

  bool AckSentBytes(uint32_t bytes){
    return true;
  }

  void WaitSend(uint64_t tail){ 
    while(replicated_tail < tail){
      TestSend(tail);
    }
  }

  bool TestSend(uint64_t tail){ 
    update_tail();
    if(replicated_tail >= tail) return true;
    try_send_delayed();
    if(replicated_tail >= tail) return true;
    return false;
  }

private:

  inline void all_poll(){
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(eps[0]->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS){
        printf("Failed request %d \n",wcs[i].status);
        exit(1);  
      }  
    }
    if(ret>0){
      can_send += ret;
    }
  }

  inline void try_send_delayed(){
    if(can_send < num_of_clients*4){
      all_poll();
    }
    while(!delayed.empty()){
      Region* region = delayed.front();
      update_tail();
      char *whole_message = (char*)(void*)(local_buffer->GetWriteAddr(region->length));
      if(whole_message == NULL){
        break;  
      }
      // copy to my memory
      memcpy(whole_message,region->addr,region->length);
    
      while(can_send < num_of_clients){
        all_poll();
      }
      // notify all
      for(auto ep:eps){
        ep->write_imm(NULL, 0, 0, 0, 0, region->length);  
      }
      can_send-=num_of_clients;
      delayed.pop_front();
    }
  }


private:
  inline void update_tail(){
    volatile uint64_t* new_tail = std::min_element(tails, tails + num_of_clients);
    local_buffer->UpdateHead(*new_tail);
    replicated_tail=*new_tail;
  }
};




// sender locally copies data to buffer. It can fail as readers need to progress it.

class ReadCircularConnectionMagicByte:  public SendCommunicator {
  MagicRemoteBuffer* const local_buffer;

    
  uint64_t local_tail = 0;
  uint64_t replicated_tail = 0;


  volatile uint64_t* const tails;
  const uint32_t num_of_clients;

  std::list<Region*> delayed;

  volatile uint64_t* const current_local_tail;
  struct ibv_dm *const dm;
public:
  ReadCircularConnectionMagicByte(uint32_t num_of_clients, MagicRemoteBuffer* local_buffer, uint64_t local_tail_ptr, uint64_t tail_ptrs, struct ibv_dm *dm): 
    local_buffer(local_buffer),tails((volatile uint64_t*)(void*)(tail_ptrs)),num_of_clients(num_of_clients), 
    current_local_tail((volatile uint64_t*)(void*)(local_tail_ptr)),dm(dm)
  {
    *(LEN_BYTE_T*)(void*)(local_buffer->GetWriteAddr(0)) = (LEN_BYTE_T) 0;
    if(current_local_tail!=NULL){
      *current_local_tail = 0;    
    }
        
    printf("Local tail that is read by clients is at  %p\n", current_local_tail);
    printf("Tails of clients should be written to  %p\n", tails);

  };

  ~ReadCircularConnectionMagicByte( ){};
 
  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Read Circular MagicByte Sender\n");
    printf("Role: Sender\n");
    printf("Type: One to Many.\n");
    printf("Used OPs: No\n");
    printf("Way of informing the receiver: write local tail that is RDMA readable\n");
    printf("Way of informing the receiver 2: writes len after writing data. So magic byte.\n");
    printf("Prefix: Yes - length.\n");
    printf("Suffix: No.\n"); 
    printf("Downside: Copy on send!\n"); 
    printf("Comment: Cannot use sges as we write/copy to local mem.\n");
    printf("-----------------------------------------\n");
  }

  uint32_t ReqSize(Region& region) { return region.length+ sizeof(LEN_BYTE_T); } // TODO. check later

  uint64_t SendAsync(Region& region){
 
    try_send_delayed();

    uint32_t real_length = region.length+ sizeof(LEN_BYTE_T) ;
        
    if(!delayed.empty()){
      delayed.push_back(&region);
      local_tail+=real_length;
      return local_tail;  
    }

    char* whole_message = (char*)(void*)(local_buffer->GetWriteAddr(real_length));
    if(whole_message == NULL){
      // printf("Failed to write, update heads\n");
      update_tail();

      whole_message = (char*)(void*)(local_buffer->GetWriteAddr(real_length));
      if(whole_message == NULL){
        delayed.push_back(&region);
        local_tail+=real_length;
        return local_tail;   
      }
    };  

    // we write data and then write len.
    memcpy(whole_message + sizeof(LEN_BYTE_T),region.addr,region.length);
    *(volatile LEN_BYTE_T*)(whole_message+ real_length) = (LEN_BYTE_T)0; // write 0 at the end
    *(volatile LEN_BYTE_T*)(whole_message) = (LEN_BYTE_T)region.length; // write len at the begin
 
    local_tail+=real_length;

    if(current_local_tail == NULL){
      //    printf("Copy local tail to device\n");
      if(ibv_memcpy_to_dm(dm, 0, &local_tail, sizeof(local_tail))){
        printf("failed to write to dev memory\n");
        exit(1);
      }
    }else{
      //           printf("write local tail to %p\n",current_local_tail);
      *current_local_tail = local_tail;
    }
        
    return local_tail;
  }

 

  inline void try_send_delayed(){
 
    while(!delayed.empty()){
      Region* region = delayed.front();
            
      update_tail();

      uint32_t real_length = region->length + sizeof(LEN_BYTE_T) ;
      char *whole_message = (char*)(void*)(local_buffer->GetWriteAddr(real_length));
      if(whole_message == NULL){
        break;  
      }
 
      // copy to my memory
      memcpy(whole_message + sizeof(LEN_BYTE_T),region->addr,region->length);
      *(LEN_BYTE_T*)(whole_message) = (LEN_BYTE_T)region->length; // write len at the begin
      delayed.pop_front();
    }
  }

  bool AckSentBytes(uint32_t bytes){
    return true;
  }

  void WaitSend(uint64_t tail){ 
    while(replicated_tail < tail){
      TestSend(tail);
    }
  }

  bool TestSend(uint64_t tail){ 
    update_tail();
    if(replicated_tail >= tail) return true;
    try_send_delayed();
    if(replicated_tail >= tail) return true;
    return false;
  }

private:
  inline void update_tail(){
    volatile uint64_t* new_tail = std::min_element(tails, tails + num_of_clients);
    local_buffer->UpdateHead(*new_tail);
    replicated_tail=*new_tail;
  }

};


// receiver fetch data from remote machine with read
class ReadCircularNotifyReceiver:  public PartialReceiveCommunicator {
  VerbsEP* const ep;
  MagicRingBuffer* const remote_buffer;


  const uint64_t rem_tail;    
  const uint32_t rem_tail_rkey;

  struct ibv_send_wr wr;
  struct ibv_sge sge;

  uint64_t tail = 0;
  
public:
  ReadCircularNotifyReceiver(VerbsEP* ep, MagicRingBuffer* remote_buffer,const uint64_t rem_tail, const uint32_t rem_tail_rkey): 
    ep(ep), remote_buffer(remote_buffer) , rem_tail(rem_tail), rem_tail_rkey(rem_tail_rkey)
  {
        
    sge = {(uint64_t)(void*)&tail, sizeof(tail), 0 };
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;   
    wr.wr.rdma.remote_addr = rem_tail; 
    wr.wr.rdma.rkey        = rem_tail_rkey;
    wr.next = NULL;
  };

  ~ReadCircularNotifyReceiver( ){};

 
  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Read Circular Notify Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: One to Many.\n");
    printf("Used OPs: Post Recv (get notifications), Read (fetch data), and Write (to notify progress)\n");
    printf("Passive: no\n");
    printf("-----------------------------------------\n");
  }

  int Receive(std::vector<Region> &v){
    int c = 0;
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->recv_cq,16,wcs);
        
    if(ret > 0) ep->post_empty_recvs((uint32_t)ret);

    ret+= ibv_poll_cq(ep->qp->send_cq,16-ret,&wcs[ret]);

    for(int i=0; i < ret; i++){
      if(wcs[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM){
        uint32_t length = wcs[i].imm_data;
        //printf("got imm with length %u\n",length);

        char* whole_message = remote_buffer->Read(length);
        //  printf("MEssage is expected to be at %p",whole_message);
        v.push_back({1, whole_message, (uint32_t)length, remote_buffer->GetKey() });  // it is pointer to remote memory
        c++;
      } else if(wcs[i].opcode == IBV_WC_RDMA_READ){
        //  printf("got read completion\n");
        char* whole_message = (char*)(void*)wcs[i].wr_id;
        v.push_back({0, whole_message, wcs[i].byte_len, 0 });  
        c++;
      }
    }
        
    return c;  
  }

  bool Receive(Region &s, Region &d){ // receive from source to destination
    // printf("Finishing recv\n");
    struct ibv_sge sge = {(uint64_t)(void*)d.addr,s.length,d.lkey};
    uint64_t wr_id = (uint64_t)(void*)d.addr;
    ep->read(&sge, 1, s.lkey, (uint64_t)(void*)s.addr, wr_id);

    return true;  
  }

  void FreeReceive(Region& region){
    remote_buffer->Free(region.length);
  }

  uint32_t GetFreedReceiveBytes(){
    // it will notify the sender that we read. we do it by writing the tail.
    uint32_t bytes = remote_buffer->GetFreedBytes();
    if(bytes){
      tail+=bytes;
      struct ibv_send_wr *bad_wr;
      if(ibv_post_send(ep->qp, &wr, &bad_wr )){
        printf("Failed to send\n");
        exit(1);
      }
    }
    return bytes; 
  }
};


// receiver fetch data from remote machine with read
class ReadCircularMagicByteReceiver:  public PartialReceiveCommunicator {
  VerbsEP* const ep;
  MagicRingBuffer* const remote_buffer;


  const uint64_t rem_tail;    
  const uint32_t rem_tail_rkey;

  struct ibv_send_wr wr;
  struct ibv_sge sge;

  uint64_t tail = 0;

  std::vector<uint8_t> slots;
  std::vector<uint64_t> remote_temp_addrs;
  std::vector<uint64_t> temp_recv_regions;
  const uint32_t lkey_of_prefetch;
  const uint32_t prefetch_size;

  struct ibv_sge temp_sges[16];
  uint8_t count=0;

  bool has_pending_prefetch_read = false;

  std::vector< std::pair<uint64_t,uint64_t> > done;
  
public:
  ReadCircularMagicByteReceiver(VerbsEP* ep, MagicRingBuffer* remote_buffer,const uint64_t rem_tail, const uint32_t rem_tail_rkey,
				uint64_t local_mem, uint32_t local_mem_lkey, uint32_t prefetch_size)
  : ep(ep), remote_buffer(remote_buffer) , rem_tail(rem_tail), rem_tail_rkey(rem_tail_rkey), lkey_of_prefetch(local_mem_lkey), prefetch_size(prefetch_size)
  {
        
    sge = {(uint64_t)(void*)&tail, sizeof(tail), 0 };
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;   
    wr.wr.rdma.remote_addr = rem_tail; 
    wr.wr.rdma.rkey        = rem_tail_rkey;
    wr.next = NULL;

    for(uint8_t i =0; i<16; i++){
      slots.push_back(i);
      temp_recv_regions.push_back(local_mem+ i*prefetch_size);
      remote_temp_addrs.push_back(0);
    }

    uint64_t raddr = (uint64_t)(void*)remote_buffer->GetReadPtr();
    printf("I will try to read from %lx\n", raddr);
    printf("I replicate my tail to  %lx\n", rem_tail);
    printf("Prefetch is  %u\n", prefetch_size);
  };

  ~ReadCircularMagicByteReceiver( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Read Circular MagicByte Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: One to Many.\n");
    printf("Used OPs: Post Recv (get notifications), Read (prefetch len and fetch data), and Write (to notify progress)\n");
    printf("Passive: no\n");
    printf("Add-ons: can prefetch data\n");
    printf("Comment: it expects in order processing from clients. Otherwise I need to implement out-of-order free\n");
    printf("-----------------------------------------\n");
  }

  int Receive(std::vector<Region> &v){
    int c = 0;

    // we have only one prefetch read at a time.
    if(!has_pending_prefetch_read){
      // make a temp read.
      //printf("make a temp read\n");
      if(!slots.empty()){
        has_pending_prefetch_read = true;
        uint64_t slot = slots.back();
        //  printf("got slot %lu\n",slot);
        uint64_t laddr =  temp_recv_regions[slot];

        uint64_t raddr = (uint64_t)(void*)remote_buffer->GetReadPtr();

        remote_temp_addrs[slot] = raddr;

        struct ibv_sge sge = {laddr,prefetch_size,lkey_of_prefetch};
                
        slots.pop_back();
        uint64_t wr_id = (slot<<56) ;
        //  printf("make read\n");
        ep->read(&sge, 1, remote_buffer->GetKey(),raddr , wr_id);

      }
    }

    if(!done.empty()){
      for(auto p: done){
        //   printf("push done %u \n",(uint32_t)p.second);
        c++;
        v.push_back({0, (char*)(void*)p.first, (uint32_t)p.second , 0 });
      }
      done.clear();
    }

    //  printf("poll cq\n");
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].opcode == IBV_WC_RDMA_READ){
        uint64_t addr = (wcs[i].wr_id &0xFFFFFFFFFFFFFF);
        if(!addr){
          // it is temp read. 
          uint8_t slot = wcs[i].wr_id >> 56;
          has_pending_prefetch_read = false;

          char* part_message = (char*)(void*)(temp_recv_regions[slot]);
          uint32_t offset = 0;

          while(offset <= prefetch_size){
            if(offset + sizeof(LEN_BYTE_T) > prefetch_size ){ // we do not have length in the message.
              // printf("discard some prefetched bytes\n");
              slots.push_back(slot);
              break;
            }

            LEN_BYTE_T length = *(LEN_BYTE_T*)(void*)(part_message+offset);
            if(length == 0){
              //   printf("discard as no data here\n");
              slots.push_back(slot);
              break;
            }else{
              // printf("got data in slot %u, with %lu bytes\n",slot, length);
              remote_buffer->Read(length+ sizeof(LEN_BYTE_T)); // we can move the pointer.
              //uint64_t raddr = (uint64_t)(void*)remote_buffer->GetReadPtr();
              //printf("I will try to read next from %lx\n", raddr);
              v.push_back({1, part_message + offset + sizeof(LEN_BYTE_T), (uint32_t)length, slot });  // slot will be used only in the last partial
              offset+=(length+sizeof(LEN_BYTE_T));
              c++;
            }
          }
        } else{
          // it is a normal read.
          struct ibv_sge* t = (struct ibv_sge*)(void*)addr;
          uint32_t length = t->length;
          // printf("got normal read: we prefetched: %u we  fetched extra %u \n",length-wcs[i].byte_len, wcs[i].byte_len  );
          v.push_back({0, (char*)(void*)t->addr, (uint32_t)length, 0 });      
          c++;
        }
      }
    }

    return c;  
  }

  bool Receive(Region &s, Region &d){ // receive from source to destination
    //  printf("finish recv\n");
        
    uint8_t slot = (uint8_t)s.lkey;
    uint32_t fetched =  (temp_recv_regions[slot] + prefetch_size) - ((uint64_t)(void*)s.addr); // only payload

    // printf("We prefetched %u \n",fetched);

    if(s.length  > fetched){
      //    printf("Fetch from remote\n");
      uint32_t need_to_fetch_more  = s.length - fetched;
      struct ibv_sge sge = {(uint64_t)(void*)d.addr + fetched, need_to_fetch_more, d.lkey};

      uint64_t wr_id = (uint64_t)(void*)&temp_sges[count];;

      temp_sges[count].addr = (uint64_t)(void*)d.addr;
      temp_sges[count].length = s.length;
      temp_sges[count].lkey = d.lkey;
      count = (count+1) % 16;

      uint64_t remote_addr = remote_temp_addrs[slot] + prefetch_size - fetched;
      ep->read(&sge, 1, remote_buffer->GetKey(), remote_addr, wr_id);
    }else{
      // done. 
      //   printf("done as fully fetched");
      done.push_back( {(uint64_t)(void*)d.addr,s.length});
    }

    if(fetched){
      //    printf("copy data \n");
      memcpy(d.addr, s.addr, MIN(fetched,s.length) );
    }

    slots.push_back(slot);

    return true;  
  }

  void FreeReceive(Region& region){ // we free full reads only
    //   printf("free recv %u \n",region.length + sizeof(LEN_BYTE_T));
    remote_buffer->Free(region.length + sizeof(LEN_BYTE_T));
  }

  uint32_t GetFreedReceiveBytes(){
    // it will notify the sender that we read. we do it by writing the tail.
    uint32_t bytes = remote_buffer->GetFreedBytes();
    if(bytes){
      //    printf("push freed\n");
      tail+=bytes;
      struct ibv_send_wr *bad_wr;
      if(ibv_post_send(ep->qp, &wr, &bad_wr )){
        printf("Failed to send\n");
        exit(1);
      }
    }
    return bytes; 
  }
};
 
 

// receiver fetch data from remote machine with read
class ReadCircularMailboxReceiver:  public PartialReceiveCommunicator {
  VerbsEP* const ep;
  MagicRingBuffer* const remote_buffer; 

  const uint64_t rem_tail;    
  const uint32_t rem_tail_rkey;


  const uint64_t rem_head;    
  const uint32_t rem_head_rkey;

  struct ibv_send_wr wr;
  struct ibv_sge sge;

  uint64_t tail = 0;
 
  uint64_t pending_head = 0;

  const uint64_t local_mem;  // read here and faa here. should be at least 16 bytes
  const uint32_t local_mem_lkey;
  volatile uint64_t* const remote_head;
    
  bool has_pending_read = false; // we have at most one outstanding head read

  std::vector< std::pair<uint64_t,uint64_t> > delayed;

  const bool in_order;

  std::unordered_map< uint64_t,uint64_t> addrs_mapper;
  
public:
  ReadCircularMailboxReceiver(VerbsEP* ep, MagicRingBuffer* remote_buffer,const uint64_t rem_tail, const uint32_t rem_tail_rkey,
			      uint64_t rem_head , uint32_t rem_head_rkey,uint64_t local_mem , uint32_t local_mem_lkey, bool in_order)
  : ep(ep), remote_buffer(remote_buffer) , rem_tail(rem_tail), rem_tail_rkey(rem_tail_rkey), rem_head(rem_head), rem_head_rkey(rem_head_rkey),
  local_mem(local_mem), local_mem_lkey(local_mem_lkey), remote_head(( volatile uint64_t*)(void*)local_mem), in_order(in_order)
  {
        
    sge = {(uint64_t)(void*)&tail, sizeof(tail), 0 };
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;   
    wr.wr.rdma.remote_addr = rem_tail; 
    wr.wr.rdma.rkey        = rem_tail_rkey;
    wr.next = NULL;

    *remote_head = 0;

    // printf("rem_tail is at %lx:%u\n",rem_tail,rem_tail_rkey);
  };
  ~ReadCircularMailboxReceiver( ){};

  void PrintInfo(){
    printf("-----------------------------------------\n");
    printf("Name: Shared Read Circular MagicByte Receiver\n");
    printf("Role: Receiver\n");
    printf("Type: One to Many.\n");
    printf("Used OPs:  Read (fetch tail and fetch data), and Write (to notify progress)\n");
    printf("Passive: no\n"); 
    printf("Add-ons:  ordered and not ordered recv.\n"); 
    printf("Problems:  The receiver does not know the size of fetched records.\n"); 
    printf("-----------------------------------------\n");
  }

  int Receive(std::vector<Region> &v){
    int c = 0;
        
    struct ibv_wc wcs[16];
    int ret = ibv_poll_cq(ep->qp->send_cq,16,wcs);
    for(int i=0; i < ret; i++){
      if(wcs[i].status != IBV_WC_SUCCESS){
        printf("Failed request %d \n",wcs[i].status);
        exit(1);  
      }  
      if(wcs[i].opcode == IBV_WC_RDMA_READ){
        if(wcs[i].wr_id == 0){ // updated head
          has_pending_read=false;
        }else{ 
          bool is_a_delay_cause = (wcs[i].wr_id >> 63); 

          uint64_t laddr = (wcs[i].wr_id &0xFFFFFFFFFFFFFF);
          uint64_t raddr = 0;
          if(!in_order){
                        
            auto it = addrs_mapper.find(laddr);
         
            laddr = it->second;
            raddr = it->first;
            //  printf("remove map %lx -> %lx\n",laddr, raddr);
            addrs_mapper.erase(it);
          }

          uint32_t total_length = wcs[i].byte_len; 
          uint32_t offset = 0;
          while(offset < total_length){
            LEN_BYTE_T length = *(LEN_BYTE_T*)(void*)(laddr+offset);
            if(offset + sizeof(LEN_BYTE_T)+length > total_length){
              //    printf("partial read.\n");

              uint64_t rem_addr = delayed[0].first - (total_length-offset);
              uint32_t len_region = delayed[0].second + (total_length-offset);
              if(length + sizeof(LEN_BYTE_T) != len_region){ // we can strip first entry
                                
                //   printf(">>>>>>>>>>>>>>>>..Push partial\n");
                v.push_back({1, (char*)(void*)rem_addr, (uint32_t)(length + sizeof(LEN_BYTE_T)), 0 });

                rem_addr+=(length + sizeof(LEN_BYTE_T));
                len_region-=(length + sizeof(LEN_BYTE_T));
                c++;
              }
              delayed[0] = {rem_addr,len_region};
            }else{
              uint32_t lkey = in_order ? 0 : remote_buffer->GetOffset((char*)(void*)raddr);
                            
              // printf(">>>>>>>>>>>>>>>>..Push finished\n");
              v.push_back({0, (char*)(void*)laddr + offset + sizeof(LEN_BYTE_T), (uint32_t)length, lkey});
              c++; 
            }
            offset += (length + sizeof(LEN_BYTE_T));
            raddr += (length + sizeof(LEN_BYTE_T));
                        
          }
          if(is_a_delay_cause){
            //    printf("Pushing delayed\n");
            for(auto p: delayed){
              v.push_back({1, (char*)(void*)p.first, (uint32_t)p.second , 0 });
              c++;
            }
            delayed.clear();
          }
        }
      }
    }

    if(!has_pending_read){  // fetch remote head
      struct ibv_sge sge = {(uint64_t)(void*)local_mem, sizeof(uint64_t),local_mem_lkey};
      //  printf("Fetch from %lx rkey %u\n",rem_head,rem_head_rkey);
      ep->read(&sge, 1, rem_head_rkey, rem_head, 0);
    }
    has_pending_read = true;

    uint32_t can_fetch = *remote_head - pending_head;
    if(can_fetch){
      //    printf("can fetch something %u\n",can_fetch);
      // uint32_t read_offset = remote_buffer->GetReadOff();
      uint64_t rem_addr = (uint64_t)(void*)remote_buffer->Read(can_fetch);
      if(!delayed.empty() && in_order){
        //        printf("I could read. but because of in-order requirement I delay it.\n");
        delayed.push_back({ rem_addr, (uint32_t)can_fetch});
      }else{
	//         printf("Create partial read\n");
        v.push_back({1, (char*)(void*)rem_addr, (uint32_t)can_fetch, 0 }); 
      }
      pending_head+=can_fetch; // progress
      c++;
    }

    return c;  
  }


  bool Receive(Region &s, Region &d){ // receive from source to destination
    // printf("finish recv\n");
        
    uint32_t to_fetch = MIN(s.length,d.length);

    uint64_t partial_read =  (to_fetch<s.length) ? (1ULL<<63) : 0ULL ; 

    if((in_order || partial_read) && !delayed.empty() ){
      to_fetch=0;        
    }
 
    if(partial_read){
      //      printf("we could not fetch everything. I delay a piece\n");
      delayed.push_back( { (uint64_t)(void*)s.addr + to_fetch,s.length - to_fetch});
    }
        
    if(to_fetch){
      //     printf("Fetch from remote\n"); 
      struct ibv_sge sge = {(uint64_t)(void*)d.addr, to_fetch, d.lkey};
      uint64_t remote_addr = (uint64_t)(void*)s.addr;

      uint64_t wr_id = ((uint64_t)(void*)d.addr) | partial_read ;
      if(!in_order){
	//          printf("add to map %lx -> %lx\n",sge.addr, remote_addr);
        addrs_mapper[remote_addr] = sge.addr;
        wr_id = remote_addr | partial_read ;
      }
      ep->read(&sge, 1, remote_buffer->GetKey(), remote_addr, wr_id);
            
    }
 
    return !partial_read;  
  }

  void FreeReceive(Region& region){ // we free full reads only
    //   printf("free recv\n");
    if(in_order){
      remote_buffer->Free(region.length + sizeof(LEN_BYTE_T));
    }else{
      remote_buffer->FreeOrdered(remote_buffer->GetReadPtr(region.lkey), region.length + sizeof(LEN_BYTE_T));
    }

  }

  uint32_t GetFreedReceiveBytes(){
    // it will notify the sender that we read. we do it by writing the tail.
    uint32_t bytes = remote_buffer->GetFreedBytes();
    if(bytes){
      //       printf("push tail to remote\n");
      tail+=bytes;
      struct ibv_send_wr *bad_wr;
      if(ibv_post_send(ep->qp, &wr, &bad_wr )){
        printf("Failed to send\n");
        exit(1);
      }
    }
    return bytes; 
  }
};
 
