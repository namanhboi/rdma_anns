#pragma once
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>




struct ibv_qp_init_attr prepare_qp(struct ibv_pd *pd,  uint32_t max_send_size, uint32_t max_recv_size, bool with_srq){

   struct ibv_srq *srq = NULL;

    if(with_srq){
      printf("create SRQ");
      struct ibv_srq_init_attr srq_init_attr = {};
      srq_init_attr.attr.max_wr = max_recv_size;
      srq_init_attr.attr.max_sge = 1;

      srq = ibv_create_srq(pd,&srq_init_attr);
      if(!srq){
        printf("Failed to create SRQ.\n");
        exit(1);
      }  
    } else{
      printf("NO shared receive\n");
    }
     

    struct ibv_cq* send_cq = ibv_create_cq(pd->context, max_send_size*2, NULL, NULL, 0); 
    struct ibv_cq* recv_cq = ibv_create_cq(pd->context, max_recv_size*2, NULL, NULL, 0); 
    if (!send_cq || !recv_cq) {
      fprintf(stderr, "Couldn't create CQ\n");
      exit(1);
    }   

    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = send_cq;
    attr.recv_cq = recv_cq;
    attr.cap.max_send_wr = max_send_size; 
    attr.cap.max_recv_wr = max_recv_size; 
    attr.cap.max_send_sge = 1;  
    attr.cap.max_recv_sge =   1;
    attr.cap.max_inline_data = 8;
    attr.srq = srq;
    attr.qp_type = IBV_QPT_RC;
    attr.sq_sig_all = 0;

    return attr;
};


class VerbsEP{
public:
  struct rdma_cm_id * const id;

  struct ibv_qp * const qp;
  struct ibv_pd * const pd;
  const uint32_t max_inline_data;
  const uint32_t max_send_size;
  const uint32_t max_recv_size;

  const uint32_t recv_batch;
  struct ibv_srq* const srq;

  uint32_t can_post = 0;


  const uint32_t max_send_sge;

  std::vector<struct ibv_sge> recv_sges;
  std::vector<struct ibv_recv_wr> recv_wr;
  
  VerbsEP(struct rdma_cm_id *id, struct ibv_qp_init_attr attr, uint32_t recv_batch, bool recv_with_data = false):
  id(id), qp(id->qp), pd(qp->pd), max_inline_data(attr.cap.max_inline_data), max_send_size(attr.cap.max_send_wr), 
  max_recv_size(attr.cap.max_recv_wr), recv_batch(recv_batch), srq(attr.srq ), max_send_sge(attr.cap.max_send_sge)
  {
    recv_wr.resize(recv_batch);
    recv_sges.resize(recv_batch);

    uint64_t cid = (uint64_t)id->context;

    struct ibv_recv_wr* wrs = recv_wr.data();
    struct ibv_sge* sges = recv_sges.data();

    for(uint32_t i=0; i < recv_batch; i++){
      wrs[i].sg_list = &sges[i];
      wrs[i].num_sge = recv_with_data ? 1 : 0;
      wrs[i].wr_id = cid;
      wrs[i].next = &wrs[i+1];
    }
    wrs[recv_batch-1].next = NULL;

    if(!recv_with_data){
      if(srq == nullptr || cid == 0){
          printf(">>>>>>>> post recvs\n");
          post_empty_recvs(max_recv_size);
      } 
    }

  }

  ~VerbsEP(){
    // empty
  }

  uint32_t GetMaxSendSge() const{
    return max_send_sge;
  }
 
 
  enum rdma_cm_event_type get_event(){
      int ret;
      struct rdma_cm_event *event;
      
      ret = rdma_get_cm_event(id->channel, &event);
      if (ret) {
          perror("rdma_get_cm_event");
          exit(ret);
      }
      enum rdma_cm_event_type out = event->event;
     /* switch (event->event){
          case RDMA_CM_EVENT_ADDR_ERROR:
          case RDMA_CM_EVENT_ROUTE_ERROR:
          case RDMA_CM_EVENT_CONNECT_ERROR:
          case RDMA_CM_EVENT_UNREACHABLE:
          case RDMA_CM_EVENT_REJECTED:
   
               text(log_fp,"[rdma_get_cm_event] Error %u \n",event->event);
              break;

          case RDMA_CM_EVENT_DISCONNECTED:
              text(log_fp,"[rdma_get_cm_event] Disconnect %u \n",event->event);
              break;

          case RDMA_CM_EVENT_DEVICE_REMOVAL:
              text(log_fp,"[rdma_get_cm_event] Removal %u \n",event->event);
              break;
          default:
              text(log_fp,"[rdma_get_cm_event] %u \n",event->event);

      }*/
      rdma_ack_cm_event(event);
      return out;
  }   

  inline int post_empty_recvs(uint32_t post){
      int ret = 0;
      struct ibv_recv_wr *bad;
      can_post+=post;
      while(can_post >= recv_batch){
          can_post-=recv_batch;
          if(srq != nullptr){
            ret = ibv_post_srq_recv(srq, recv_wr.data(), &bad);
          } else {
            ret = ibv_post_recv(qp, recv_wr.data(), &bad);
          }
      }
      return ret;
  } 


  inline int post_recv(struct ibv_sge &sge){
      int ret = 0;
      struct ibv_recv_wr *bad;
      recv_sges[can_post] = sge;
      recv_wr[can_post].wr_id = sge.addr;

      can_post+=1;
      if(can_post >= recv_batch){
          can_post-=recv_batch;
          if(srq != nullptr){
            ret = ibv_post_srq_recv(srq, recv_wr.data(), &bad);
          } else {
            ret = ibv_post_recv(qp, recv_wr.data(), &bad);
          }
      }
      return ret;
  }

  inline int trigget_post_recv( ){
      int ret = 0;
      struct ibv_recv_wr *bad;
      if(can_post){
          struct ibv_recv_wr* wrs = recv_wr.data();
          wrs[can_post-1].next = NULL;
          if(srq != nullptr){
            ret = ibv_post_srq_recv(srq, recv_wr.data(), &bad);
          } else {
            ret = ibv_post_recv(qp, recv_wr.data(), &bad);
          }
          wrs[can_post-1].next = &wrs[can_post]; 
          can_post=0;
      }
      return ret;
  }

  inline int send(struct ibv_sge* sges, uint32_t sgelen, uint64_t wr_id){        
      struct ibv_send_wr wr, *bad;

      wr.wr_id = wr_id;
      wr.next = NULL;
      wr.sg_list = sges;
      wr.num_sge = sgelen;
      wr.opcode = IBV_WR_SEND;

      wr.send_flags = IBV_SEND_SIGNALED;   

      return ibv_post_send(this->qp, &wr, &bad);  
  }

  inline int write(struct ibv_sge* sges, uint32_t sgelen, uint32_t rkey, uint64_t remote_addr, uint64_t wr_id){        
      struct ibv_send_wr wr, *bad;

      wr.wr_id = wr_id;
      wr.next = NULL;
      wr.sg_list = sges;
      wr.num_sge = sgelen;
      wr.opcode = IBV_WR_RDMA_WRITE;

      wr.send_flags = IBV_SEND_SIGNALED;   
 
      wr.wr.rdma.remote_addr = remote_addr;
      wr.wr.rdma.rkey        = rkey;

      return ibv_post_send(this->qp, &wr, &bad);  
  }
  inline int write_imm(struct ibv_sge* sges, uint32_t sgelen, uint32_t rkey, uint64_t remote_addr, uint64_t wr_id, uint32_t imm_data){        
      struct ibv_send_wr wr, *bad;

      wr.wr_id = wr_id;
      wr.next = NULL;
      wr.sg_list = sges;
      wr.num_sge = sgelen;
      wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      wr.imm_data = imm_data;

      wr.send_flags = IBV_SEND_SIGNALED;   
 
      wr.wr.rdma.remote_addr = remote_addr;
      wr.wr.rdma.rkey        = rkey;

      return ibv_post_send(this->qp, &wr, &bad);  
  }

  inline int read(struct ibv_sge* sges, uint32_t sgelen, uint32_t rkey, uint64_t remote_addr, uint64_t wr_id){        

 
      struct ibv_send_wr wr, *bad;

      wr.wr_id = wr_id;
      wr.next = NULL;
      wr.sg_list = sges;
      wr.num_sge = sgelen;
      wr.opcode = IBV_WR_RDMA_READ;

      wr.send_flags = IBV_SEND_SIGNALED;   
 
      wr.wr.rdma.remote_addr = remote_addr;
      wr.wr.rdma.rkey        = rkey;

      return ibv_post_send(this->qp, &wr, &bad);  
  }

};
