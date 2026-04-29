#pragma once
#include <infiniband/verbs.h>
#include <vector>
#include <arpa/inet.h>
#include "VerbsEP.hpp"
#include "../region.h"

class SendConnection {
    VerbsEP* const ep;
    uint64_t last_wrid = 0;
    uint64_t next_id_ = 1; // Starts at 1 to prevent Ghost 0!

public:
    SendConnection(VerbsEP* ep): ep(ep) {};
    ~SendConnection() {};

    // Sends pure Data. Payload = User Data. IMM = 0
uint64_t SendAsync(Region* region) {
        struct ibv_sge sge;
        sge.addr = (uint64_t)(void*)region->addr;
        sge.length = region->length;
        sge.lkey = region->lkey;

        struct ibv_send_wr wr = {};
        wr.wr_id = next_id_;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        wr.opcode = IBV_WR_SEND_WITH_IMM;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.imm_data = htonl(0);

        struct ibv_send_wr *bad_wr;
        int ret = ibv_post_send(ep->qp, &wr, &bad_wr);
        if (ret) {
            // SILENT FAILURE EXPOSED:
            printf("Hardware rejected SendAsync! Error: %d\n", ret);
            return (uint64_t)-1;
        }

        // Print success so we know Node 0 is actually firing!
        // printf("Blasted Data Message ID: %lu\n", next_id_);
        return next_id_++;
    }

    uint64_t SendAckAsync(uint32_t freed_credits) {
        struct ibv_send_wr wr = {};
        wr.wr_id = next_id_;
        wr.sg_list = nullptr;
        wr.num_sge = 0;

        wr.opcode = IBV_WR_SEND_WITH_IMM;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.imm_data = htonl(freed_credits);

        struct ibv_send_wr *bad_wr;
        int ret = ibv_post_send(ep->qp, &wr, &bad_wr);
        if (ret) {
            printf("Hardware rejected ACK! Error: %d\n", ret);
            return (uint64_t)-1;
        }

        printf("Fired an ACK for %u credits back to Sender!\n", freed_credits);
        return next_id_++;
    }
    bool TestSend(uint64_t id) {
        if(last_wrid >= id) return true;
        struct ibv_wc wcs[16];
        int ret = ibv_poll_cq(ep->qp->send_cq, 16, wcs);
        for(int i = 0; i < ret; i++) {
            if(wcs[i].status != IBV_WC_SUCCESS) {
                printf("Failed send request %d\n", wcs[i].status);
                exit(1);
            }
            last_wrid = wcs[i].wr_id;
        }
        return last_wrid >= id;
    }
};

class ReceiveReceiver {
    VerbsEP* const ep;
    char* const buffer_base;
    const uint32_t size;
    const uint32_t lkey;

public:
    // Pre-posts all available buffers to the NIC
    ReceiveReceiver(VerbsEP* ep, char* mem, uint32_t size, uint32_t max_recv_size, uint32_t lkey):
    ep(ep), buffer_base(mem), size(size), lkey(lkey) {
        for(uint32_t i = 0; i < max_recv_size; i++) {
            char* addr = buffer_base + (i * size);
            FreeReceive(addr); // Hand the "empty plate" to the NIC
        }
    }
    ~ReceiveReceiver() {};

    // Modified to return BOTH data and flow-control credits
int Receive(std::vector<Region> &v, std::vector<uint32_t> &incoming_credits) {
        int c = 0;
        struct ibv_wc wcs[16];
        int ret = ibv_poll_cq(ep->qp->recv_cq, 16, wcs);

        for(int i = 0; i < ret; i++) {
            if(wcs[i].status == IBV_WC_SUCCESS) {
                char* consumed_buffer = (char*)(void*)wcs[i].wr_id;

                // 1. Did the hardware include our Immediate Data?
                if (wcs[i].wc_flags & IBV_WC_WITH_IMM) {
                    uint32_t imm_val = ntohl(wcs[i].imm_data);

                    if (imm_val == 0) {
                        uint32_t length = wcs[i].byte_len;
                        v.push_back({0, consumed_buffer, length, lkey});
                        c++;
                    } else {
                        incoming_credits.push_back(imm_val);
                        FreeReceive(consumed_buffer);
                    }
                } else {
                    // SILENT FAILURE EXPOSED:
                    printf("WARNING: Received a message without IMM data! Dropping it.\n");
                    // We must free it anyway or the NIC gets permanently starved!
                    FreeReceive(consumed_buffer);
                }
            } else {
                printf("Failed recv WQE! Status code: %d\n", wcs[i].status);
                exit(1);
            }
        }
        return c;
    }
    void FreeReceive(char* addr) {
        struct ibv_sge sge = {(uint64_t)addr, size, lkey};
        struct ibv_recv_wr wr = {};
        struct ibv_recv_wr *bad_wr;

        wr.wr_id = (uint64_t)addr; // Track the buffer address!
        wr.sg_list = &sge;
        wr.num_sge = 1;

        int ret = ibv_post_recv(ep->qp, &wr, &bad_wr);
        if (ret) {
          // THE FIX: Print the exact error code
          printf("Failed to post recv! Error code: %d\n", ret);
          exit(1);
        }
    }
};
