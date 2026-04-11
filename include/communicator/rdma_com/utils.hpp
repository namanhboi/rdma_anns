#pragma once

#include <sys/mman.h>
#include <sys/shm.h>

void *GetMagicBuffer(size_t buf_size) {
  // (fischi) We reserve virtual memory two times the size of the buffer. This
  // mapping is not actually used, but we make sure that we have enough reserved
  // space. I stole this hack from https://github.com/smcho-kr/magic-ring-buffer
  // . This might actually be racy, but I think this should be fine

  size_t PAGEMASK = (4096 - 1);
  if (buf_size & PAGEMASK) {
    printf("Sise must be the multiple of the page size\n");
    return NULL;
  }

  void *buf_addr = mmap(NULL, 2 * buf_size, PROT_READ | PROT_WRITE,
                        MAP_ANONYMOUS | MAP_SHARED, -1, 0);
  if (buf_addr == MAP_FAILED) {
    printf("Failled to mmap memory\n");
    return NULL;
  }

  // allocate shared memory segment that needs to be mapped twice
  int shm_id = shmget(IPC_PRIVATE, buf_size, IPC_CREAT | 0700);
  if (shm_id < 0) {
    munmap(buf_addr, 2 * buf_size);
    printf("Failled to shmget memory\n");
    return NULL;
  }
  // We actually don't need this mapping
  munmap(buf_addr, 2 * buf_size);

  // attach shared memory to first buffer segment
  if (shmat(shm_id, buf_addr, 0) != buf_addr) {
    shmctl(shm_id, IPC_RMID, NULL);
    printf("Failled to shmmat memory\n");
    return NULL;
  }
  // attach shared memory to second buffer segment
  void *sec_addr = (void *)((size_t)buf_addr + buf_size);
  if (shmat(shm_id, sec_addr, 0) != sec_addr) {
    shmdt(buf_addr);
    shmctl(shm_id, IPC_RMID, NULL);
    printf("Failled to shmmat second part of memory\n");
    return NULL;
  }

  // frees shared memory as soon as process exits.
  if (shmctl(shm_id, IPC_RMID, NULL) < 0) {
    shmdt(buf_addr);
    shmdt((char *)buf_addr + buf_size);
    printf("Failled to shmctl second part of memory. errno: %d\n", errno);
    return NULL;
  }

  return buf_addr;
}

void FreeMagicBuffer(void *buf_addr, size_t buf_size) {
  if (shmdt(buf_addr)) {
    printf("Failled to umapping first half of buffer\n");
  }
  if (shmdt((char *)buf_addr + buf_size)) {
    printf("Failled to umapping second half of buffer\n");
  }
}
