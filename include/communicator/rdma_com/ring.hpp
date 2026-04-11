#pragma once
/*
 * Ring Buffer for RDMA connection
 */



struct BufferContext {
  uint64_t addr;
  uint32_t rkey;
  uint32_t length;
};

struct connect_info{
  BufferContext ctx;
  uint32_t code;
  uint32_t rkey_magic; // we use it for faa as well
  uint64_t addr_magic;
  uint32_t dm_rkey; // dm_memory is always zero based.
  uint32_t reserved1;
  uint64_t addr_magic2;
  uint64_t server_id;
};

static_assert(sizeof(struct connect_info) == 56, "Connect info is of unexpected size.");

class Buffer {
  public:

    virtual ~Buffer() = default;

    //returns length of the buffer
    virtual uint32_t GetLength() const = 0;
    // Return the rkey of the MR
    virtual uint32_t GetKey() const = 0;
    
    // Returns a Context with all infomation necessary for the setup of a remote buffer
    virtual BufferContext GetContext() const = 0;
    // Returns the current read pointer without updating it
    virtual char *GetReadPtr() const = 0;
    // Returns the current read offset without updating it
    virtual uint32_t GetReadOff() const = 0;

    // Reads len bytes and updates the read pointer
    virtual char *Read(uint32_t len) = 0;
    // Frees the memory region and returns the new header position
    virtual uint64_t FreeOrdered(char *addr, uint32_t len) = 0;
    virtual uint64_t Free(uint32_t len) = 0; 

    virtual uint32_t GetFreedBytes() = 0;
};

class RemoteBuffer {
  public:
    virtual ~RemoteBuffer() = default;

    // Updates the head position to the new value
    virtual bool UpdateHead(uint64_t head) = 0;
    
    // Return the rkey of the MR
    virtual uint32_t GetKey() const = 0;
    // Returns the current tail  
    // There is always at least 4 bytes of usable space directly after the tail
    virtual uint32_t GetTail() const = 0;

    // Returns the write address if you want to write len bytes to the buffer and updates the tail accordingly
    // Can return an error if there is not enough free space
    virtual uint64_t GetWriteAddr(uint32_t len) = 0;

    virtual bool FreeBytes(uint32_t  bytes) = 0;
    

    // Returns the write address if you want to write len bytes to the buffer without updating the tail
    // Can return an error if there is not enough free space
    
    // Returns the tail address if you want to write len bytes to the buffer without actually updating the tail
    // Can return an error if there is not enough free space
    virtual uint32_t GetNextTail(uint32_t len) const = 0;
    
 
};
