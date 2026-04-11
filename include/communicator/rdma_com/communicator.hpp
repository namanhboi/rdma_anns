#pragma once



struct Region {
  uint64_t    context;
  char *      addr;
  uint32_t    length;
  uint32_t    lkey; 
};

class SendCommunicator {
public:
  virtual ~SendCommunicator() = default;

  virtual uint32_t ReqSize(Region& region) = 0; // get number of bytes that we will send for that request.
  virtual uint64_t SendAsync(Region& region) = 0;
  virtual bool AckSentBytes(uint32_t bytes) = 0;
  virtual bool TestSend(uint64_t id)= 0;
  virtual void WaitSend(uint64_t id)= 0;
};

class ReceiveCommunicator {
public:
  virtual ~ReceiveCommunicator() = default;

  virtual int Receive(std::vector<Region> &v) = 0;
  virtual void FreeReceive(Region& region) = 0;
  virtual uint32_t GetFreedReceiveBytes() = 0;
  virtual void PrintInfo() = 0;
};

class PartialReceiveCommunicator {
public:
  virtual ~PartialReceiveCommunicator() = default;
  virtual bool Receive(Region &s, Region &d) = 0;
  virtual int Receive(std::vector<Region> &v) = 0;
  virtual void FreeReceive(Region& region) = 0;
  virtual uint32_t GetFreedReceiveBytes() = 0;
  virtual void PrintInfo() = 0;
};



class ReadCommunicator {
public:
  virtual ~ReadCommunicator() = default;

  virtual uint64_t SendAsync(Region* region) = 0;
  virtual int Receive(std::vector<Region*> &in,std::vector<Region*> &out) = 0;
  virtual void Free(Region& region) = 0;
  virtual void PrintInfo() = 0;
};
