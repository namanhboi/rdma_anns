#include <zmq.h>
int main() {
  int major, minor, patch;
  zmq_version (&major, &minor, &patch);
  printf("Current 0MQ version is %d.%d.%d\n", major, minor, patch);
  void *ctx = zmq_ctx_new();
  void * rep = zmq_socket (ctx, ZMQ_PEER);
}
