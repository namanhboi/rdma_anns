#include <zmq.h>
#include <stdio.h>

int main() {
    int major, minor, patch;
    zmq_version(&major, &minor, &patch);
    printf("ZeroMQ library version: %d.%d.%d\n", major, minor, patch);
    return 0;
}
