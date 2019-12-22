
#include "Buffer.h"

Buffer::Buffer()
{
}

void Buffer::operator=(const Buffer& other)
{
    max_recv = other.max_recv;
    send = other.send;
    recv = other.recv;
}
