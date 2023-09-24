
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

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
