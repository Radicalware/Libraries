#pragma once

#include "xstring.h"

struct Buffer
{
    size_t max_recv = 1024;
    xstring send;
    xstring recv;

    Buffer();

    void operator=(const Buffer& other);
};
