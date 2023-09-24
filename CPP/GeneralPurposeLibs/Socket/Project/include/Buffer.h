#pragma once

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "xstring.h"

struct Buffer
{
    size_t max_recv = 0;
    xstring send;
    xstring recv;

    Buffer();

    void operator=(const Buffer& other);
};
