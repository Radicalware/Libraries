#pragma once

#include "xstring.h"

#ifdef __unix__

#include<sys/socket.h>
#include<sys/wait.h>
#include<sys/types.h>

#include<arpa/inet.h>

#include<stdio.h>
#include<string.h>
#include<unistd.h>
#include<stdlib.h>


struct NixNet
{
private:
    char* blank;
    size_t m_send_size = 0;
    socklen_t m_size = 0;
    int ret_size;
public:
    int* mtu = nullptr;
    bool connected = false;
    struct sockaddr_in addr;
    int connect = 0;
    int listen = 0;

    NixNet();
    void operator=(const NixNet& other);
    int send(const char* message);
    int recv(xstring& message);
    socklen_t& size();
};

#endif