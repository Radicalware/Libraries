#pragma once

#include <stdexcept>

#include "Buffer.h"
#include "xstring.h"

class Client
{
    friend class Socket;

protected:
    xstring m_ip;
    xstring m_port;
    xstring m_relay;

    int m_send_result;

public:
    int& mtu;
    Buffer buffer;

    Client(int* mtu);
    virtual Client& connect() = 0;
    virtual Client& connect(const xstring& ip, const xstring& port) = 0;
    virtual Client& send(const xstring& buff = "") = 0;
    virtual Client& recv(int size = 0) = 0;
    virtual Client& close() = 0;

    virtual void operator<<(const xstring& buff);
    virtual void operator=(const Client& other);

    xstring ip() const;
    xstring port() const;
};