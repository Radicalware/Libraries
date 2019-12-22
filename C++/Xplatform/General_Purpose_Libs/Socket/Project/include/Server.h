#pragma once

#include <stdexcept>
#include <string_view>

#include "Buffer.h"
#include "xstring.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))      
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#pragma comment (lib, "Ws2_32.lib")
#else

#endif

class Server
{
    friend class Socket;

protected:
    SOCKET m_listen_socket;
    SOCKET m_client_socket;

    int m_result = 0;
    int m_send_result = 0;

    xstring m_ip;
    xstring m_port;

    std::function<xstring()> m_method;
    xstring m_relay;

public:
    int& mtu;
    Buffer buffer;


    template<typename F, typename... A>
    void set_function(F&& Function, A&&... Args);

    Server(int* mtu);
    virtual Server& listen(const xstring& port) = 0;
    virtual Server& accept() = 0;
    virtual Server& recv(int size = 0) = 0; // derrived
    virtual Server& respond() = 0; // derrived
    virtual Server& close() = 0; 
};


template<typename F, typename ...A>
inline void Server::set_function(F&& Function, A&& ...Args)
{
    //m_method = Function(&this->buffer.recv, Args...);
    m_method = std::bind(Function, &this->buffer.recv, Args...);
}