
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Socket.h"


int Socket::MTU = 1400;

void Socket::init_sockets(Start start)
{
    switch (start)
    {
    case Start::Client:
        this->init_client();
    case Start::Server:
        this->init_server();
    case Start::Both:
        this->init_client();
        this->init_server();
    }
}


Socket::Socket(Start start, Protocol protocol)
    : client(*((Client*)nullptr)), server(*((Server*)nullptr)), m_pro(protocol), self(*this)
{
    if (m_pro != Protocol::TCP) {
        xstring err = "Only TCP Protocol is Supported at this Time\n";
        err.bold().red().reset().print();
        throw std::runtime_error(err);
    }
    this->init_sockets(start);
}

Socket::Socket(const Socket& other)
    : client(*((Client*)nullptr)), server(*((Server*)nullptr)), self(*this)
{
    *this = other;
}

Socket::~Socket()
{
    if(&server != nullptr) delete &server;
    if(&client != nullptr) delete &client;
}

void Socket::set_protocol(Socket::Protocol pro){
    m_pro = pro;
}

Socket& Socket::init_server()
{
    if (&server != nullptr)
        delete& server;

#ifdef WIN_BASE
    *(&this->m_offset_server + m_offset_count) = (size_t)new Win_Server(&Socket::MTU, &verbose, (int)m_pro);
#else
    *(&this->m_offset_server + m_offset_count) = (size_t)new Nix_Server(&Socket::MTU, &verbose, (int)m_pro);
#endif

    return *this;
}

Socket& Socket::init_client()
{
    if (&client != nullptr)
        delete &client;

#ifdef WIN_BASE
    *(&this->m_offset_client + m_offset_count) = (size_t)new Win_Client(&Socket::MTU, &verbose, (int)m_pro);
#else
    *(&this->m_offset_client + m_offset_count) = (size_t)new Nix_Client(&Socket::MTU, &verbose, (int)m_pro);
#endif
    return *this;
}

void Socket::operator=(const Socket& other)
{
    m_pro = other.m_pro;

    self = other.client;
    self = other.server;
}

void Socket::operator=(Client& other)
{
    this->init_client();
    
    if (&other == nullptr) return;

    client.m_ip = other.m_ip;
    client.m_port = other.m_port;
    client.buffer = other.buffer;
}

void Socket::operator=(Server& other)
{
    this->init_server();

    if (&other == nullptr) return;

    server.m_ip = other.m_ip;
    server.m_port = other.m_port;
    server.buffer = other.buffer;
}


