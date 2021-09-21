
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Socket.h"


int Socket::MTU = 1400;

void Socket::InitSockets(Start start)
{
    switch (start)
    {
    case Start::Client:
        this->Initclient();
    case Start::Server:
        this->InitServer();
    case Start::Both:
        this->Initclient();
        this->InitServer();
    }
}


Socket::Socket(Start start, Protocol protocol)
    : client(*((Client*)nullptr)), server(*((Server*)nullptr)), m_pro(protocol), self(*this)
{
    if (m_pro != Protocol::TCP) {
        xstring err = "Only TCP Protocol is Supported at this Time\n";
        err.ToBold().ToRed().ResetColor().Print();
        throw std::runtime_error(err);
    }
    this->InitSockets(start);
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

void Socket::SetProtocol(Socket::Protocol pro){
    m_pro = pro;
}

Socket& Socket::InitServer()
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

Socket& Socket::Initclient()
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
    this->Initclient();
    
    if (&other == nullptr) return;

    client.m_ip = other.m_ip;
    client.m_port = other.m_port;
    client.buffer = other.buffer;
}

void Socket::operator=(Server& other)
{
    this->InitServer();

    if (&other == nullptr) return;

    server.m_ip = other.m_ip;
    server.m_port = other.m_port;
    server.buffer = other.buffer;
}


