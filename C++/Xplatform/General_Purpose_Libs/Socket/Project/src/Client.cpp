#include "Client.h"

Client::Client(int* mtu): mtu(*mtu)
{
}

void Client::operator<<(const xstring& buff) {
    buffer.send += buff;
}

void Client::operator=(const Client& other)
{
    m_ip = other.m_ip;
    m_port = other.m_port;
    buffer = other.buffer;
}

xstring Client::ip() const{
    return m_ip;
}

xstring Client::port() const{
    return m_port;
}

