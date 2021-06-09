
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Client.h"

Client::Client(int* mtu, bool* verbose, int pro): m_mtu(*mtu), m_verbose(*verbose), m_pro(pro)
{
}

Client::Client(const Client& other): m_mtu(other.m_mtu), m_verbose(other.m_verbose)
{
    m_ip = other.m_ip;
    m_port = other.m_port;
    m_relay = other.m_relay;

    m_send_result = other.m_send_result;
    
    buffer = other.buffer;
}

void Client::operator<<(const xstring& buff) {
    buffer.send += buff;
}

void Client::operator=(const Client& other)
{
    m_pro = other.m_pro;
    m_ip = other.m_ip;
    m_port = other.m_port;
    buffer = other.buffer;
}

xstring Client::GetIP() const{
    return m_ip;
}

xstring Client::GetPort() const{
    return m_port;
}

