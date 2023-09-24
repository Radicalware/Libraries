
#include "NixNet.h"

#ifdef __unix__

NixNet::NixNet(){
    m_size = sizeof(addr);
}

void NixNet::operator=(const NixNet& other)
{
    m_send_size = other.m_send_size;
    m_size = other.m_size;
    ret_size = other.ret_size;

    mtu = other.mtu;
    connected = other.connected;
    addr = other.addr;
    connect = other.connect;
    listen = other.listen;
}

int NixNet::Send(const char* message)
{
    int mtu_cpy = *mtu;
    ::send(connect, &mtu_cpy, sizeof(mtu_cpy), 0);
    return ::send(connect, &message[0], mtu_cpy, 0); 
}

int NixNet::Recv(xstring& message)
{
    int size = *mtu;
    message.clear();
    message.resize(size);
    ::recv(connect, &size, sizeof(size), 0);
    ret_size = ::recv(connect, &message[0], size, 0);
    int str_size = strlen(message.c_str());
    if(str_size < size && str_size > 0)
        message.erase(message.begin() + str_size, message.end());
    return (str_size < ret_size) ? str_size : ret_size;
}

socklen_t& NixNet::Size() {
    return m_size;
}

#endif
