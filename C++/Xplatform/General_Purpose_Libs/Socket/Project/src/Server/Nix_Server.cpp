
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#ifdef __unix__

#include "Server/Nix_Server.h"

Nix_Server::Nix_Server(int* mtu, bool* verbose, int pro): Server(mtu, verbose, pro)
{
    m_net.mtu = &m_mtu;
}

Server& Nix_Server::listen(const xstring& port)
{
    if (port.size())
        m_port = port;

    m_net.listen = socket(AF_INET, SOCK_STREAM, 0);
    if(m_net.listen < 0)
    {
        xstring err_str("! Server: Setting Socket Failed with Error: " + to_xstring(m_net.connect) + '\n');
        throw std::runtime_error(err_str);
    }

    memset(&m_net.addr, 0, sizeof(m_net.addr));

    m_net.addr.sin_family        = AF_INET;
    m_net.addr.sin_addr.s_addr   = htonl(INADDR_ANY);
    m_net.addr.sin_port          = htons(atoi(m_port.c_str()));

    m_result = bind
    (
        m_net.listen, 
        (struct sockaddr*) &m_net.addr, 
        sizeof(m_net.addr)
    );
    if(m_result < 0)
    {
        xstring err_str("! Server: Bind Failed with Error: " + to_xstring(m_result) + '\n');
        throw std::runtime_error(err_str);
    }

    m_result  = ::listen(m_net.listen, m_mtu);
    if(m_result < 0)
    {
        xstring err_str("! Server: Listen Failed with Error: " + to_xstring(m_result) + '\n');
        throw std::runtime_error(err_str);
    }

    return *this;
}


Server& Nix_Server::accept()
{
    size_t len = m_net.size();

    m_net.connect = ::accept
    (
        m_net.listen, 
        (struct sockaddr*) &m_net.addr, 
        &m_net.size()
    );
    return *this;
}
Server& Nix_Server::recv(int size)
{
    if (size)
        buffer.max_recv = size;

    // Receive and React until the peer shuts down the connection
    int bytes_left = this->buffer.max_recv;
    do {
        m_result = m_net.recv(m_relay);

        if (m_result > 0){
            if(m_verbose)
                xstring("Server >> Bytes Received: " + to_xstring(m_result)).bold().red().reset().print();
        }
        else if (m_result == 0) 
        {
            if (m_verbose)
                xstring("Server >> Connection closing... ").bold().red().reset().print();
        }
        else {
            xstring err_str("! Server: Send Failed with Error: " + to_xstring(m_result) + '\n');            
            throw std::runtime_error(err_str);
        }

        if (!buffer.max_recv) // grab all the data
        {
            buffer.recv += m_relay;
        }
        else // limit data intake to server
        {
            if (m_result <= bytes_left)
            {
                if (m_result == m_mtu)
                    buffer.recv += m_relay;
                else {
                    buffer.recv += m_relay.substr(0, m_result);
                    break;
                }
            }
            else {
                buffer.recv += m_relay.substr(0, bytes_left);
                break;
            }
            bytes_left -= m_result;
        }
        
    } while (m_result >= m_mtu);
    m_relay.clear();

    if (m_verbose)
        xstring(xstring("Server Received Data: ") + buffer.recv).bold().red().reset().print();
    return *this;
}


Server& Nix_Server::respond()
{
    buffer.send.clear();
    this->buffer.send = m_method();

    size_t leng = 0;
    if (m_mtu < buffer.send.size())
    {
        size_t count = 0;
        size_t max = buffer.send.size();

        std::string_view view(buffer.send.c_str(), max);
        while (max > count)
        {
            if(m_mtu + count < max){
                leng = m_mtu;
                m_result = m_net.send(view.substr(count, leng).data());
            }else {
                leng = max - count;
                m_result = m_net.send(view.substr(count, leng).data());
            }
            count += m_mtu;

            if (m_verbose)
                xstring("Server >> Bytes sent: " + to_xstring(leng)).bold().red().reset().print();
        }
    }
    else
    {
        m_result = m_net.send(buffer.send.c_str());

        if (m_verbose)
            xstring("Server >> Bytes sent: " + to_xstring(buffer.send.size()) + '\n').bold().red().reset().print();
    }

    if (m_result < 0)
    {
        xstring err_str("! Server: Send Failed with Error: " + to_xstring(m_result) + '\n');
        throw std::runtime_error(err_str);
    }

    return *this;
}

Server& Nix_Server::close()
{
    m_result = ::close(m_net.connect);
    if (m_result < 0)
    {
        xstring err_str("! Server: Failed to close 'm_net.connect' with Error: " + to_xstring(m_result) + '\n');
        throw std::runtime_error(err_str);
    }

    m_result = ::close(m_net.listen);
    if (m_result < 0)
    {
        xstring err_str("! Server: Failed to close 'm_net.listen' with Error: " + to_xstring(m_result) + '\n');
        throw std::runtime_error(err_str);
    }    
    
    return *this;
}

#endif