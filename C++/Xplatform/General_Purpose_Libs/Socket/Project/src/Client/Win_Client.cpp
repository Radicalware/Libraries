
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence


#include "Client/Win_Client.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))

Win_Client::Win_Client(int* mtu, bool* verbose, int pro) : Client(mtu, verbose, pro)
{
}

Win_Client::Win_Client(const Win_Client& other) : Client(other)
{
    m_socket = other.m_socket;
    m_result = other.m_result;
}

Win_Client::Win_Client(const Client& other) : Client(other)
{
}

Client& Win_Client::connect()
{
    WSADATA wsaData;
    struct addrinfo* result = nullptr, * ptr = nullptr, addr;

    // Initialize Winsock
    m_result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (m_result != 0)
    {
        xstring err_str = "! (WSAStartup) Failed with Error: " + std::to_string(m_result) + '\n';
        throw std::runtime_error(err_str);
    }

    ZeroMemory(&addr, sizeof(addr));
    addr.ai_family = AF_UNSPEC;

    if (m_pro == 1)
        addr.ai_socktype = SOCK_STREAM;
    else if (m_pro == 2)
        addr.ai_socktype = SOCK_DGRAM;
    else
        throw std::runtime_error("! Windows RAW Ports not Programmed yet\n");

    addr.ai_protocol = IPPROTO_TCP;

    // Resolve the server address and port
    m_result = getaddrinfo(m_ip.c_str(), m_port.c_str(), &addr, &result);
    if (m_result != 0)
    {
        xstring err_str = "! (getaddrinfo) Failed with Error: " + std::to_string(m_result) + '\n';
        WSACleanup();
        throw std::runtime_error(err_str);
    }

    // Attempt to connect to an address until one succeeds
    for (ptr = result; ptr != NULL; ptr = ptr->ai_next)
    {
        // Create a SOCKET for connecting to server
        m_socket = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
        if (m_socket == INVALID_SOCKET)
        {
            xstring err_str = "! (socket) Failed with Error: " + std::to_string(WSAGetLastError()) + '\n';
            WSACleanup();
            throw std::runtime_error(err_str);
        }

        // Connect to server.
        m_result = ::connect(m_socket, ptr->ai_addr, (int)ptr->ai_addrlen);
        if (m_result == SOCKET_ERROR)
        {
            closesocket(m_socket);
            m_socket = INVALID_SOCKET;
            continue;
        }
        break;
    }
    freeaddrinfo(result);

    if (m_socket == INVALID_SOCKET)
    {
        xstring err_str = "! Unable to connect to server!\n";
        WSACleanup();
        throw std::runtime_error(err_str);
    }

    return *this;
}

Client& Win_Client::connect(const xstring& ip, const xstring& port)
{
    m_ip = ip;
    m_port = port;
    this->connect();
    return *this;
}

Client& Win_Client::send(const xstring& buff)
{
    buffer.send += buff;
    
    if (m_mtu < buffer.send.size())
    {
        size_t count = 0;
        size_t max = buffer.send.size();

        std::string_view view(buffer.send.c_str(), max);
        while (max > count)
        {
            if (m_mtu + count < max)
                m_send_result = ::send(m_socket, view.substr(count, count + m_mtu).data(), m_mtu, 0);
            else {
                m_send_result = ::send(m_socket, view.substr(count, max - count).data(), max - count, 0);
            }
            count += m_mtu;
            if(m_verbose)
                xstring("Client >> Bytes sent: " + to_xstring(m_send_result)).bold().yellow().reset().print();
        }
    }
    else
    {
        m_send_result = ::send(m_socket, buffer.send.c_str(), buffer.send.size(), 0);
        if(m_verbose)
            xstring("Client >> Bytes sent: " + to_xstring(m_send_result) + '\n').bold().yellow().reset().print();
    }

    if (m_result == SOCKET_ERROR)
    {
        xstring err_str = "! (send) Failed with Error: " + std::to_string(WSAGetLastError()) + '\n';
        closesocket(m_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }
    if(m_verbose)
        xstring("Client >> Bytes sent: " + to_xstring(m_send_result)).bold().yellow().reset().print();
    
    // shutdown the connection since no more data will be sent
    m_result = shutdown(m_socket, SD_SEND);
    if (m_result == SOCKET_ERROR)
    {
        xstring err_str = "! (shutdown) Failed with Error: " + std::to_string(WSAGetLastError()) + '\n';
        closesocket(m_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }
    return *this;
}

Client& Win_Client::recv(int size)
{
    if (size)
        buffer.max_recv = size;

    size_t counter = 0;
    // Receive until the peer closes the connection
    bool full = false;
    do {
        m_relay.clear();
        m_relay.resize(m_mtu);
        m_result = ::recv(m_socket, &m_relay[0], m_mtu, 0);

        if (m_result < 0)
        {
            xstring err_str = "! (recv) Failed with Error: " + std::to_string(WSAGetLastError()) + '\n';
            throw std::runtime_error(err_str);
        }

        if (m_verbose)
        {
            if (m_result > 0)
                xstring("Client >> Bytes received: " + to_xstring(m_result)).bold().yellow().reset().print();
            else if (m_result == 0)
                xstring("Client >> Connection closed\n").bold().yellow().reset().print();
        }

        counter += m_mtu;
        if (!full) 
        {
            if ((!this->buffer.max_recv) || (buffer.max_recv > counter))
                this->buffer.recv += m_relay;
            else {
                this->buffer.recv += m_relay.substr(0, buffer.max_recv - buffer.recv.size());
                full = true;
            }
        }

    } while (m_result > 0);
    return *this;
}

Client& Win_Client::close()
{
    // cleanup
    closesocket(m_socket);
    WSACleanup();

    return *this;
}

#endif