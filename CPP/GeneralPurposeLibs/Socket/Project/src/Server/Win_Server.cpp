
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Server/Win_Server.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))

Win_Server::Win_Server(int* mtu, bool* verbose, int pro): Server(mtu, verbose, pro)
{
    m_listen_socket = INVALID_SOCKET;
    m_client_socket = INVALID_SOCKET;
}

Server& Win_Server::Listen(const xstring& port)
{
    if (port != "")
        m_port = port;

    WSADATA wsaData;

    struct addrinfo* addr_data = NULL;
    struct addrinfo addr;

    // Initialize Winsock
    m_result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (m_result != 0)
    {
        xstring err_str("! (WSAStartup) Failed with Error: " + RA::ToXString(m_result) + '\n');
        throw std::runtime_error(err_str);
    }

    ZeroMemory(&addr, sizeof(addr));
    addr.ai_family = AF_INET;
    
    if (m_pro == 1)
        addr.ai_socktype = SOCK_STREAM;
    else if (m_pro == 2)
        addr.ai_socktype = SOCK_DGRAM;
    else
        throw std::runtime_error("! Windows RAW Ports not Programmed yet\n");

    addr.ai_protocol = IPPROTO_TCP;
    addr.ai_flags = AI_PASSIVE;

    // Resolve the server address and port
    m_result = getaddrinfo(NULL, m_port.c_str(), &addr, &addr_data);
    if (m_result != 0)
    {
        xstring err_str("! (getaddrinfo) Failed with Error: " + RA::ToXString(m_result) + '\n');
        WSACleanup();
        throw std::runtime_error(err_str);
    }

    // Create a SOCKET for connecting to server
    m_listen_socket = socket(addr_data->ai_family, addr_data->ai_socktype, addr_data->ai_protocol);
    if (m_listen_socket == INVALID_SOCKET)
    {
        xstring err_str("! (socket) Failed with Error: " + RA::ToXString(WSAGetLastError()) + '\n');
        freeaddrinfo(addr_data);
        WSACleanup();
        throw std::runtime_error(err_str);
    }

    // Setup the TCP listening socket
    m_result = bind(m_listen_socket, addr_data->ai_addr, (int)addr_data->ai_addrlen);
    if (m_result == SOCKET_ERROR)
    {
        xstring err_str("! (bind) Failed with Error: " + RA::ToXString(WSAGetLastError()) + '\n');
        freeaddrinfo(addr_data);
        closesocket(m_listen_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }
    freeaddrinfo(addr_data);


    m_result = ::listen(m_listen_socket, SOMAXCONN);
    if (m_result == SOCKET_ERROR)
    {
        xstring err_str("! (listen) Failed with Error: " + RA::ToXString(WSAGetLastError()) + '\n');
        closesocket(m_listen_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }
    return *this;
}


Server& Win_Server::Accept()
{
    // Accept a client socket
    m_client_socket = ::accept(m_listen_socket, NULL, NULL);
    if (m_client_socket == INVALID_SOCKET)
    {
        xstring err_str("! (accept) Failed with Error: " + RA::ToXString(WSAGetLastError()) + '\n');
        closesocket(m_listen_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }
    // No longer need server socket
    closesocket(m_listen_socket);
    return *this;
}

Server& Win_Server::Recv(int size)
{
    if (size)
        buffer.max_recv = size;

    buffer.recv.resize(buffer.max_recv + 1);

    // Receive and React until the peer shuts down the connection
    buffer.recv.clear();
    m_relay.resize(m_mtu);
    int bytes_left = this->buffer.max_recv;
    do {
        m_result = ::recv(m_client_socket, &m_relay[0], this->m_mtu, 0);

        if (!this->buffer.max_recv)
        {
            if (m_result == this->m_mtu)
                buffer.recv += m_relay;
            else
                buffer.recv += m_relay.substr(0, m_result);
        }
        else
        {
            if (m_result <= bytes_left)
            {
                if (m_result == this->m_mtu)
                    buffer.recv += m_relay;
                else {
                    buffer.recv += m_relay.substr(0, m_result);
                    bytes_left -= m_result;
                }
            }
            else {
                buffer.recv += m_relay.substr(0, bytes_left);
                m_relay.clear();
                break;
            }
        }

        m_relay.clear();
        m_relay.resize(m_mtu);

        if (m_result > 0)
        {

            if (m_verbose)
                xstring("Server >> Bytes received: " + RA::ToXString(m_result)).ToBold().ToRed().ResetColor().Print();

            if (m_result == SOCKET_ERROR)
            {
                xstring err_str("send Failed with Error: " + RA::ToXString(WSAGetLastError()) + '\n');
                closesocket(m_listen_socket);
                WSACleanup();
                throw std::runtime_error(err_str);
            }
        }
        else if (m_result == 0) 
        {
            if (m_verbose)
                xstring("Server >> Connection closing...").ToBold().ToRed().ResetColor().Print();
        }
        else {
            xstring err_str("! (send) Failed with Error: " + RA::ToXString(WSAGetLastError()) + '\n');
            closesocket(m_listen_socket);
            WSACleanup();
            throw std::runtime_error(err_str);
        }

    } while (m_result > 0);
    return *this;
}


Server& Win_Server::Respond()
{
    buffer.send.clear();
    this->buffer.send = m_method();

    if (m_mtu < buffer.send.size())
    {
        size_t count = 0;
        size_t max = buffer.send.size();

        std::string_view view(buffer.send.c_str(), max);
        while (max > count)
        {
            if(m_mtu + count < max)
                m_result = send(m_client_socket, view.substr(count, count + m_mtu).data(), m_mtu, 0);
            else {
                m_result = send(m_client_socket, view.substr(count, max - count).data(), max - count, 0);
            }
            count += m_mtu;

            if(m_verbose)
                xstring("Server >> Bytes sent: " + RA::ToXString(m_result)).ToBold().ToRed().ResetColor().Print();
        }
    }
    else
    {
        m_result = send(m_client_socket, buffer.send.c_str(), buffer.send.size(), 0);

        if (m_verbose)
            xstring("Server >> Bytes sent: " + RA::ToXString(m_result) + '\n').ToBold().ToRed().ResetColor().Print();
    }

    if (m_result == SOCKET_ERROR)
    {
        xstring err_str("! (send) Failed with Error: " + RA::ToXString(WSAGetLastError()) + '\n');
        closesocket(m_listen_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }

    return *this;
}

Server& Win_Server::Close()
{
    // shutdown the connection since we're done
    m_result = shutdown(m_client_socket, SD_SEND);
    if (m_result == SOCKET_ERROR)
    {
        xstring err_str("! (send) Failed with Error: " + RA::ToXString(WSAGetLastError()) + '\n');
        closesocket(m_client_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }

    // cleanup
    closesocket(m_client_socket);
    WSACleanup();

    return *this;
}

#endif