
#include "Server/Win_Server.h"



#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))

Win_Server::Win_Server(int* mtu): Server(mtu)
{
    m_listen_socket = INVALID_SOCKET;
    m_client_socket = INVALID_SOCKET;
}

Server& Win_Server::listen(const xstring& port)
{
    if (port != "")
        m_port = port;

    WSADATA wsaData;

    struct addrinfo* result = NULL;
    struct addrinfo hints;

    // Initialize Winsock
    m_result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (m_result != 0)
    {
        xstring err_str("WSAStartup failed with error: " + to_xstring(m_result) + '\n');
        throw std::runtime_error(err_str);
    }

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    // Resolve the server address and port
    m_result = getaddrinfo(NULL, m_port.c_str(), &hints, &result);
    if (m_result != 0)
    {
        xstring err_str("getaddrinfo failed with error: " + to_xstring(m_result) + '\n');
        WSACleanup();
        throw std::runtime_error(err_str);
    }

    // Create a SOCKET for connecting to server
    m_listen_socket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (m_listen_socket == INVALID_SOCKET)
    {
        xstring err_str("socket failed with error: " + to_xstring(WSAGetLastError()) + '\n');
        freeaddrinfo(result);
        WSACleanup();
        throw std::runtime_error(err_str);
    }

    // Setup the TCP listening socket
    m_result = bind(m_listen_socket, result->ai_addr, (int)result->ai_addrlen);
    if (m_result == SOCKET_ERROR)
    {
        xstring err_str("bind failed with error: " + to_xstring(WSAGetLastError()) + '\n');
        freeaddrinfo(result);
        closesocket(m_listen_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }
    freeaddrinfo(result);


    m_result = ::listen(m_listen_socket, SOMAXCONN);
    if (m_result == SOCKET_ERROR)
    {
        xstring err_str("listen failed with error: " + to_xstring(WSAGetLastError()) + '\n');
        closesocket(m_listen_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }
    return *this;
}


Server& Win_Server::accept()
{
    // Accept a client socket
    m_client_socket = ::accept(m_listen_socket, NULL, NULL);
    if (m_client_socket == INVALID_SOCKET)
    {
        xstring err_str("accept failed with error: " + to_xstring(WSAGetLastError()) + '\n');
        closesocket(m_listen_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }
    // No longer need server socket
    closesocket(m_listen_socket);
    return *this;
}

Server& Win_Server::recv(int size)
{
    if (size)
        buffer.max_recv = size;

    buffer.recv.resize(buffer.max_recv + 1);

    // Receive and React until the peer shuts down the connection
    buffer.recv.clear();
    m_relay.resize(mtu);
    int bytes_left = this->buffer.max_recv;
    do {
        m_result = ::recv(m_client_socket, &m_relay[0], this->mtu, 0);

        if (!this->buffer.max_recv)
        {
            if (m_result == this->mtu)
                buffer.recv += m_relay;
            else
                buffer.recv += m_relay.substr(0, m_result);
        }
        else
        {
            if (m_result <= bytes_left)
            {
                if (m_result == this->mtu)
                    buffer.recv += m_relay;
                else {
                    buffer.recv += m_relay.substr(0, m_result);
                    bytes_left -= m_result;
                }
            }
            else {
                buffer.recv += m_relay.substr(0, bytes_left);
                bytes_left = 0;
                m_relay.clear();
                break;
            }
        }

        m_relay.clear();
        m_relay.resize(mtu);

        if (m_result > 0)
        {
            xstring("Server >> Bytes received: " + to_xstring(m_result)).bold().red().reset().print();

            // Echo the buffer back to the sender
            //m_send_result = send(m_client_socket, &buffer.send[0], buffer.send.size(), 0);
            if (m_send_result == SOCKET_ERROR)
            {
                xstring err_str("send failed with error: " + to_xstring(WSAGetLastError()) + '\n');
                closesocket(m_listen_socket);
                WSACleanup();
                throw std::runtime_error(err_str);
            }
        }
        else if (m_result == 0)
            printf("Server >> Connection closing...\n");
        else {
            xstring err_str("send failed with error: " + to_xstring(WSAGetLastError()) + '\n');
            closesocket(m_listen_socket);
            WSACleanup();
            throw std::runtime_error(err_str);
        }

    } while (m_result > 0);
    return *this;
}


Server& Win_Server::respond()
{
    buffer.send.clear();
    this->buffer.send = m_method();

    if (mtu < buffer.send.size())
    {
        size_t count = 0;
        size_t max = buffer.send.size();

        std::string_view view(buffer.send.c_str(), max);
        while (max > count)
        {
            if(mtu + count < max)
                m_send_result = send(m_client_socket, view.substr(count, count + mtu).data(), mtu, 0);
            else {
                m_send_result = send(m_client_socket, view.substr(count, max - count).data(), max - count, 0);
            }
            count += mtu;
            xstring("Server >> Bytes sent: " + to_xstring(m_send_result)).bold().red().reset().print();
        }
    }
    else
    {
        m_send_result = send(m_client_socket, buffer.send.c_str(), buffer.send.size(), 0);
        xstring("Server >> Bytes sent: " + to_xstring(m_send_result) + '\n').bold().red().reset().print();
    }

    if (m_send_result == SOCKET_ERROR)
    {
        xstring err_str("send failed with error: " + to_xstring(WSAGetLastError()) + '\n');
        closesocket(m_listen_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }

    return *this;
}

Server& Win_Server::close()
{
    // shutdown the connection since we're done
    m_result = shutdown(m_client_socket, SD_SEND);
    if (m_result == SOCKET_ERROR)
    {
        xstring err_str("send failed with error: " + to_xstring(WSAGetLastError()) + '\n');
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