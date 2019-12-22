#include "Client/Win_Client.h"


Win_Client::Win_Client(int* mtu) : Client(mtu)
{
    m_socket = INVALID_SOCKET;
}

Win_Client::Win_Client(const Win_Client& client) : Client(&client.mtu)
{
    m_socket = client.m_socket;
    m_result = client.m_result;
    
    m_ip = client.m_ip;
    m_port = client.m_port;
    buffer = client.buffer;
}

Win_Client::Win_Client(const Client& client) : Client(&client.mtu)
{
    m_ip = client.ip();
    m_port = client.port();
    buffer = client.buffer;
}

Client& Win_Client::connect()
{
    WSADATA wsaData;
    struct addrinfo* result = nullptr, * ptr = nullptr, addr;

    // Initialize Winsock
    m_result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (m_result != 0)
    {
        xstring err_str = "WSAStartup failed with error: " + std::to_string(m_result) + '\n';
        throw std::runtime_error(err_str);
    }

    ZeroMemory(&addr, sizeof(addr));
    addr.ai_family = AF_UNSPEC;
    addr.ai_socktype = SOCK_STREAM;
    addr.ai_protocol = IPPROTO_TCP;

    // Resolve the server address and port
    m_result = getaddrinfo(m_ip.c_str(), m_port.c_str(), &addr, &result);
    if (m_result != 0)
    {
        xstring err_str = "getaddrinfo(ip, port, addr, result) failed with error: " + std::to_string(m_result) + '\n';
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
            xstring err_str = "socket(ai_family, ai_socktype, ai_protocol) failed with error: " + std::to_string(WSAGetLastError()) + '\n';
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
        xstring err_str = "Unable to connect to server!\n";
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
    
    if (mtu < buffer.send.size())
    {
        size_t count = 0;
        size_t max = buffer.send.size();

        std::string_view view(buffer.send.c_str(), max);
        while (max > count)
        {
            if (mtu + count < max)
                m_send_result = ::send(m_socket, view.substr(count, count + mtu).data(), mtu, 0);
            else {
                m_send_result = ::send(m_socket, view.substr(count, max - count).data(), max - count, 0);
            }
            count += mtu;
            xstring("Client >> Bytes sent: " + to_xstring(m_send_result)).bold().yellow().reset().print();
        }
    }
    else
    {
        m_send_result = ::send(m_socket, buffer.send.c_str(), buffer.send.size(), 0);
        xstring("Client >> Bytes sent: " + to_xstring(m_send_result) + '\n').print();
    }

    if (m_result == SOCKET_ERROR)
    {
        xstring err_str = "send failed with error: " + std::to_string(WSAGetLastError()) + '\n';
        closesocket(m_socket);
        WSACleanup();
        throw std::runtime_error(err_str);
    }
    printf("Client >> Bytes Sent: %ld\n", m_result);
    
    // shutdown the connection since no more data will be sent
    m_result = shutdown(m_socket, SD_SEND);
    if (m_result == SOCKET_ERROR)
    {
        xstring err_str = "shutdown failed with error: " + std::to_string(WSAGetLastError()) + '\n';
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
        m_relay.resize(mtu);
        m_result = ::recv(m_socket, &m_relay[0], mtu, 0);

        counter += mtu;
        if (!full) 
        {
            if ((!this->buffer.max_recv) || (buffer.max_recv > counter))
                this->buffer.recv += m_relay;
            else {
                this->buffer.recv += m_relay.substr(0, buffer.max_recv - buffer.recv.size());
                full = true;
            }
        }


        if (m_result > 0)
            xstring("Client >> Bytes received: " + to_xstring(m_send_result)).bold().yellow().reset().print();

        else if (m_result == 0)
            xstring("Client >> Connection closed\n").bold().yellow().reset().print();
        else {
            xstring err_str = "recv failed with error: " + std::to_string(WSAGetLastError()) + '\n';
            throw std::runtime_error(err_str);
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


