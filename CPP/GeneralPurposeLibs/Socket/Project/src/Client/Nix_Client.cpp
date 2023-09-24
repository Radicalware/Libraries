
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#ifdef __unix__

#include "Client/Nix_Client.h"


Nix_Client::Nix_Client(int* mtu, bool* verbose, int pro): Client(mtu, verbose, pro)
{
    m_net.mtu = &m_mtu;
}

Nix_Client::Nix_Client(const Nix_Client& other) : Client(other)
{
    m_net = other.m_net;
    m_result = other.m_result;
}

Nix_Client::Nix_Client(const Client& other) : Client(other)
{
}

Client& Nix_Client::Connect()
{
    // Initialize Socket
    m_net.connect = socket(AF_INET, SOCK_STREAM, 0);
    
    if (m_net.connect < 0)
    {
        xstring err_str = "! Client: Creating Socket Failed with Error: " + ToXString(m_result) + '\n';
        throw std::runtime_error(err_str);
    }

    memset(&m_net.addr, 0, sizeof(m_net.addr));

    m_net.addr.sin_family = AF_INET;
    inet_pton(AF_INET, m_ip.c_str(), &m_net.addr.sin_addr);
    m_net.addr.sin_port = htons((short unsigned int)(atoi(m_port.c_str())));


    m_result = ::connect(m_net.connect, (struct sockaddr*) &m_net.addr, sizeof(m_net.addr));
    if (m_result != 0)
    {
        xstring err_str = "! Client: Connecting the Socket Failed with Error: " + ToXString(m_result) + '\n';
        throw std::runtime_error(err_str);
    }    
    m_net.connected = true;
    return *this;
}

Client& Nix_Client::Connect(const xstring& ip, const xstring& port)
{
    m_ip = ip;
    m_port = port;
    this->Connect();
    return *this;
}

Client& Nix_Client::Send(const xstring& buff)
{
    buffer.send += buff;
    
    size_t high_seg = 0;
    size_t low_seg = 0;
    size_t inc = 0;
    if (m_mtu < buffer.send.size())
    {
        size_t max = buffer.send.size();

        std::string_view view(buffer.send.c_str(), max);
        while (max > low_seg)
        {
            high_seg = low_seg + m_mtu;
            if (max > high_seg)
            {
                inc = high_seg - low_seg;
            }else 
            {
                inc = max - low_seg;                
            }
            m_result = m_net.Send(view.substr(low_seg, inc).data());

            if (m_verbose)
                xstring(xstring("Client >> Bytes sent: ") + ToXString(inc)).ToBold().ToYellow().ResetColor().Print();
            low_seg += m_mtu;
        }
    }
    else
    {
        m_result = m_net.Send(buffer.send.c_str());

        if (m_verbose)
            xstring("Client >> Bytes sent: " + ToXString(buffer.send.size()) + '\n').ToBold().ToYellow().ResetColor().Print();
    }
    m_net.Send(""); // ensures that at least one packet will be less than the m_mtu to break the recv loop
    return *this;
}

Client& Nix_Client::Recv(int size)
{
    if (size)
        buffer.max_recv = size;

    size_t counter = 0;
    // Receive until the peer closes the connection
    bool full = false;
    do {       
        m_result = m_net.Recv(m_relay);
        
        if (m_result < 0)
        {
            xstring err_str = "! Client: Recv Failed with Error: " + ToXString(m_relay.size()) + '\n';
            throw std::runtime_error(err_str);
        }

        if (m_verbose)
        {
            if (m_result > 0)
                xstring("Client >> Bytes received: " + ToXString(m_result)).ToBold().ToYellow().ResetColor().Print();
            else if (m_result == 0)
                xstring("Client >> Connection closed\n").ToBold().ToYellow().ResetColor().Print();
        }

        counter += m_mtu;
        if (!full) 
        {
            if ((!this->buffer.max_recv) || (buffer.max_recv > counter)){
                this->buffer.recv += m_relay;
            }else {
                this->buffer.recv += m_relay.substr(0, buffer.max_recv - buffer.recv.size());
                full = true;
            }
        }
        
    } while (m_result > 1);
    return *this;
}

Client& Nix_Client::Close()
{
    m_result = ::close(m_net.connect);
    if (m_result < 0)
    {
        xstring err_str("! Client: Failed to close 'm_net.connect' with Error: " + ToXString(m_result) + '\n');
        throw std::runtime_error(err_str);
    }
    
    m_net.connected = false;
    return *this;
}


#endif
