#pragma once

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#ifdef __unix__

#include "Client.h"
#include "NixNet.h"

class Nix_Client : public Client
{
    NixNet m_net;
    
    int m_result = 0;

public:
    Nix_Client(int* mtu, bool* verbose, int pro);
    Nix_Client(const Nix_Client& other);
    Nix_Client(const Client& other);

    virtual Client& connect();
    virtual Client& connect(const xstring& ip, const xstring& port);
    virtual Client& send(const xstring& buff = "");
    virtual Client& recv(int size = 0);
    virtual Client& close();

};

#endif