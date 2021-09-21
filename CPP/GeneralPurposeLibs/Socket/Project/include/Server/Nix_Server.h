#pragma once

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#ifdef __unix__

#include "Server.h"
#include "NixNet.h"

class Nix_Server : public Server
{
    NixNet m_net;
    int m_result;

public:
    Nix_Server(int* mtu, bool* verbose, int pro);

    virtual Server& Listen(const xstring& port);
    virtual Server& Accept();
    virtual Server& Recv(int size = 0);
    virtual Server& Respond();
    virtual Server& Close();
};

#endif

