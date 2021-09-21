#pragma once

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))

#include "Server.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#pragma comment (lib, "Ws2_32.lib")

// This Code is based off of Microsoft's Approved and Tested Client/Server Methods
// https://docs.microsoft.com/en-us/windows/win32/winsock/complete-client-code
// https://docs.microsoft.com/en-us/windows/win32/winsock/complete-server-code

class Win_Server : public Server
{
    SOCKET m_listen_socket;
    SOCKET m_client_socket;
    
public:
    Win_Server(int* mtu, bool* verbose, int pro);

    virtual Server& Listen(const xstring& port);
    virtual Server& Accept();
    virtual Server& Recv(int size = 0);
    virtual Server& Respond();
    virtual Server& Close();
};

// =====================================================================================

#endif


