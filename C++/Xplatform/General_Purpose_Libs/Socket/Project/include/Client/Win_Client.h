#pragma once

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))

#include "Client.h"

#define WIN32_LEAN_AND_MEAN

#include <iostream>
#include <string>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>

// This Code is based off of Microsoft's Approved and Tested Client/Server Methods
// https://docs.microsoft.com/en-us/windows/win32/winsock/complete-client-code
// https://docs.microsoft.com/en-us/windows/win32/winsock/complete-server-code


// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")



class Win_Client : public Client
{
    SOCKET m_socket;
    int m_result = 0;

public:
    Win_Client(int* mtu, bool* verbose, int pro);
    Win_Client(const Win_Client& other);
    Win_Client(const Client& other);

    virtual Client& connect();
    virtual Client& connect(const xstring& ip, const xstring& port);
    virtual Client& send(const xstring& buff = "");
    virtual Client& recv(int size = 0);
    virtual Client& close();

};

#endif