#pragma once

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))

#include "Server.h"

// This Code is based off of Microsoft's Approved and Tested Client/Server Methods
// https://docs.microsoft.com/en-us/windows/win32/winsock/complete-client-code
// https://docs.microsoft.com/en-us/windows/win32/winsock/complete-server-code

class Win_Server : public Server
{
public:
    Win_Server(int* mtu);

    virtual Server& listen(const xstring& port);
    virtual Server& accept();
    virtual Server& recv(int size = 0);
    virtual Server& respond();
    virtual Server& close();
};

// =====================================================================================

#endif


