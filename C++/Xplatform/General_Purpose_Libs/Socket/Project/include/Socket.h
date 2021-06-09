#pragma once

/*
* Copyright[2019][Joel Leagues aka Scourge]
* Scourge /at\ protonmail /dot\ com
* www.radicalware.net
* https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http ://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

// -----------------------------
// Support Libs
#include "xstring.h"
// -----------------------------
// Project Libs
#include "Buffer.h"
#include "Server.h"
#include "Client.h"
// -----------------------------
// STD Libs
#include<string>
// -----------------------------
// Networking Libs

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#ifndef WIN_BASE
#define WIN_BASE
#endif
#include "Client/Win_Client.h"
#include "Server/Win_Server.h"
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#else // --------------
#ifndef NIX_BASE
#define NIX_BASE
#endif
#ifndef __unix__
#define __unix__
#endif
#include "Client/Nix_Client.h"
#include "Server/Nix_Server.h"
#include<sys/socket.h>
#include<sys/wait.h>
#include<sys/types.h>
#include<arpa/inet.h>
#include<stdio.h>
#include<string.h>
#include<unistd.h>
#include<stdlib.h>
#endif
// -----------------------------

class client;
class Server;

class Socket
{
public:
    enum class Protocol
    {
        RAW, // Not Built Yet
        TCP,
        UDP  // Not Build Yet
    };
    enum class Start
    {
        None,
        Client,
        Server,
        Both
    };
private:
    Socket& self;
    Protocol m_pro = Protocol::TCP;

    // --------------------------------
    short int m_offset_count = 2;
    size_t m_offset_client = 0;
    size_t m_offset_server = 0;
    // --------------------------------

    void InitSockets(Start start);

public:
    Client& client;
    Server& server;

    bool verbose = false;
    static int MTU;

    Socket(Start start = Start::Client, Protocol protocol = Protocol::TCP);
    Socket(const Socket& other);
    ~Socket();

    void SetProtocol(Socket::Protocol pro);

    Socket& InitServer();
    Socket& Initclient();

    void operator=(const Socket& other);
    void operator=(Client& other);
    void operator=(Server& other);
};
