#pragma once

/*
* Copyright[2019][Joel Leagues aka Scourge]
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.net
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
#include <stdexcept>

#include "Buffer.h"
#include "xstring.h"

class Client
{
    friend class Socket;

protected:
    xstring m_ip;
    xstring m_port;
    xstring m_relay;

    int m_send_result;

    int&  m_mtu;
    bool& m_verbose;
    int m_pro;

public:
    Buffer buffer;

    Client(int* mtu, bool* verbose, int pro);
    Client(const Client& other);
    virtual Client& Connect() = 0;
    virtual Client& Connect(const xstring& ip, const xstring& port) = 0;
    virtual Client& Send(const xstring& buff = "") = 0;
    virtual Client& Recv(int size = 0) = 0;
    virtual Client& Close() = 0;

    virtual void operator<<(const xstring& buff);
    virtual void operator=(const Client& other);

    xstring GetIP() const;
    xstring GetPort() const;
};