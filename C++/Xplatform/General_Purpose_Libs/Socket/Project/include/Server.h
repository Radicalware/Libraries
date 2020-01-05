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
#include <string_view>

#include "Buffer.h"
#include "xstring.h"

class Server
{
    friend class Socket;

protected:
    int m_result = 0;

    xstring m_ip;
    xstring m_port;

    std::function<xstring()> m_method;
    xstring m_relay;

    int& m_mtu;
    bool& m_verbose;
    int m_pro;

public:

    Buffer buffer;

    template<typename F, typename... A>
    void set_function(F&& Function, A&&... Args);

    Server(int* mtu, bool* verbose, int pro);
    virtual Server& listen(const xstring& port) = 0;
    virtual Server& accept() = 0;
    virtual Server& recv(int size = 0) = 0; // derrived
    virtual Server& respond() = 0; // derrived
    virtual Server& close() = 0; 
};


template<typename F, typename ...A>
inline void Server::set_function(F&& Function, A&& ...Args)
{
    //m_method = Function(&this->buffer.recv, Args...);
    m_method = std::bind(Function, &this->buffer.recv, Args...);
}