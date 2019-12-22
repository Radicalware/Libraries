#pragma once

#include "Server.h"

#ifdef NIX_BASE
class Nix_Server : public Server
{
public:
    Nix_Server();
    virtual void listen(const xstring& port = "");
};
#endif // NIX_BASE

