#pragma once

#include "Macros.h"

class EXI OS;
namespace OS_O // OS Object
{
    class EXI CMD
    {
        friend class ::OS;

        xstring m_cmd;
        xstring m_out;
        xstring m_err;
        xstring m_err_message;

    public:
        CMD();
        CMD(const CMD& cmd);
        void operator=(const CMD& cmd);
        xstring GetCommand() const;
        xstring GetOutput() const;
        xstring GetError() const;
        xstring GetErrorMessage() const;
    };
};
