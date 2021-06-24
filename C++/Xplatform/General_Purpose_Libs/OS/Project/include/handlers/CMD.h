#pragma once

#include "xstring.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    #ifdef DLL_EXPORT
       #define EXI __declspec(dllexport)
    #else
       #define EXI __declspec(dllimport)
    #endif
#else
    #define EXI
#endif

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
