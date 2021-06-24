#pragma once

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <climits>

#include "dir_support/Dir_Type.h"
#include "xstring.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#define WIN_BASE
#else
#define NIX_BASE
#endif

class EXI OS;
namespace OS_O // OS Object
{
    class EXI File : public Dir_Type
    {
        friend class ::OS;

        xstring m_name;
        xstring m_data;
        std::ofstream m_out_stream; // output new file data
        std::ifstream m_in_stream;  // intake file data to var
        char m_handler = 'n'; // 'r' = read    'w' = write    'a' = append

        static RE2 s_get_file;
        static RE2 s_forwardslash;

        void SetFile(const xstring& iname);

    public:
        File();
        File(const File& file);
        File(const xstring& iname);
        void operator=(const File& file);

        xstring GetName() const;
        xstring GetData() const;

        void SetRead();
        void SetWrite();
        void SetAppend();

        void Close();
        void Clear();

        void Remove();
        void RM();

        void Copy(const xstring& location);
        void CP(const xstring& location);

        void Move(const xstring& location);
        void MV(const xstring& location);
    };
};
