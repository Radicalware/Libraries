#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <climits>

#include "../dir_support/Dir_Type.h"
#include "xstring.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#define WIN_BASE
#else
#define NIX_BASE
#endif

class OS;
namespace OS_O // OS Object
{
    class File
    {
        friend class ::OS;

        xstring m_name;
        xstring m_data;
        std::ofstream m_out_stream; // output new file data
        std::ifstream m_in_stream;  // intake file data to var
        char m_handler = 'n'; // 'r' = read    'w' = write    'a' = append

        void set_file(const xstring& iname);

    public:
        File();
        File(const File& file);
        File(const xstring& iname);
        void operator=(const File& file);

        xstring name() const;
        xstring data() const;

        void set_read();
        void set_write();
        void set_append();

        void close();
        void clear();

        void remove();
        void rm();

        void copy(const xstring& location);
        void cp(const xstring& location);

        void move(const xstring& location);
        void mv(const xstring& location);

        static xstring full_path(const xstring& ipath);
    };
};