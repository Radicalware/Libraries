#pragma once


#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#define WIN_BASE
#else
#define NIX_BASE
#endif

#include "xstring.h"

namespace OS_O // OS Object
{
    class Dir_Type
    {
    protected:
        enum class DT // dir type
        {
            none,
            file,
            directory
        };

        static const unsigned char IsFile;
        static const unsigned char IsFolder;

    public:
        Dir_Type();
        ~Dir_Type();

        static DT Get_Dir_Type(const xstring& item);
        static bool Has(const xstring& item);
        static bool Has_File(const xstring& file);
        static bool Has_Dir(const xstring& folder);
        static bool Has_File(xstring&& file);
        static bool Has_Dir(xstring&& folder);
        static xstring Dir_Item_Str(const xstring& item);


        static xstring BWD(); // binary pwd
        static xstring PWD();  // user pwd
        static xstring Home(); // home dir
    };
};
