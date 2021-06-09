#pragma once


#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#define WIN_BASE
#else
#define NIX_BASE
#endif

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    #ifdef DLL_EXPORT
       #define EXI __declspec(dllexport)
    #else
       #define EXI __declspec(dllimport)
    #endif
#else
    #define EXI
#endif

#include "xstring.h"

namespace OS_O // OS Object
{
    class EXI Dir_Type
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

        static re2::RE2 s_backslash;
        static re2::RE2 s_back_n_forward_slashes;
        static re2::RE2 s_forwardslash;

    public:
        Dir_Type();
        ~Dir_Type();

        static DT GetDirType(const xstring& item);
        static bool Has(const xstring& item);
        static bool HasFile(const xstring& file);
        static bool HasDir(const xstring& folder);
        static bool HasFile(xstring&& file);
        static bool HasDir(xstring&& folder);
        static xstring GetDirItem(const xstring& item);


        static xstring BWD(); // binary pwd
        static xstring PWD();  // user pwd
        static xstring Home(); // home dir

        static xstring FullPath(const xstring& file);
    };
};
