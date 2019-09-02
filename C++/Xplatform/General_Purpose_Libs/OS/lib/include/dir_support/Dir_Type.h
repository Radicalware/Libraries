#pragma once


#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#define WIN_BASE
#else
#define NIX_BASE
#endif

#include "xstring.h"

#include <string>

class Dir_Type
{
protected:
    enum dir_type
    {
        os_none,
        os_file,
        os_directory
    };

    const unsigned char isFile =0x8;
    const unsigned char isFolder =0x4;

public:
    Dir_Type();
    ~Dir_Type();

    dir_type has(const xstring& item);
	bool file(const xstring& file);
	bool directory(const xstring& folder);
	bool file(xstring&& file);
	bool directory(xstring&& folder);
    xstring dir_item_str(const xstring& item);

    
    xstring bpwd(); // binary pwd
    xstring pwd();  // user pwd
    xstring home(); // home dir
};

