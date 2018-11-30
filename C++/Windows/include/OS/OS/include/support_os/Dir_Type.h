#pragma once

#include<string>



#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#define WIN_BASE
#else
#define NIX_BASE
#endif


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

    dir_type has(const std::string& item);
    bool file(const std::string& file);
    bool directory(const std::string& folder);
    std::string dir_item_type(const std::string& item);
};

