#pragma once

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#define WIN_BASE
#else
#define NIX_BASE
#endif

#include "dir_support/Dir_Type.h"
#include "xstring.h"

#include<string>
#include<limits.h>
#include<stdlib.h>


class File_Names : public Dir_Type
{
private:

#ifdef WIN_BASE
    char full[_MAX_PATH];
#else
    char full[PATH_MAX];
#endif

    xstring m_traverse_target;

    xstring m_old;
    xstring m_target;


    bool m_rexit;
    // Checking for file input consumes a lot of time
    // Set it true for when users input data.
    // Set it false for any other time.
    // ex: os.set_file_regex(true);
    // to turn it on.

public:
    File_Names(bool rexit);
    ~File_Names() {};

    File_Names(bool rexit, const xstring& i_target);
    File_Names(bool rexit, const xstring& i_old, const xstring& i_target);

    void set_old();
    void set_target();
    void set_old(xstring& m_old);
    void set_target(xstring& m_target);

    void fix_slash(xstring& item);

    void assert_folder_syntax(const xstring& folder1, const xstring& folder2 = "");

    const xstring old() const;
    const xstring target() const;
    const xstring traverse_target() const;
};
