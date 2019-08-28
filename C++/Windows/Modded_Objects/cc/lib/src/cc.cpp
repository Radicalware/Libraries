//! NOT PROGRAMED BY SCOUGE FILE 1/1
//! termcolor from https://raw.githubusercontent.com/ikalnytskyi/termcolor/master/include/termcolor/termcolor.hpp
//! ~~~~~~~~~
//!
//! termcolor is a header-only c++ library for printing colored messages
//! to the terminal. Written just for fun with a help of the Force.
//!
//! :copyright: (c) 2013 by Ihor Kalnytskyi
//! :license: BSD, see LICENSE for details
//!

#include "cc.h"

// the following snippet of code detects the current OS and
// defines the appropriate macro that is used to wrap some
// platform specific things
#if defined(_WIN32) || defined(_WIN64)
#   define TERMCOLOR_OS_WINDOWS
#elif defined(__APPLE__)
#   define TERMCOLOR_OS_MACOS
#elif defined(__unix__) || defined(__unix)
#   define TERMCOLOR_OS_LINUX
#else
#   error unsupported platform
#endif


// This headers provides the `isatty()`/`fileno()` functions,
// which are used for testing whether a standart stream refers
// to the terminal. As for Windows, we also need WinApi funcs
// for changing colors attributes of the terminal.
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
#   include <unistd.h>
#elif defined(TERMCOLOR_OS_WINDOWS)
#   include <io.h>
#   include <windows.h>
#endif

#include <iostream>
#include <cstdio>

// Forward declaration of the `cc::_internal` namespace.
// All comments are below.

// An index to be used to access a private storage of I/O streams. See
// colorize / nocolorize I/O manipulators for details.
static int colorize_index = std::ios_base::xalloc();

FILE* cc::_internal::get_standard_stream(const std::ostream& stream);
bool cc::_internal::is_colorized(std::ostream& stream);
bool cc::_internal::is_atty(const std::ostream& stream);

#if defined(TERMCOLOR_OS_WINDOWS)
    void cc::_internal::win_change_attributes(std::ostream& stream, int foreground, int background);
#endif



std::ostream& cc::colorize(std::ostream& stream)
{
    stream.iword(cc::_internal::colorize_index) = 1L;
    return stream;
}


std::ostream& cc::nocolorize(std::ostream& stream)
{
    stream.iword(cc::_internal::colorize_index) = 0L;
    return stream;
}


std::ostream& cc::reset(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[00m" << "";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream, -1, -1);
#endif
    }
    return stream;
}



std::ostream& cc::bold(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[1m";
#elif defined(TERMCOLOR_OS_WINDOWS)
#endif
    }
    return stream;
}



std::ostream& cc::dark(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[2m";
#elif defined(TERMCOLOR_OS_WINDOWS)
#endif
    }
    return stream;
}



std::ostream& cc::underline(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[4m";
#elif defined(TERMCOLOR_OS_WINDOWS)
#endif
    }
    return stream;
}



std::ostream& cc::blink(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[5m";
#elif defined(TERMCOLOR_OS_WINDOWS)
#endif
    }
    return stream;
}



std::ostream& cc::reverse(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[7m";
#elif defined(TERMCOLOR_OS_WINDOWS)
#endif
    }
    return stream;
}



std::ostream& cc::concealed(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[8m";
#elif defined(TERMCOLOR_OS_WINDOWS)
#endif
    }
    return stream;
}



std::ostream& cc::black(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[30m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream,
            0
        );
#endif
    }
    return stream;
}


std::ostream& cc::red(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[31m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream,
            FOREGROUND_RED
        );
#endif
    }
    return stream;
}


std::ostream& cc::green(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[32m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream,
            FOREGROUND_GREEN
        );
#endif
    }
    return stream;
}


std::ostream& cc::yellow(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[33m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream,
            FOREGROUND_GREEN | FOREGROUND_RED
        );
#endif
    }
    return stream;
}


std::ostream& cc::blue(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[34m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream,
            FOREGROUND_BLUE
        );
#endif
    }
    return stream;
}


std::ostream& cc::magenta(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[35m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream,
            FOREGROUND_BLUE | FOREGROUND_RED
        );
#endif
    }
    return stream;
}


std::ostream& cc::cyan(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[36m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream,
            FOREGROUND_BLUE | FOREGROUND_GREEN
        );
#endif
    }
    return stream;
}


std::ostream& cc::grey(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[37m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream,
            FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED
        );
#endif
    }
    return stream;
}


std::ostream& cc::white(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[39m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream,
            FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED
        );
#endif
    }
    return stream;
}



std::ostream& cc::on_black(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[40m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream, -1,
            0
        );
#endif
    }
    return stream;
}


std::ostream& cc::on_red(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[41m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream, -1,
            BACKGROUND_RED
        );
#endif
    }
    return stream;
}


std::ostream& cc::on_green(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[42m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream, -1,
            BACKGROUND_GREEN
        );
#endif
    }
    return stream;
}


std::ostream& cc::on_yellow(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[43m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream, -1,
            BACKGROUND_GREEN | BACKGROUND_RED
        );
#endif
    }
    return stream;
}


std::ostream& cc::on_blue(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[44m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream, -1,
            BACKGROUND_BLUE
        );
#endif
    }
    return stream;
}


std::ostream& cc::on_magenta(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[45m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream, -1,
            BACKGROUND_BLUE | BACKGROUND_RED
        );
#endif
    }
    return stream;
}


std::ostream& cc::on_cyan(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[46m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream, -1,
            BACKGROUND_GREEN | BACKGROUND_BLUE
        );
#endif
    }
    return stream;
}


std::ostream& cc::on_grey(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[47m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream, -1,
            BACKGROUND_GREEN | BACKGROUND_BLUE | BACKGROUND_RED
        );
#endif
    }

    return stream;
}



std::ostream& cc::on_clear(std::ostream& stream)
{
    if (cc::_internal::is_colorized(stream))
    {
#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
        stream << "\033[49m";
#elif defined(TERMCOLOR_OS_WINDOWS)
        cc::_internal::win_change_attributes(stream, -1,
            BACKGROUND_GREEN | BACKGROUND_BLUE | BACKGROUND_RED
        );
#endif
    }

    return stream;
}

//! Since C++ hasn't a way to hide something in the header from
//! the outer access, I have to introduce this namespace which
//! is used for internal purpose and should't be access from
//! the user code.

//! Since C++ hasn't a true way to extract stream handler
//! from the a given `std::ostream` object, I have to write
//! this kind of hack.

FILE* cc::_internal::get_standard_stream(const std::ostream& stream)
{
    if (&stream == &std::cout)
        return stdout;
    else if ((&stream == &std::cerr) || (&stream == &std::clog))
        return stderr;

    return 0;
}

// Say whether a given stream should be colorized or not. It's always
// true for ATTY streams and may be true for streams marked with
// colorize flag.

bool cc::_internal::is_colorized(std::ostream& stream)
{
    return cc::_internal::is_atty(stream) || static_cast<bool>(stream.iword(colorize_index));
}

//! Test whether a given `std::ostream` object refers to
//! a terminal.

bool cc::_internal::is_atty(const std::ostream& stream)
{
    FILE* std_stream = cc::_internal::get_standard_stream(stream);

    // Unfortunately, fileno() ends with segmentation fault
    // if invalid file descriptor is passed. So we need to
    // handle this case gracefully and assume it's not a tty
    // if standard stream is not detected, and 0 is returned.
    if (!std_stream)
        return false;

#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
    return ::isatty(fileno(std_stream));
#elif defined(TERMCOLOR_OS_WINDOWS)
    return ::_isatty(_fileno(std_stream));
#endif
}

#if defined(TERMCOLOR_OS_WINDOWS)
//! Change Windows Terminal colors attribute. If some
//! parameter is `-1` then attribute won't changed.
void cc::_internal::win_change_attributes(std::ostream& stream, int foreground, int background)
{
    // yeah, i know.. it's ugly, it's windows.
    static WORD defaultAttributes = 0;

    // Windows doesn't have ANSI escape sequences and so we use special
    // API to change Terminal output color. That means we can't
    // manipulate colors by means of "std::stringstream" and hence
    // should do nothing in this case.

    if (!cc::_internal::is_atty(stream))
        return;

    // get terminal handle
    HANDLE hTerminal = INVALID_HANDLE_VALUE;
    if (&stream == &std::cout)
        hTerminal = GetStdHandle(STD_OUTPUT_HANDLE);
    else if (&stream == &std::cerr)
        hTerminal = GetStdHandle(STD_ERROR_HANDLE);

    // save default terminal attributes if it unsaved
    if (!defaultAttributes)
    {
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (!GetConsoleScreenBufferInfo(hTerminal, &info))
            return;
        defaultAttributes = info.wAttributes;
    }

    // restore all default settings
    if (foreground == -1 && background == -1)
    {
        SetConsoleTextAttribute(hTerminal, defaultAttributes);
        return;
    }

    // get current settings
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(hTerminal, &info))
        return;

    if (foreground != -1)
    {
        info.wAttributes &= ~(info.wAttributes & 0x0F);
        info.wAttributes |= static_cast<WORD>(foreground);
    }

    if (background != -1)
    {
        info.wAttributes &= ~(info.wAttributes & 0xF0);
        info.wAttributes |= static_cast<WORD>(background);
    }

    SetConsoleTextAttribute(hTerminal, info.wAttributes);
}
#endif // TERMCOLOR_OS_WINDOWS




#undef TERMCOLOR_OS_WINDOWS
#undef TERMCOLOR_OS_MACOS
#undef TERMCOLOR_OS_LINUX

