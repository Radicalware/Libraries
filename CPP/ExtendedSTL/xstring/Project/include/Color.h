﻿#pragma once

#ifndef __COLORS__
#define __COLORS__

#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define GREY    "\033[37m"
#define WHITE   "\033[39m"

#define ON_BLACK   "\033[40m"
#define ON_RED     "\033[41m"
#define ON_GREEN   "\033[42m"
#define ON_YELLOW  "\033[43m"
#define ON_BLUE    "\033[44m"
#define ON_MAGENTA "\033[45m"
#define ON_CYAN    "\033[46m"
#define ON_GREY    "\033[47m"
#define ON_WHITE   "\033[49m"

#define RESET       "\033[00m"
#define BOLD        "\033[01m"
#define UNDERLINE   "\033[04m"
#define REVERSE     "\033[07m"

// Linux Only
#define DARK        "\033[02m"
#define BLINK       "\033[05m"
#define HIDE        "\033[08m"

class xstring;

struct Color
{
    static xstring Black;
    static xstring Red;
    static xstring Green;
    static xstring Yellow;
    static xstring Blue;
    static xstring Magenta;
    static xstring Cyan;
    static xstring Grey;
    static xstring White;

    struct On
    {
        static xstring Black;
        static xstring Red;
        static xstring Green;
        static xstring Yellow;
        static xstring Blue;
        static xstring Magenta;
        static xstring Cyan;
        static xstring Grey;
        static xstring White;
    };

    struct Mod
    {
        static xstring Reset;
        static xstring Bold;
        static xstring Underline;
        static xstring Reverse;

        // Works only on Linux
        static xstring Dark;
        static xstring Blink;
        static xstring Hide;
    };

    static void Reset();
};

#endif