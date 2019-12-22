#include <iostream>

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
