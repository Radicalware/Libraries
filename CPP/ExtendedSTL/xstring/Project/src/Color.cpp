
#include "Color.h"
#include "xstring.h"

void Color::Reset()
{
    printf("\033[00m");
}


xstring Color::Black    = BLACK;
xstring Color::Red      = RED;
xstring Color::Green    = GREEN;
xstring Color::Yellow   = YELLOW;
xstring Color::Blue     = BLUE;
xstring Color::Magenta  = MAGENTA;
xstring Color::Cyan     = CYAN;
xstring Color::Grey     = GREY;
xstring Color::White    = WHITE;

xstring Color::On::Black    = ON_BLACK;
xstring Color::On::Red      = ON_RED;
xstring Color::On::Green    = ON_GREEN;
xstring Color::On::Yellow   = ON_YELLOW;
xstring Color::On::Blue     = ON_BLUE;
xstring Color::On::Magenta  = ON_MAGENTA;
xstring Color::On::Cyan     = ON_CYAN;
xstring Color::On::Grey     = ON_GREY;
xstring Color::On::White    = ON_WHITE;


xstring Color::Mod::Reset     = RESET;
xstring Color::Mod::Bold      = BOLD;
xstring Color::Mod::Underline = UNDERLINE;
xstring Color::Mod::Reverse   = REVERSE;

// Operates only on Linux 
xstring Color::Mod::Dark  = DARK;
xstring Color::Mod::Blink = BLINK;
xstring Color::Mod::Hide  = HIDE;