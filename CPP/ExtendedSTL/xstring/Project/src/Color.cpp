
#include "Color.h"
#include "xstring.h"

void Color::Reset()
{
    printf("\033[00m");
}


const xstring Color::Black    = BLACK;
const xstring Color::Red      = RED;
const xstring Color::Green    = GREEN;
const xstring Color::Yellow   = YELLOW;
const xstring Color::Blue     = BLUE;
const xstring Color::Magenta  = MAGENTA;
const xstring Color::Cyan     = CYAN;
const xstring Color::Grey     = GREY;
const xstring Color::White    = WHITE;

const xstring Color::On::Black    = ON_BLACK;
const xstring Color::On::Red      = ON_RED;
const xstring Color::On::Green    = ON_GREEN;
const xstring Color::On::Yellow   = ON_YELLOW;
const xstring Color::On::Blue     = ON_BLUE;
const xstring Color::On::Magenta  = ON_MAGENTA;
const xstring Color::On::Cyan     = ON_CYAN;
const xstring Color::On::Grey     = ON_GREY;
const xstring Color::On::White    = ON_WHITE;


const xstring Color::Mod::Reset     = RESET;
const xstring Color::Mod::Bold      = BOLD;
const xstring Color::Mod::Underline = UNDERLINE;
const xstring Color::Mod::Reverse   = REVERSE;

// Operates only on Linux 
const xstring Color::Mod::Dark  = DARK;
const xstring Color::Mod::Blink = BLINK;
const xstring Color::Mod::Hide  = HIDE;
