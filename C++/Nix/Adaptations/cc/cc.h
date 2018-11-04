
// v1.0.0 --(MOD)-- Console Colors
// A mod on termcolor.h

// Original was called termcolor.h by Ihor Kalnytskyi copyright: (c) 2013 with BSD LICENCE
// https://raw.githubusercontent.com/ikalnytskyi/termcolor/master/include/termcolor/termcolor.hpp

/*
* Copyright[2018][Ihor Kalnytskyi & modded by Scourge]
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
* https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http ://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

// Modifications by Scourge 
// 1. on_grey and grey are now black representatives
// 2. the real grey and on_grey are actually grey
// 3. white and on_clear are now added
// 4. all functionality was moved to the .ccp file
//    The original had everything in the header file which
//    made it more difficult to see all of it's features at a glance.
// 5. All comments were removed form the .h file and palced into the .cpp file
// 6. Fixed a bug where the background color would not be cleared (cc::reset)
//    if a newline was used right after the call;
//    new line was uesd 
// 7. termcolor was renamed to cc (console color) becaues 
//    cc takes up less space and is quicker to type than termcolor.
//    Bit ghanks to "Ihor Kalnytskyi" who did a greate job!!


#pragma once

#if defined(_WIN32) || defined(_WIN64)
#   define TERMCOLOR_OS_WINDOWS
#elif defined(__APPLE__)
#   define TERMCOLOR_OS_MACOS
#elif defined(__unix__) || defined(__unix)
#   define TERMCOLOR_OS_LINUX
#else
#   error unsupported platform
#endif


#if defined(TERMCOLOR_OS_MACOS) || defined(TERMCOLOR_OS_LINUX)
#   include <unistd.h>
#elif defined(TERMCOLOR_OS_WINDOWS)
#   include <io.h>
#   include <windows.h>
#endif

#include <iostream>
#include <cstdio>

namespace cc
{
    namespace _internal
    {
        static int colorize_index = std::ios_base::xalloc();

        FILE* get_standard_stream(const std::ostream& stream);
        bool is_colorized(std::ostream& stream);
        bool is_atty(const std::ostream& stream);

    #if defined(TERMCOLOR_OS_WINDOWS)
        void win_change_attributes(std::ostream& stream, int foreground, int background=-1);
    #endif
    }

    // Specials
    std::ostream& colorize(std::ostream& stream);
    std::ostream& nocolorize(std::ostream& stream);
    std::ostream& reset(std::ostream& stream);
    std::ostream& bold(std::ostream& stream);
    std::ostream& dark(std::ostream& stream);
    std::ostream& underline(std::ostream& stream);
    std::ostream& blink(std::ostream& stream);
    std::ostream& reverse(std::ostream& stream);
    std::ostream& concealed(std::ostream& stream);

    // Foreground
    std::ostream& black(std::ostream& stream);
    std::ostream& red(std::ostream& stream);
    std::ostream& green(std::ostream& stream);
    std::ostream& yellow(std::ostream& stream);
    std::ostream& blue(std::ostream& stream);
    std::ostream& magenta(std::ostream& stream);
    std::ostream& cyan(std::ostream& stream);
    std::ostream& grey(std::ostream& stream);
    std::ostream& white(std::ostream& stream);

    // Background
    std::ostream& on_black(std::ostream& stream);
    std::ostream& on_red(std::ostream& stream);
    std::ostream& on_green(std::ostream& stream);
    std::ostream& on_yellow(std::ostream& stream);
    std::ostream& on_blue(std::ostream& stream);
    std::ostream& on_magenta(std::ostream& stream);
    std::ostream& on_cyan(std::ostream& stream);
    std::ostream& on_grey(std::ostream& stream);
    std::ostream& on_clear(std::ostream& stream);

    namespace _internal
    {
        FILE* get_standard_stream(const std::ostream& stream);
        bool is_colorized(std::ostream& stream);
        bool is_atty(const std::ostream& stream);

    #if defined(TERMCOLOR_OS_WINDOWS)
        void win_change_attributes(std::ostream& stream, int foreground, int background);
    #endif 

    }
}

#undef TERMCOLOR_OS_WINDOWS
#undef TERMCOLOR_OS_MACOS
#undef TERMCOLOR_OS_LINUX

