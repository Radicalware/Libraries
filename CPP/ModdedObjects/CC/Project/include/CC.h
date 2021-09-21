#pragma once
// v1.0.1 --(MOD)-- Console Colors
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

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    #ifdef DLL_EXPORT
       #define EXI __declspec(dllexport)
    #else
       #define EXI __declspec(dllimport)
    #endif
#else
    #define EXI
#endif

#include <iostream>
#include <ostream>
#include <cstdio>

namespace CC
{
    namespace _internal
    {
        static int colorize_index = std::ios_base::xalloc();

        EXI FILE* GetStandardStream(const std::ostream& stream);
        EXI bool IsColorized(std::ostream& stream);
        EXI bool IsAtty(const std::ostream& stream);

#if defined(TERMCOLOR_OS_WINDOWS)
        EXI void win_change_attributes(std::ostream& stream, int foreground, int background = -1);
#endif
    }

    // Specials
    EXI std::ostream& Colorize(std::ostream& stream);
    EXI std::ostream& RemoveColor(std::ostream& stream);
    EXI std::ostream& Reset(std::ostream& stream);
    EXI std::ostream& Bold(std::ostream& stream);
    EXI std::ostream& Dark(std::ostream& stream);
    EXI std::ostream& Underline(std::ostream& stream);
    EXI std::ostream& Blink(std::ostream& stream);
    EXI std::ostream& Reverse(std::ostream& stream);
    EXI std::ostream& Concealed(std::ostream& stream);

    // Foreground
    EXI std::ostream& Black(std::ostream& stream);
    EXI std::ostream& Red(std::ostream& stream);
    EXI std::ostream& Green(std::ostream& stream);
    EXI std::ostream& Yellow(std::ostream& stream);
    EXI std::ostream& Blue(std::ostream& stream);
    EXI std::ostream& Magenta(std::ostream& stream);
    EXI std::ostream& Cyan(std::ostream& stream);
    EXI std::ostream& Grey(std::ostream& stream);
    EXI std::ostream& White(std::ostream& stream);

    // Background
    EXI std::ostream& OnBlack(std::ostream& stream);
    EXI std::ostream& OnRed(std::ostream& stream);
    EXI std::ostream& OnGreen(std::ostream& stream);
    EXI std::ostream& OnYellow(std::ostream& stream);
    EXI std::ostream& OnBlue(std::ostream& stream);
    EXI std::ostream& OnMagenta(std::ostream& stream);
    EXI std::ostream& OnCyan(std::ostream& stream);
    EXI std::ostream& OnGrey(std::ostream& stream);
    EXI std::ostream& RemoveBackgroundColor(std::ostream& stream);
}

#undef TERMCOLOR_OS_WINDOWS
#undef TERMCOLOR_OS_MACOS
#undef TERMCOLOR_OS_LINUX

