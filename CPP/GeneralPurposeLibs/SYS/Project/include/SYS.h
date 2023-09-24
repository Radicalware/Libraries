#pragma once

// SYS.h version 1.5.0

/*
* Copyright[2018][Joel Leagues aka Scourge]
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

// -------------------------------------------------------------------------------
// ALERT iterator will majorly slow down your performance if you don't
// optimize your compiler settings "-O2", else it will increase speed when
// not on windows (windows will give you even speeds with optimization else lag)
// also, on windows, be sure to remove the debugging for the iterator. 
// -------------------------------------------------------------------------------

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    #ifdef DLL_EXPORT
       #define EXI __declspec(dllexport)
    #else
       #define EXI __declspec(dllimport)
    #endif
#else
    #define EXI
#endif

#include<iostream>
#include<stdexcept>
#include<unordered_map>
#include<stdio.h>
#include<algorithm>
#include<regex>
#include<cstddef>

// ----- Radicalware Libs -------------------
// ----- eXtended STL Functionality ---------
#include "xstring.h"
#include "xvector.h"
#include "xmap.h"
// ----- eXtended STL Functionality ---------
// ----- Radicalware Libs -------------------

namespace RA
{
    class EXI SYS
    {
    private:
        bool MbArgsSet = false;

        xint MnSize = 0;
        xvector<xstring> MvCliArgs;
        xmap<xstring, char> MmAliasSC; // Alias
        xmap<char, xstring> MmAliasCS; // Alias
        xmap<char, xvector<xstring>> MmArgs;
        xvector<const xstring*> MvKeysStr;
        xvector<char>     MvKeysChr;
        xmap<xstring, xstring>  MmENV;

        xstring MsFile;
        xstring MsPath;

        // (From Str) = MmChrArgs.Key(MmAliasStrChar.Get("--Account"))
        // (From Chr) = MmCharArgs.Key('a')
        // ======================================================================================================================
    public:

        SYS() {}
        SYS(int argc, char** argv, char** env = nullptr);
        ~SYS() {}
        // -------------------------------------------------------------------------------------------------------------------
        // >>>> args
        void SetArgs(int argc, char** argv);
        void AddAlias(const char FChr, const xstring& FStr); // char_arg, string_arg
        // -------------------------------------------------------------------------------------------------------------------
        RIN int ArgC() const { return MnSize; } // Includes File Name
        RIN int ArgCount() const { return MnSize - 1; }
        xvector<xstring> ArgV() const;
        xstring ArgV(const size_t Idx) const;
        bool    Arg(const size_t Idx, const char FChar) const;

        xvector<char>           GetChrKeys() const;
        xvector<const xstring*> GetStrKeys() const;
        // -------------------------------------------------------------------------------------------------------------------
        xstring FullPath();
        xstring Path();
        xstring File();
        // -------------------------------------------------------------------------------------------------------------------
        xvector<xstring> Key(const xstring& FKey) const;
        xvector<xstring> Key(const char FKey) const;
        bool HasArgs() const;
        // -------------------------------------------------------------------------------------------------------------------
        // Used for str/bool
        bool Has(const xstring& FKey) const;
        bool Has(const char FKey) const;
        // -------------------------------------------------------------------------------------------------------------------
        // Used for str/str key-value args (not str/str[]) key-value args
        bool HasVal(const xstring& FKey) const;
        bool HasVal(const char FKey) const;
        xstring GetVal(const xstring& FKey) const;
        xstring GetVal(const char FKey) const;
        //// -----------------------------------------------------------------------------------------------------------------
        // Almost everything above can be handled using the operator overloading below and is the prefered method
        xvector<xstring> operator[](const xstring& FKey);                     // Return Key-Values
        xvector<xstring> operator[](const char FKey);                         // Return Key-Values
        xstring operator[](const int FKey);                                   // Return value by arg location in argv
        bool operator()(const xstring& FKey) const;                           // Test boolean for key or Key
        bool operator()(const xstring& FKey, const xstring& FValue) const;    // Test boolean for key or KVP
        bool operator()(const char FKey) const;                               // Test boolean for Key
        bool operator()(const char FKey, const xstring& FValue) const;        // Test boolean for KVP
        // ------------------------------------------------------------------------------------------------------------------
        bool Help();
        // ======================================================================================================================
    private:
        bool HasSingleDash(const xstring& FsArg) const;
        bool HasDoubleDash(const xstring& FsArg) const;
        bool IsArgType(const xstring& FsArg) const;
    };
};
