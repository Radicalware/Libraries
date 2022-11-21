﻿#pragma once
#pragma warning (disable : 26444) // allow anynomous objects

/*
* Copyright[2019][Joel Leagues aka Scourge]
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

#include<string>
#include<vector>
#include<utility>
#include<regex>
#include<functional> 
#include<algorithm>
#include<locale>
#include<memory>
#include<initializer_list>
#include<iostream>
#include<string.h>
#include<ctype.h>
#include<type_traits>
#include<iomanip>
#include<sstream>

#ifndef UsingNVCC
#include "re2/re2.h"
#endif

#include "Color.h"
#include "xvector.h"

class xstring : public std::string
{
    using ull = unsigned long long;
public:
    using std::string::basic_string;
    static const xstring static_class;

    xstring(std::initializer_list<char> lst) : std::string(std::move(lst)) {};

    xstring(const xp<std::string>& str)      : std::string(str.Get()) {};
    xstring(const std::string& str)          : std::string(str) {};
    xstring(std::string&& str) noexcept      : std::string(std::move(str)) {};
    xstring(size_t repeat, const char chr)   : std::string(repeat, chr) {};

    xstring(std::stringstream& Stream) : std::string(std::move(Stream.str())) {}
    xstring(std::ostringstream& Stream) : std::string(std::move(Stream.str())) {}
    xstring(std::stringstream&& Stream) : std::string(std::move(Stream.str())) {}
    xstring(std::ostringstream&& Stream) : std::string(std::move(Stream.str())) {}

    xstring(const char chr);
    xstring(const char* chrs);
    xstring(const unsigned char* chrs);
    xstring(const unsigned char* chrs, size_t len);
    xstring(const wchar_t* chrs);
    xstring(const std::wstring& wstr);

    bool operator!(void) const;

    void operator+=(const char chr);
    void operator+=(const char* chr);
    void operator+=(const unsigned char* chr);
    void operator+=(const std::string& str);
    void operator+=(std::string&& str);

    xstring operator+(const char chr);
    xstring operator+(const char* chr);
    xstring operator+(const unsigned char* chr);
    xstring operator+(const std::string& str);
    xstring operator+(std::string&& str);

    char  At(size_t Idx) const;
    char& At(size_t Idx);

    char  First(size_t Idx = 0) const;
    char& First(size_t Idx = 0);

    char  Last(size_t Idx = 0) const;
    char& Last(size_t Idx = 0);

    INL size_t Size() const { return The.size(); }
    INL const char* Ptr() const { return The.c_str(); }

    void Print() const;
    void Print(int num) const;
    void Print(const xstring& front, const xstring& end = "") const;
    void Print(const char chr1, const char chr2 = ' ') const;
    void Print(const char* chr1, const char* chr2 = "\0") const;

    std::string   ToStdString() const;
    std::wstring  ToStdWString() const;
    RA::SharedPtr<unsigned char*> ToUnsignedChar() const;
    xstring       ToByteCode() const;
    xstring       FromByteCodeToASCII() const;

    xstring ToUpper()  const;
    xstring ToLower()  const;
    xstring ToProper() const;

    xstring operator*(int total) const;
    void operator*=(int total);

    xstring Remove(const char val) const;
    void    InRemove(const char val); // Inline Remove

    xstring Reverse() const;
    // =================================================================================================================================

    xvector<xstring> SingleSplit(size_t loc) const;
    xvector<xstring> Split(size_t loc) const;
    xvector<xstring> Split(const std::regex& rex) const;
    xvector<xstring> Split(const xstring& pattern, RXM::Type mod = RXM::ECMAScript) const;
    xvector<xstring> Split(const char splitter, RXM::Type mod = RXM::ECMAScript) const;

    xvector<xstring> InclusiveSplit(const std::regex& rex, bool single = true) const;
    xvector<xstring> InclusiveSplit(const xstring& splitter, RXM::Type mod = RXM::ECMAScript, bool single = true) const;
    xvector<xstring> InclusiveSplit(const char* splitter, RXM::Type mod = RXM::ECMAScript, bool aret = true) const;
    xvector<xstring> InclusiveSplit(const char splitter, RXM::Type mod = RXM::ECMAScript, bool aret = true) const;

    //// =================================================================================================================================
#ifndef UsingNVCC
    bool IsByteCode() const;
    bool Match(const re2::RE2& rex) const;
    bool MatchLine(const re2::RE2& rex) const;
    bool MatchAllLines(const re2::RE2& rex) const;
    bool Scan(const re2::RE2& rex) const;
    bool ScanLine(const re2::RE2& rex) const;
    bool ScanAllLines(const re2::RE2& rex) const;
    bool ScanList(const xvector<re2::RE2>& rex_lst) const;
    bool ScanList(const xvector<re2::RE2*>& rex_lst) const;
    xvector<xstring> Findall(const re2::RE2& rex) const;
    xvector<xstring> Findwalk(const re2::RE2& rex) const;
    xstring Sub(const RE2& rex, const std::string& replacement) const;
    xstring Sub(const RE2& rex, const re2::StringPiece& replacement) const;
    xstring Sub(const RE2& rex, const char* replacement) const;
#endif

    //   match is based on regex_match
    bool Match(const std::regex& rex) const;
    bool Match(const xstring& pattern, RXM::Type mod = RXM::ECMAScript) const;
    bool Match(const char* str, RXM::Type mod = RXM::ECMAScript) const;

    bool MatchLine(const std::regex& rex) const;
    bool MatchLine(const xstring& pattern, RXM::Type mod = RXM::ECMAScript) const;
    bool MatchLine(const char* str, RXM::Type mod = RXM::ECMAScript) const;

    bool MatchAllLines(const std::regex& rex) const;
    bool MatchAllLines(const xstring& pattern, RXM::Type mod = RXM::ECMAScript) const;
    bool MatchAllLines(const char* str, RXM::Type mod = RXM::ECMAScript) const;

    //   scan is based on regex_search
    bool Scan(const std::regex& rex) const;
    bool Scan(const char pattern, RXM::Type mod = RXM::ECMAScript) const;
    bool Scan(const xstring& pattern, RXM::Type mod = RXM::ECMAScript) const;
    bool Scan(const char* str, RXM::Type mod = RXM::ECMAScript) const;

    bool ScanLine(const std::regex& rex) const;
    bool ScanLine(const xstring& pattern, RXM::Type mod = RXM::ECMAScript) const;
    bool ScanLine(const char* str, RXM::Type mod = RXM::ECMAScript) const;

    bool ScanAllLines(const std::regex& rex) const;
    bool ScanAllLines(const xstring& pattern, RXM::Type mod = RXM::ECMAScript) const;
    bool ScanAllLines(const char* str, RXM::Type mod = RXM::ECMAScript) const;

    bool ScanList(const xvector<std::regex>& rex_lst) const;
    bool ScanList(const xvector<xstring>& lst, RXM::Type mod = RXM::ECMAScript) const;

    bool ScanList(const xvector<std::regex*>& rex_lst) const;
    bool ScanList(const xvector<xstring*>& lst, RXM::Type mod = RXM::ECMAScript) const;

    // exact match (no regex)
    bool Is(const xstring& other) const;
    size_t hash() const;

    // =================================================================================================================================

    int HasNonAscii(int front_skip = 0, int end_skip = 0, int threshold = 0) const;
    bool HasNulls() const;
    bool HasDualNulls() const;
    void RemoveNulls();
    xstring RemoveNonAscii() const;

    // =================================================================================================================================

    xvector<xstring> Findall(const std::regex& rex) const;
    xvector<xstring> Findall(const xstring& pattern, RXM::Type mod = RXM::ECMAScript) const;
    xvector<xstring> Findall(const char* pattern, RXM::Type mod = RXM::ECMAScript) const;

    xvector<xstring> Findwalk(const std::regex& rex) const;
    xvector<xstring> Findwalk(const xstring& pattern, RXM::Type mod = RXM::ECMAScript) const;
    xvector<xstring> Findwalk(const char* pattern, RXM::Type mod = RXM::ECMAScript) const;

    xvector<xstring> Search(const std::regex& rex, int count = -1) const;
    xvector<xstring> Search(const xstring& pattern, RXM::Type mod = RXM::ECMAScript, int count = -1) const;
    xvector<xstring> Search(const char* pattern, RXM::Type mod = RXM::ECMAScript, int count = -1) const;

    // =================================================================================================================================

    bool Has(const char var_char) const;
    bool Lacks(const char var_char) const;
    size_t Count(const char var_char) const;
    size_t Count(const xstring& pattern) const;

    // =================================================================================================================================

    xstring Sub(const std::regex& rex, const std::string& replacement) const;
    xstring Sub(const std::string& pattern, const std::string& replacement, RXM::Type mod = RXM::ECMAScript) const;
    xstring Sub(const char* pattern, const std::string& replacement, RXM::Type mod = RXM::ECMAScript) const;
    xstring Sub(const char* pattern, const char* replacement, RXM::Type mod = RXM::ECMAScript) const;

    xstring& Trim();
    xstring& LeftTrim();
    xstring& RightTrim();
    xstring& Trim(const xstring& trim);
    xstring& LeftTrim(const xstring& trim);
    xstring& RightTrim(const xstring& trim);

    // =================================================================================================================================

    xstring operator()(const long long int x) const;
    xstring operator()(
        const long long int x,
        const long long int y,
        const long long int z = 0,
        const char removal_method = 's') const;

    // =================================================================================================================================

    bool        BxNumber() const;
    int         ToInt() const;
    long        ToLong() const;
    long long   ToLongLong() const;
    size_t      To64() const;
    double      ToDouble() const;
    float       ToFloat() const;

    // =================================================================================================================================

    xstring ToBlack() const;
    xstring ToRed() const;
    xstring ToGreen() const;
    xstring ToYellow() const;
    xstring ToBlue() const;
    xstring ToMegenta() const;
    xstring ToCyan() const;
    xstring ToGrey() const;
    xstring ToWhite() const;

    xstring ToOnBlack() const;
    xstring ToOnRed() const;
    xstring ToOnGreen() const;
    xstring ToOnYellow() const;
    xstring ToOnBlue() const;
    xstring ToOnMegenta() const;
    xstring ToOnCyan() const;
    xstring ToOnGrey() const;
    xstring ToOnWhite() const;

    xstring ResetColor() const;
    xstring ToBold() const;
    xstring ToUnderline() const;
    xstring ToInvertedColor() const;

    
    INL static xstring Black()       { return static_class.ToBlack(); }
    INL static xstring Red()         { return static_class.ToRed(); }
    INL static xstring Green()       { return static_class.ToGreen(); }
    INL static xstring Yellow()      { return static_class.ToYellow(); }
    INL static xstring Blue()        { return static_class.ToBlue(); }
    INL static xstring Megenta()     { return static_class.ToMegenta(); }
    INL static xstring Cyan()        { return static_class.ToCyan(); }
    INL static xstring Grey()        { return static_class.ToGrey(); }
    INL static xstring White()       { return static_class.ToWhite(); }

    INL static xstring OnBlack()     { return static_class.ToOnBlack(); }
    INL static xstring OnRed()       { return static_class.ToOnRed(); }
    INL static xstring OnGreen()     { return static_class.ToOnGreen(); }
    INL static xstring OnYellow()    { return static_class.ToOnYellow(); }
    INL static xstring OnBlue()      { return static_class.ToOnBlue(); }
    INL static xstring OnMegenta()   { return static_class.ToOnMegenta(); }
    INL static xstring OnCyan()      { return static_class.ToOnCyan(); }
    INL static xstring OnGrey()      { return static_class.ToOnGrey(); }
    INL static xstring OnWhite()     { return static_class.ToOnWhite(); }
    // =================================================================================================================================
};

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

xstring operator+(const char First, const xstring&  Second);
xstring operator+(const char First,       xstring&& Second);

xstring operator+(const char* const First, const xstring& Second);
xstring operator+(const char* const First, xstring&& Second);

namespace RA
{
    // stand-alone function

    // Object
    template<typename T>
    inline typename std::enable_if<std::is_class<T>::value && !std::is_pointer<T>::value, xstring>::type
        ToXString(const T& obj)
    {
        std::ostringstream ostr;
        ostr << obj;
        return xstring(ostr.str().c_str());
    }

    // Number
    template<typename T>
    inline typename std::enable_if<!std::is_class<T>::value && !std::is_pointer<T>::value, xstring>::type
        ToXString(const T& obj)
    {
        std::ostringstream ostr;
        ostr << obj;
        return xstring(ostr.str().c_str());
    }

    // Number
    template<typename T>
    inline typename std::enable_if<!std::is_class<T>::value && !std::is_pointer<T>::value, xstring>::type
        ToXString(const T& obj, const size_t FnPercision, const bool FbFixed = true)
    {
        std::ostringstream ostr;
        ostr.precision(FnPercision);
        if (FbFixed)
            ostr << std::fixed << obj;
        else
            ostr << obj;
        return xstring(ostr.str().c_str());
    }


    // Object Pointer
    template<typename T>
    inline typename std::enable_if<std::is_class<T>::value && std::is_pointer<T>::value, xstring>::type
        ToXString(const T& obj)
    {
        std::ostringstream ostr;
        ostr << *obj;
        return xstring(ostr.str().c_str());
    }

    // Fundamental Pointer (probably a pointer array)
    template<typename T>
    inline typename std::enable_if<!std::is_class<T>::value&& std::is_pointer<T>::value, xstring>::type
        ToXString(const T& obj)
    {
        throw "Do Not Use!!";
    }


    template<class N>
    UsingFundamental(xstring) TruncateNum(N FnValue, const size_t FnPercision = 0, const bool FbFromStart = false)
    {
        std::ostringstream SS;
        SS.precision(17);
        SS << std::fixed << FnValue;
        xstring RetStr = SS;

        if (!RetStr.Count('.'))
            return SS;

        if (FbFromStart)
        {
            const auto Loc = RetStr.find('.');
            if (Loc + 1 >= FnPercision)
                return RetStr.substr(0, Loc);
            return RetStr.substr(0, FnPercision);
        }

        xvector<xstring> Vec = RetStr.Split('.');
        return Vec[0] + '.' + Vec[1].substr(0, FnPercision);
    }

    template<typename N>
    UsingFundamental(size_t) GetPrecision(N FnValue)
    {
        size_t Counter = 0;
        while (FnValue < 1)
        {
            FnValue *= 10;
            Counter++;
        }
        return Counter;
    }

    template<class N>
    UsingFundamental(xstring) FormatNum(N FnValue, const size_t FnPercision = 0)
    {
        std::ostringstream SS;
        SS.precision(17);
        SS.imbue(std::locale(""));
        SS << std::fixed << FnValue;
        xstring Out = SS;

        if (FnPercision)
        {
            if (Out.Count('.')) // only if it is a decimal number
            {
                xvector<xstring> Vec = Out.Split('.');
                return Vec[0] + '.' + Vec[1].substr(0, FnPercision);
            }
            return SS;
        }

        if (!Out.Count('.'))
            return Out;

        static const std::regex OnlyZeros(R"(^((\-?)0\.0*)$)", RXM::ECMAScript);
        static const std::regex JustTrailingZeros(R"(^((\-?)[\d,]+)(\.0+)$)", RXM::ECMAScript);

        if (Out.Match(OnlyZeros))
            return "0"; // return a single zero: 0.0000

        if (Out.Size() >= 3 && Out[0] == '0' && Out[1] == '.')
            return Out.Search(R"(^((\-?)0\.0*[1-9]*)(.*)$)").At(0); // when (num < 1), trim trailing zeros:  0.000500

        if (Out.Match(JustTrailingZeros))
            return Out.Search(R"(^((\-?)[\d,]+)(\.0+)$)").At(0); // remove trailing zeros: 555.00000

        static const std::regex TrailingZeroPattern(R"((0{5,}\d*)$)", RXM::ECMAScript); // find trailnig 0s and unreliable trailers
        static const std::regex TrailingNinePattern(R"((9{5,}\d*)$)", RXM::ECMAScript); // find trailnig 0s and unreliable trailers
        
        if (!Out.Scan(TrailingZeroPattern) && !Out.Scan(TrailingNinePattern))
            return Out;

        Out = Out.Sub(TrailingZeroPattern, "").Sub(TrailingNinePattern, "");
        if (Out.Last() == '.')
        {
            FnValue += 1;
            SS.str(xstring::static_class);
            SS << static_cast<size_t>(FnValue);
            return SS;
        }
        return Out;
    }

    template<class T>
    xstring FormatInt(T Value)
    {
        std::ostringstream SS;
        SS.imbue(std::locale(""));
        SS << round(Value);
        return SS;
    }


    template<class T>
    xstring FormatInt(T Value, const size_t FnPercision)
    {
        static const std::regex StripDouble(R"(\..*$)", RXM::ECMAScript);
        std::ostringstream SS;
        if (FnPercision)
            SS.precision(FnPercision);
        else
            SS.precision(17);
        SS.imbue(std::locale(""));
        SS << std::fixed << Value;
        xstring Out = SS;
        return Out.Sub(StripDouble, "");
    }

    // wide string to xstring
    xstring WTXS(const wchar_t* wstr);
}

#include "std_xstring.h"
