#pragma once
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
#include<sstream>

#include "re2/re2.h"

#include "Memory.h"
#include "SharedPtr.h"
#include "Color.h"
#include "re2/re2.h"
#include "xvector.h"

class xstring : public std::string
{
    using ull = unsigned long long;
public:
    using type = xstring;
    using std::string::basic_string;

    static const xstring StaticClass;

    xp<std::stringstream> MoStreamPtr = nullptr; // constexpr

    xstring(std::initializer_list<char> lst) : std::string(std::move(lst)) {};

    xstring(const xp<std::string>& FsTarget) : std::string(FsTarget.Get()) {};
    xstring(const std::string& FsTarget) : std::string(FsTarget) {};
    xstring(std::string&& FsTarget) noexcept : std::string(std::move(FsTarget)) {};
    xstring(xint repeat, const char chr) : std::string(repeat, chr) {};

    xstring(std::stringstream& Stream) : std::string(std::move(Stream.str())) {}
    xstring(std::ostringstream& Stream) : std::string(std::move(Stream.str())) {}
    xstring(std::stringstream&& Stream) : std::string(std::move(Stream.str())) {}
    xstring(std::ostringstream&& Stream) : std::string(std::move(Stream.str())) {}

    xstring(const char chr);
    xstring(const char* chrs);
    xstring(const unsigned char* chrs);
    xstring(const unsigned char* chrs, xint len);
    xstring(const wchar_t* chrs);
    xstring(const std::wstring& wstr);

    void operator=(const char chr);
    void operator=(const char* chrs);
    void operator=(const unsigned char* chrs);
    void operator=(const wchar_t* chrs);
    void operator=(const std::wstring& wstr);

    bool operator!(void) const;

    void operator+=(const char chr);
    void operator+=(const char* chr);
    void operator+=(const unsigned char* chr);
    void operator+=(const std::string& FsTarget);
    void operator+=(std::string&& FsTarget);

    xstring operator+(const char chr);
    xstring operator+(const char* chr);
    xstring operator+(const unsigned char* chr);
    xstring operator+(const std::string& FsTarget);
    xstring operator+(std::string&& FsTarget);

    TTT RIN RetIfTrue(xstring&,  IsSame(T, xstring)) operator<<(const   xp<T>& Other) { The += *Other; return The; }
    TTT RIN RetIfTrue(xstring&,  IsSame(T, xstring)) operator<<(const      T&  Other) { The +=  Other; return The; }

    TTT RIN RetIfTrue(xstring,   IsSame(T, xstring)) operator<<(const   xp<T>& Other) const { auto Str = The; Str += *Other; return Str; }
    TTT RIN RetIfTrue(xstring,   IsSame(T, xstring)) operator<<(const      T&  Other) const { auto Str = The; Str +=  Other; return Str; }

    TTT RIN RetIfTrue(xstring&, !IsSame(T, xstring)) operator<<(const xp<T>& Other);
    TTT RIN RetIfTrue(xstring&, !IsSame(T, xstring)) operator<<(const    T& Other);

    TTT RIN RetIfTrue(xstring,  !IsSame(T, xstring)) operator<<(const xp<T>& Other) const;
    TTT RIN RetIfTrue(xstring,  !IsSame(T, xstring)) operator<<(const    T& Other)  const;

    char  At(xint Idx) const;
    char& At(xint Idx);

    char  First(xint Idx = 0) const;
    char& First(xint Idx = 0);

    char  Last(xint Idx = 0) const;
    char& Last(xint Idx = 0);

    RIN xint Size() const { return The.size(); }
    RIN const char* Ptr() const { return The.c_str(); }

    void Print() const;
    void Print(int num) const;
    void Print(const xstring& front, const xstring& end = "") const;
    void Print(const char chr1, const char chr2 = ' ') const;
    void Print(const char* chr1, const char* chr2 = "\0") const;

    std::string   ToStdString() const;
    std::wstring  ToStdWString() const;
#if _HAS_CXX20
    RA::SharedPtr<unsigned char[]> ToUnsignedChar() const;
#else
    RA::SharedPtr<unsigned char*> ToUnsignedChar() const;
#endif
    xstring       ToByteCode() const;
    xstring       FromByteCodeToASCII() const;

    xstring ToUpper()  const;
    xstring ToLower()  const;
    xstring ToProper() const;

    xstring operator*(int total) const;
    void operator*=(int total);

    xstring Reverse() const;
    // =================================================================================================================================

    xvector<xstring> SingleSplit(xint loc) const;
    xvector<xstring> Split(xint loc) const;
    xvector<xstring> Split(const std::regex& FsRex) const;
    xvector<xstring> Split(const xstring& FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;
    xvector<xstring> Split(const char splitter, RXM::Type FeMod = RXM::ECMAScript) const;

    xvector<xstring> InclusiveSplit(const std::regex& FsRex, bool single = true) const;
    xvector<xstring> InclusiveSplit(const xstring& splitter, RXM::Type FeMod = RXM::ECMAScript, bool single = true) const;
    xvector<xstring> InclusiveSplit(const char* splitter, RXM::Type FeMod = RXM::ECMAScript, bool aret = true) const;
    xvector<xstring> InclusiveSplit(const char splitter, RXM::Type FeMod = RXM::ECMAScript, bool aret = true) const;

    //// =================================================================================================================================
// #ifndef UsingNVCC
    bool IsByteCode() const;

    bool Match(const re2::RE2& FsRex) const;
    bool MatchLine(const re2::RE2& FsRex) const;
    bool MatchAllLines(const re2::RE2& FsRex) const;

    bool Scan(const re2::RE2& FsRex) const;
    bool ScanLine(const re2::RE2& FsRex) const;
    bool ScanAllLines(const re2::RE2& FsRex) const;

    bool ScanList(const xvector<re2::RE2>& FvRex) const;
    bool ScanList(const xvector<xp<re2::RE2>>& FvRex) const;

    xvector<xstring> Findall(const re2::RE2& FsRex) const;
    xvector<xstring> Findwalk(const re2::RE2& FsRex) const;

    xstring  Sub(const RE2& FsRex, const std::string& FsReplacement) const;
    xstring  Sub(const RE2& FsRex, const re2::StringPiece& FsReplacement) const;
    xstring  Sub(const RE2& FsRex, const char* FsReplacement) const;
    xstring  Sub(const char FsRex, const char FsReplacement)  const { auto Str = The; return Str.InSub(FsRex, FsReplacement); }
    xstring  Sub(const char FsRex, const char* FsReplacement) const { auto Str = The; return Str.InSub(FsRex, FsReplacement); }

    xstring& InSub(const RE2& FsRex, const std::string& FsReplacement);
    xstring& InSub(const RE2& FsRex, const re2::StringPiece& FsReplacement);
    xstring& InSub(const RE2& FsRex, const char* FsReplacement);

    xstring& InRemove(const char FsRex);
    xstring    Remove(const char val) const;

    xstring& InSub(const char FsRex, const char FsReplacement);
    xstring& InSub(const char FsRex, const char* FsReplacement);
    xstring& InSub(const char FsRex, const xstring& FsReplacement);

    xstring IfFindReplace(const RE2& LoFind, const RE2& LoReplace, const std::string& FsReplacement) const;

    // #endif

        //   match is based on regex_match
    bool Match(const std::regex& FsRex) const;
    bool Match(const xstring& FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;
    bool Match(const char* FsTarget, RXM::Type FeMod = RXM::ECMAScript) const;

    bool MatchLine(const std::regex& FsRex) const;
    bool MatchLine(const xstring& FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;
    bool MatchLine(const char* FsTarget, RXM::Type FeMod = RXM::ECMAScript) const;

    bool MatchAllLines(const std::regex& FsRex) const;
    bool MatchAllLines(const xstring& FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;
    bool MatchAllLines(const char* FsTarget, RXM::Type FeMod = RXM::ECMAScript) const;

    //   scan is based on regex_search
    bool Scan(const std::regex& FsRex) const;
    bool Scan(const char FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;
    bool Scan(const xstring& FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;
    bool Scan(const char* FsTarget, RXM::Type FeMod = RXM::ECMAScript) const;

    bool ScanLine(const std::regex& FsRex) const;
    bool ScanLine(const xstring& FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;
    bool ScanLine(const char* FsTarget, RXM::Type FeMod = RXM::ECMAScript) const;

    bool ScanAllLines(const std::regex& FsRex) const;
    bool ScanAllLines(const xstring& FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;
    bool ScanAllLines(const char* FsTarget, RXM::Type FeMod = RXM::ECMAScript) const;

    bool ScanList(const xvector<std::regex>& FvRex) const;
    bool ScanList(const xvector<xstring>& lst, RXM::Type FeMod = RXM::ECMAScript) const;

    bool ScanList(const xvector<std::regex*>& FvRex) const;
    bool ScanList(const xvector<xstring*>& lst, RXM::Type FeMod = RXM::ECMAScript) const;

    // exact match (no regex)
    bool Is(const xstring& other) const;
    xint hash() const;

    // =================================================================================================================================

    int HasNonAscii(int front_skip = 0, int end_skip = 0, int threshold = 0) const;
    bool HasNulls() const;
    bool HasDualNulls() const;
    void RemoveNulls();
    xstring RemoveNonAscii() const;

    // =================================================================================================================================

    xvector<xstring> Findall(const std::regex& FsRex) const;
    xvector<xstring> Findall(const xstring& FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;
    xvector<xstring> Findall(const char* FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;

    xvector<xstring> Findwalk(const std::regex& FsRex) const;
    xvector<xstring> Findwalk(const xstring& FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;
    xvector<xstring> Findwalk(const char* FsPattern, RXM::Type FeMod = RXM::ECMAScript) const;

    xvector<xstring> Search(const std::regex& FsRex, int count = -1) const;
    xvector<xstring> Search(const xstring& FsPattern, RXM::Type FeMod = RXM::ECMAScript, int count = -1) const;
    xvector<xstring> Search(const char* FsPattern, RXM::Type FeMod = RXM::ECMAScript, int count = -1) const;

    // =================================================================================================================================

    bool Has(const char var_char) const;
    bool Lacks(const char var_char) const;
    xint Count(const char var_char) const;
    xint Count(const xstring& FsPattern) const;

    // =================================================================================================================================

    xstring Sub(const std::regex& FsRex, const std::string& FsReplacement) const;
    xstring Sub(const std::string& FsPattern, const std::string& FsReplacement, RXM::Type FeMod = RXM::ECMAScript) const;
    xstring Sub(const char* FsPattern, const std::string& FsReplacement, RXM::Type FeMod = RXM::ECMAScript) const;
    xstring Sub(const char* FsPattern, const char* FsReplacement, RXM::Type FeMod = RXM::ECMAScript) const;

    xstring IfFindReplace(const std::regex& LoFind, const std::regex& LoReplace, const std::string& FsReplacement) const;
    xstring IfFindReplace(const std::string& LsFind, const std::string& LsReplace, const std::string& FsReplacement, RXM::Type FeMod = RXM::ECMAScript) const;
    xstring IfFindReplace(const char* LsFind, const char* LsReplace, const std::string& FsReplacement, RXM::Type FeMod = RXM::ECMAScript) const;
    xstring IfFindReplace(const char* LsFind, const char* LsReplace, const char* FsReplacement, RXM::Type FeMod = RXM::ECMAScript) const;

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

    bool      BxNumber()   const;
    int       ToInt()      const;
    long      ToLong()     const;
    long long ToLongLong() const;
    xint      To64()       const;
    double    ToDouble()   const;
    float     ToFloat()    const;

    // =================================================================================================================================

    xstring ToBlack()      const;
    xstring ToRed()        const;
    xstring ToGreen()      const;
    xstring ToYellow()     const;
    xstring ToBlue()       const;
    xstring ToMegenta()    const;
    xstring ToCyan()       const;
    xstring ToGrey()       const;
    xstring ToWhite()      const;

    xstring ToOnBlack()    const;
    xstring ToOnRed()      const;
    xstring ToOnGreen()    const;
    xstring ToOnYellow()   const;
    xstring ToOnBlue()     const;
    xstring ToOnMegenta()  const;
    xstring ToOnCyan()     const;
    xstring ToOnGrey()     const;
    xstring ToOnWhite()    const;

    xstring ResetColor()   const;
    xstring ToBold()       const;
    xstring ToUnderline()  const;

    xstring ToInvertedColor() const;


    RIN static xstring Black() { return StaticClass.ToBlack(); }
    RIN static xstring Red() { return StaticClass.ToRed(); }
    RIN static xstring Green() { return StaticClass.ToGreen(); }
    RIN static xstring Yellow() { return StaticClass.ToYellow(); }
    RIN static xstring Blue() { return StaticClass.ToBlue(); }
    RIN static xstring Megenta() { return StaticClass.ToMegenta(); }
    RIN static xstring Cyan() { return StaticClass.ToCyan(); }
    RIN static xstring Grey() { return StaticClass.ToGrey(); }
    RIN static xstring White() { return StaticClass.ToWhite(); }

    RIN static xstring OnBlack() { return StaticClass.ToOnBlack(); }
    RIN static xstring OnRed() { return StaticClass.ToOnRed(); }
    RIN static xstring OnGreen() { return StaticClass.ToOnGreen(); }
    RIN static xstring OnYellow() { return StaticClass.ToOnYellow(); }
    RIN static xstring OnBlue() { return StaticClass.ToOnBlue(); }
    RIN static xstring OnMegenta() { return StaticClass.ToOnMegenta(); }
    RIN static xstring OnCyan() { return StaticClass.ToOnCyan(); }
    RIN static xstring OnGrey() { return StaticClass.ToOnGrey(); }
    RIN static xstring OnWhite() { return StaticClass.ToOnWhite(); }
    // =================================================================================================================================
};

xstring operator+(const char First, const xstring& Second);
xstring operator+(const char First, xstring&& Second);

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
        ToXString(const T& obj, const xint FnPercision, const bool FbFixed = true)
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
    inline typename std::enable_if<std::is_class<T>::value&& std::is_pointer<T>::value, xstring>::type
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
    UsingFundamental(xstring) TruncateNum(N FnValue, const xint FnPercision = 0, const bool FbFromStart = false)
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
    UsingFundamental(xint) GetPrecision(N FnValue)
    {
        xint Counter = 0;
        while (FnValue < 1)
        {
            FnValue *= 10;
            Counter++;
        }
        return Counter;
    }

    template<class N>
    UsingFundamental(xstring) FormatNum(N FnValue, const xint FnPercision = 0)
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
            else if (FnPercision)
            {
                Out += '.';
                for (xint i = 0; i < FnPercision; i++)
                    Out += '0';
            }
            return Out;
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
            SS.str(xstring::StaticClass);
            SS << static_cast<xint>(FnValue);
            return SS;
        }
        return Out;
    }

    template<class T>
    xstring FormatInt(T Value, const xint FnPercision = 0)
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

template<typename T>
RIN RetIfTrue(xstring&, !IsSame(T, xstring)) xstring::operator<<(const T& Other)
{
    try
    {
        if (!MoStreamPtr)
            MoStreamPtr = MKP<std::stringstream>();
        else
            (*MoStreamPtr).str(StaticClass);
        auto& MoStream = *MoStreamPtr;
        MoStream << Other;
        The += MoStream.str();
        return The;
    }
    catch (...)
    {
        throw "Error @ xstring& xstring::operator<<(const T& Other);";
    }
}

template<typename T>
RIN RetIfTrue(xstring&, !IsSame(T, xstring)) xstring::operator<<(const xp<T>& Other)
{
    try
    {
        if (!MoStreamPtr)
            MoStreamPtr = MKP<std::stringstream>();
        else
            (*MoStreamPtr).str(StaticClass);
        auto& MoStream = *MoStreamPtr;
        MoStream << *Other;
        The += MoStream.str();
        return The;
    }
    catch (...)
    {
        throw "Error @ xstring& xstring::operator<<(const xp<T>& Other);";
    }
}

template<typename T>
RIN RetIfTrue(xstring, !IsSame(T, xstring)) xstring::operator<<(const T& Other) const
{
    try
    {
        xstring LsReturn = The;
        LsReturn << Other;
        return LsReturn;
    }
    catch (...)
    {
        throw "Error @ xstring& xstring::operator<<(const T& Other);";
    }
}

template<typename T>
RIN RetIfTrue(xstring, !IsSame(T, xstring)) xstring::operator<<(const xp<T>& Other) const
{
    try
    {
        xstring LsReturn = The;
        LsReturn << *Other;
        return LsReturn;
    }
    catch (...)
    {
        throw "Error @ xstring& xstring::operator<<(const xp<T>& Other);";
    }
}
