﻿#pragma warning (disable : 26444) // allow anynomous objects
#pragma warning (disable : 26812) // allow normal enums from STL

#include "ra_xstring.h"
#include <stdlib.h>
#include "xvector.h"

const xstring xstring::StaticClass = ""; // can't be inline (else incomplete class error)

xstring::xstring(const char chr)
{
    The.insert(The.begin(), chr);
}

xstring::xstring(const char* chrs)
{
    xint len = strlen(chrs) + 1;
    The.resize(len);
    The.insert(The.begin(), chrs, &chrs[len]);
    RemoveNulls();
}

xstring::xstring(const unsigned char* chrs)
{
    //for (const unsigned char* ptr = chrs; static_cast<char>(*ptr) != '\0'; ptr++)
    //    The.insert(The.end(), static_cast<char>(*ptr));
    xint len = strlen(reinterpret_cast<const char*>(chrs));
    The.resize(len);
    The.insert(The.begin(), chrs, &chrs[len]);
    // RemoveNulls(); // unsigned char is often uesd for byte-arrays
}

xstring::xstring(const unsigned char* chrs, xint len)
{
    // Does not Terminate Correctly
    //The.reserve(len);
    //The.insert(The.begin(), chrs, &chrs[len]);

    // Does Not Size correctly (resize may add more size than you want)
    //The.resize(len);
    //xint Start = 0;
    //const unsigned char* ptr = chrs;
    //for (xint idx = 0; idx < len; idx++)
    //    The[idx] = static_cast<char>(*(ptr++));
    //The[len] = '\0';

    The.reserve(len);
    xint Start = 0;
    const unsigned char* ptr = chrs;
    for (xint idx = 0; idx < len; idx++)
        The.insert(The.end(), static_cast<char>(*(ptr++)));
}

xstring::xstring(const wchar_t* chrs)
{
    The = RA::WTXS(chrs);
}

xstring::xstring(const std::wstring& wstr)
{
    The = RA::WTXS(wstr.c_str());
}

xstring::operator std::string() const {
    return static_cast<std::string>(The);
}


xstring::operator std::wstring() const {
    return SoConverter.from_bytes(The);
}

void xstring::operator=(const char chr)
{
    The.insert(The.begin(), chr);
}

void xstring::operator=(const char* chrs)
{
    xint len = strlen(chrs) + 1;
    The.resize(len);
    The.insert(The.begin(), chrs, &chrs[len]);
    RemoveNulls();
}

void xstring::operator=(const unsigned char* chrs)
{
    //for (const unsigned char* ptr = chrs; static_cast<char>(*ptr) != '\0'; ptr++)
    //    The.insert(The.end(), static_cast<char>(*ptr));
    xint len = strlen(reinterpret_cast<const char*>(chrs));
    The.resize(len);
    The.insert(The.begin(), chrs, &chrs[len]);
    // RemoveNulls(); // unsigned char is often uesd for byte-arrays
}

void xstring::operator=(const wchar_t* chrs)
{
    The = RA::WTXS(chrs);
}

void xstring::operator=(const std::wstring& wstr)
{
    The = RA::WTXS(wstr.c_str());
}

bool xstring::operator!(void) const {
    return bool(The.size() == 0);
}

void xstring::operator+=(const char chr) {
    The.insert(The.end(), chr);
}

void xstring::operator+=(const char* chr)
{
    The.reserve(Size() + strlen(chr));
    for (const auto* Ptr = &chr[0]; *Ptr != '\0'; Ptr++)
        The.insert(The.end(), *Ptr);
}

void xstring::operator+=(const unsigned char* chr)
{
    for (const auto* Ptr = &chr[0]; *Ptr != '\0'; Ptr++)
        The.insert(The.end(), *Ptr);
}

void xstring::operator+=(const std::string& str) {
    The.insert(The.end(), str.begin(), str.end());
}

void xstring::operator+=(std::string&& str) {
    The.insert(The.end(), std::make_move_iterator(str.begin()), std::make_move_iterator(str.end()));
}

void xstring::operator+=(const std::wstring& FsTarget)
{
    The += RA::WTXS(FsTarget);
}

void xstring::operator+=(std::wstring&& FsTarget)
{
    The += RA::WTXS(std::move(FsTarget));
}

xstring xstring::operator+(const char chr)
{
    xstring rstr = The;
    rstr.insert(rstr.end(), chr);
    return rstr;
}

xstring xstring::operator+(const char* chr)
{
    xstring retr;
    retr.reserve(The.size() + strlen(chr));
    retr += The;
    retr += chr;
    return retr;
}

xstring xstring::operator+(const unsigned char* chr)
{
    xstring retr;
    retr.reserve(The.size() + strlen(reinterpret_cast<const char*>(chr)));
    retr += The;
    retr += chr;
    return retr;
}

xstring xstring::operator+(const std::string& str)
{
    xstring rstr;
    rstr.reserve(The.size() + str.size());
    rstr += The;
    rstr += str;
    return rstr;
}

xstring xstring::operator+(std::string&& str)
{
    xstring rstr;
    rstr.reserve(The.size() + str.size());
    rstr += The;
    rstr += std::move(str);
    return rstr;
}

xstring xstring::operator+(const std::wstring& FsTarget)
{
    return The + RA::WTXS(FsTarget);
}

xstring xstring::operator+(std::wstring&& FsTarget)
{
    return The + RA::WTXS(std::move(FsTarget));
}

char xstring::At(xint Idx) const
{
    return at(Idx);
}

char& xstring::At(xint Idx)
{
    return at(Idx);
}

char xstring::First(xint Idx) const
{
    if (Idx >= The.Size())
        throw "Index Out Of Range";
    return The.operator[](Idx);
}

char& xstring::First(xint Idx)
{
    if (Idx >= The.Size())
        throw "Index Out Of Range";
    return The.operator[](Idx);
}

char xstring::Last(xint Idx) const
{
    if (Idx >= The.Size())
        throw "Index Out Of Range";
    return The.operator[](this->size() - Idx - 1);
}

char& xstring::Last(xint Idx)
{
    if (Idx >= The.Size())
        throw "Index Out Of Range";
    return The.operator[](this->size() - Idx - 1);
}

void xstring::Print() const
{
    printf("%s\n", The.c_str()); // printf is faster than std::cout
}

void xstring::Print(int num) const
{
    std::cout << The;
    char* new_lines = static_cast<char*>(calloc(static_cast<xint>(num) + 1, sizeof(char)));
    // calloc was used instead of "new" because "new" would give un-wanted after-effects.
    for (int i = 0; i < num; i++)
#pragma warning(suppress:6011) // we are derferencing a pointer, but assigning it a value at the same time
        new_lines[i] = '\n';
    std::cout << new_lines;
    free(new_lines);
}

void xstring::Print(const xstring& front, const xstring& end) const {
    std::cout << front << The << end << '\n';
}

void xstring::Print(const char chr1, const char chr2) const
{
    std::cout << chr1 << The << chr2 << '\n';
}

void xstring::Print(const char* chr1, const char* chr2) const
{
    if (strlen(chr2))
        std::cout << chr1 << The << chr2 << '\n';
    else
        std::cout << The << chr1 << '\n';
}

std::string xstring::ToStdString() const {
    return std::string(The.c_str()); // todo: return only from the base class
}

std::wstring xstring::ToStdWString() const
{
    return SoConverter.from_bytes(The);
}

#if _HAS_CXX20
RA::SharedPtr<unsigned char[]> xstring::ToUnsignedChar() const
{
    const auto LnSize = The.Size();
    auto UnsignedChar = RA::SharedPtr<unsigned char[]>(LnSize);
    for (int i = 0; i < LnSize; i++)
        UnsignedChar[i] = static_cast<unsigned char>(The[i]);
    return UnsignedChar;
}
#else
RA::SharedPtr<unsigned char*> xstring::ToUnsignedChar() const
{
    const auto LnSize = The.Size();
    auto UnsignedChar = RA::SharedPtr<unsigned char*>(LnSize);
    for (int i = 0; i < LnSize; i++)
        UnsignedChar[i] = static_cast<unsigned char>(The[i]);
    return UnsignedChar;
}
#endif

xstring xstring::ToByteCode() const
{
    auto DataPtr = The.ToUnsignedChar();
    auto UData = DataPtr.Ptr();

    static const char* NibbleToHexMapping = "0123456789ABCDEF";
    static const char* HexSplit = "\\x";

    int BufSize = The.Size();
    xstring Buffer;
    Buffer.resize(BufSize * 2 * 2 + 1);
    for (int i = 0; i < BufSize * 2; i += 2)
    {
        // divide by 16
        Buffer[2 * i] = HexSplit[0];
        Buffer[2 * i + 1] = HexSplit[1];
        int Nibble = UData[i / 2] >> 4;
        Buffer[2 * i + 2] = NibbleToHexMapping[Nibble];

        Nibble = UData[i / 2] & 0x0F;
        Buffer[2 * i + 3] = NibbleToHexMapping[Nibble];
    }
    Buffer[Buffer.Size() - 1] = '\0';
    return Buffer;
}

xstring xstring::FromByteCodeToASCII() const
{
    xstring Ascii = "";
    xstring part;
    part.resize(2);
    for (xint i = 0; i < The.length() - 2; i += 4)
    {
        part[0] = The[i + 2];
        part[1] = The[i + 3];
        Ascii += stoul(part, nullptr, 16);
    }
    return Ascii;
}

xstring xstring::ToUpper() const
{
    xstring ret_str;
    std::locale loc;
    for (xstring::const_iterator it = The.begin(); it != The.end(); it++)
        ret_str += std::toupper((*it), loc);
    return ret_str;
}

xstring xstring::ToLower() const
{
    xstring ret_str;
    std::locale loc;
    for (xstring::const_iterator it = The.begin(); it != The.end(); it++)
        ret_str += std::tolower((*it), loc);
    return ret_str;
}

xstring xstring::ToProper() const
{
    xstring ret_str;
    std::locale loc;
    bool FirstPass = true;
    for (xstring::const_iterator it = The.begin(); it != The.end(); it++)
    {
        if (FirstPass)
        {
            ret_str += std::toupper((*it), loc);
            FirstPass = false;
        }
        else
            ret_str += std::tolower((*it), loc);
    }
    return ret_str;
}

xstring xstring::operator*(int total) const
{
    xstring ret_str = The;
    for (int i = 0; i < total; i++)
        ret_str.insert(ret_str.end(), The.begin(), The.end());
    return ret_str;
}

void xstring::operator*=(int total)
{
    xstring str_copy = The;
    for (int i = 0; i < total; i++)
        The.insert(The.end(), str_copy.begin(), str_copy.end());
}


xstring xstring::Reverse() const
{
    xstring str;
    str.resize(The.size());

    xstring::const_reverse_iterator from_it = The.rbegin();
    xstring::iterator to_it = str.begin();

    for (; from_it != The.rend();)
    {
        *to_it = *from_it;
        from_it++;
        to_it++;
    }
    return str;
}

// ---------------------------------------------------------------

xvector<xstring> xstring::SingleSplit(xint loc) const
{
    xvector<xstring> ret_vec;
    ret_vec.reserve(4);
    ret_vec << The.substr(0, loc);
    ret_vec << The.substr(loc, The.size() - loc);

    return ret_vec;
}

xvector<xstring> xstring::Split(xint SegSize) const
{
    if (!SegSize)
        throw "Seg Size is Zero";
    xvector<xstring> Vec;
    Vec.reserve(The.Size() / SegSize + 1);
    xstring CurrentStr;
    CurrentStr.resize(SegSize + 1);
    xint Idx = 0;
    for (xstring::const_iterator It = The.cbegin(); It != The.cend(); It++)
    {
        if (SegSize > Idx)
        {
            CurrentStr[Idx] = *It;
        }
        else
        {
            Vec << CurrentStr;
            Idx = 0;
            CurrentStr[Idx] = *It;
        }
        Idx++;
    }
    if (The.Size() % SegSize > 0)
        Vec << CurrentStr.substr(0, Idx);
    return Vec;
}

xvector<xstring> xstring::Split(const std::regex& FsRex) const
{
    xvector<xstring> split_content;
    for (std::sregex_token_iterator iter(The.begin(), The.end(), FsRex, -1); iter != std::sregex_token_iterator(); ++iter)
        split_content << xstring(*iter);

    return split_content;
}

xvector<xstring> xstring::Split(const xstring& FsPattern, RXM::Type FeMod) const {
    return The.Split(std::regex(FsPattern.c_str(), FeMod));
}

xvector<xstring> xstring::Split(const char splitter, RXM::Type FeMod) const
{
    xvector<xstring> Vec;
    xstring Current;
    bool LbHit = true;
    auto LoSplitter = xstring(splitter);
    for (xstring::const_iterator It = The.cbegin(); It != The.cend(); It++)
    {
        if (*It != splitter)
        {
            LbHit = false;
            Current += *It;
        }
        else if (Current.size())
        {
            Vec << std::move(Current);
            Current.clear();
            LbHit = true;
        }
        else if (LbHit)
            Vec << LoSplitter;
    }
    if (Current.size())
        Vec << std::move(Current);
    return Vec;
}

xvector<xstring> xstring::InclusiveSplit(const std::regex& FsRex, bool single) const
{
    if (size() < 3)
        return xvector<xstring>({ The });

    xvector<xstring> retv;
    for (std::sregex_token_iterator iter(The.begin(), The.end(), FsRex, { -1, 0 }); iter != std::sregex_token_iterator(); ++iter)
        retv << xstring(*iter);

    if (!single) {
        if (retv.size() == 1)
            return xvector<xstring>();
    }
    return retv;
}


xvector<xstring> xstring::InclusiveSplit(const xstring& splitter, RXM::Type FeMod, bool single) const
{
    std::regex FsRex(splitter.c_str(), FeMod);
    // -1       grab NOT regex
    //  0       grab regex
    //  1       grab blank
    return The.InclusiveSplit(FsRex, single);
}

xvector<xstring> xstring::InclusiveSplit(const char* splitter, RXM::Type FeMod, bool aret) const {
    return The.InclusiveSplit(xstring(splitter), FeMod, aret);
}

xvector<xstring> xstring::InclusiveSplit(const char splitter, RXM::Type FeMod, bool aret) const 
{
    xvector<xstring> Vec;
    xstring Current;
    xstring LoSplitter = xstring(splitter);
    for (xstring::const_iterator It = The.cbegin(); It != The.cend(); It++)
    {
        if (*It != splitter)
            Current += *It;
        else if (Current.size())
        {
            Vec << std::move(Current);
            Current.clear();
        }
        else
            Vec << LoSplitter;
    }
    if (Current.size())
        Vec << std::move(Current);
    return Vec;
}

// =========================================================================================================================
// #ifndef UsingNVCC

bool xstring::IsByteCode() const
{
    static const RE2 ByteStringRex(R"(^(\\x[A-Fa-f0-9]{2})*$)");
    if (The.Match(ByteStringRex))
        return true;
    return false;
}

bool xstring::Match(const re2::RE2& FsRex) const {
    return RE2::FullMatch(c_str(), FsRex);
}

bool xstring::MatchLine(const re2::RE2& FsRex) const
{
    xvector<xstring> lines = The.Split('\n');
    for (xvector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (RE2::FullMatch(*iter, FsRex)) {
            return true;
        }
    }
    return false;
}

bool xstring::MatchAllLines(const re2::RE2& FsRex) const
{
    xvector<xstring> lines = The.Split('\n');
    for (xvector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!RE2::FullMatch(*iter, FsRex)) {
            return false;
        }
    }
    return true;
}

bool xstring::Scan(const re2::RE2& FsRex) const {
    return re2::RE2::PartialMatch(The.c_str(), FsRex);
}

bool xstring::ScanLine(const re2::RE2& FsRex) const
{
    xvector<xstring> lines = The.Split('\n');
    for (xvector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++)
    {
        if (re2::RE2::PartialMatch(*iter, FsRex))
            return true;
    }
    return false;
}

bool xstring::ScanAllLines(const re2::RE2& FsRex) const
{
    xvector<xstring> lines = The.Split('\n');
    for (xvector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!re2::RE2::PartialMatch(*iter, FsRex))
            return false;
    }
    return true;
}

bool xstring::ScanList(const xvector<re2::RE2>& FvRex) const
{
    for (xvector<re2::RE2>::const_iterator iter = FvRex.begin(); iter != FvRex.end(); iter++)
    {
        if (re2::RE2::PartialMatch(c_str(), *iter)) {
            return true;
        }
    }
    return false;
}

bool xstring::ScanList(const xvector<xp<re2::RE2>>& FvRex) const
{
    for (xvector<xp<re2::RE2>>::const_iterator iter = FvRex.begin(); iter != FvRex.end(); iter++)
    {
        if (re2::RE2::PartialMatch(c_str(), **iter)) {
            return true;
        }
    }
    return false;
}

xvector<xstring> xstring::Findall(const re2::RE2& FsRex) const
{
    xvector<xstring> retv;
    re2::StringPiece data_cpy = The;
    xstring out;
    while (re2::RE2::FindAndConsume(&data_cpy, FsRex, reinterpret_cast<std::string*>(&out)))
        retv << out;

    return retv;
}

xvector<xstring> xstring::Findwalk(const re2::RE2& FsRex) const
{
    xvector<xstring> retv;
    xvector<xstring> lines = The.Split('\n');
    for (const auto& line : lines) {
        for (const xstring& val : line.Findall(FsRex))
            retv << val;
    }
    return retv;
}

xstring xstring::Sub(const RE2& FsRex, const std::string& FsReplacement) const
{
    std::string ret = The.c_str();
    RE2::GlobalReplace(&ret, FsRex, FsReplacement);
    return ret;
}

xstring xstring::Sub(const RE2& FsRex, const re2::StringPiece& FsReplacement) const
{
    std::string ret = The.c_str();
    RE2::GlobalReplace(&ret, FsRex, FsReplacement);
    return ret;
}

xstring xstring::Sub(const RE2& FsRex, const char* FsReplacement) const
{
    std::string ret = The.c_str();
    RE2::GlobalReplace(&ret, FsRex, re2::StringPiece(FsReplacement));
    return ret;
}

xstring& xstring::InSub(const RE2& FsRex, const std::string& FsReplacement)
{
    RE2::GlobalReplace(this, FsRex, FsReplacement);
    return The;
}

xstring& xstring::InSub(const RE2& FsRex, const re2::StringPiece& FsReplacement)
{
    RE2::GlobalReplace(this, FsRex, FsReplacement);
    return The;
}

xstring& xstring::InSub(const RE2& FsRex, const char* FsReplacement)
{
    RE2::GlobalReplace(this, FsRex, FsReplacement);
    return The;
}

xstring& xstring::InRemove(const char FsRex)
{
    const auto LnSize = Size();
    xint LnNextIdx = 0;
    xint LnTargetIdx = 0;
    xint LnNumOfReplacements = 0;
    for (xint i = 0; LnNextIdx < (LnSize - 1); i++)
    {
        if (The[LnNextIdx] == FsRex)
        {
            ++LnNextIdx;
            ++LnNumOfReplacements;
        }
        else
            The[LnTargetIdx++] = The [LnNextIdx++];
    }

    for (xint i = LnSize; i > LnSize - LnNumOfReplacements; --i)
        The[i] = '\0';
    return The;
}

xstring xstring::Remove(const char val) const
{
    xstring Str;
    Str.reserve(The.size()+1);
    for (const char Chr : The)
        if (val != Chr)
            Str += Chr;
    return Str;
}

xstring& xstring::InSub(const char FsRex, const char FsReplacement)
{
    for (char& Chr : The)
        if (Chr == FsRex)
            Chr = FsReplacement;
    return The;
}

xstring& xstring::InSub(const char FsRex, const char* FsReplacement)
{
    const auto LnSize = strlen(FsReplacement);
    if (!LnSize)
        return The;
    for (xint i = Size(); i > 0; --i)
    {
        if (The[i] != FsRex)
            continue;
        The[i] = FsReplacement[0];
        if (LnSize > 1)
            The.insert(i, &FsReplacement[1]);
    }
    return The;
}

xstring& xstring::InSub(const char FsRex, const xstring& FsReplacement)
{
    const auto LnSize = FsReplacement.size();
    for (xint i = Size(); i > 0; --i)
    {
        if (The[i] != FsRex)
            continue;
        The[i] = FsReplacement[0];
        if (LnSize > 1)
            The.insert(i, &FsReplacement.c_str()[1]);
    }
    return The;
}

xstring xstring::IfFindReplace(const RE2& LoFind, const RE2& LoReplace, const std::string& LsReplacement) const
{
    if (RE2::PartialMatch(c_str(), LoFind))
    {
        std::string ret = The.c_str();
        RE2::GlobalReplace(&ret, LoReplace, re2::StringPiece(LsReplacement));
        return ret;
    }
    return The;
}
// #endif
// =========================================================================================================================

bool xstring::Match(const std::regex& FsRex) const {
    return bool(std::regex_match(c_str(), FsRex));
}


bool xstring::Match(const xstring& FsPattern, RXM::Type FeMod) const {
    return bool(std::regex_match(c_str(), std::regex(FsPattern.c_str(), FeMod)));
}

bool xstring::Match(const char* str, RXM::Type FeMod) const {
    return bool(std::regex_match(c_str(), std::regex(str, FeMod)));
}

bool xstring::MatchLine(const std::regex& FsRex) const
{
    xvector<xstring> lines = The.Split('\n');
    for (xvector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (std::regex_match(*iter, FsRex)) {
            return true;
        }
    }
    return false;
}

bool xstring::MatchLine(const xstring& FsPattern, RXM::Type FeMod) const {
    return The.MatchLine(std::regex(FsPattern.c_str(), FeMod));
}

bool xstring::MatchLine(const char* str, RXM::Type FeMod) const {
    return The.MatchLine(std::regex(str, FeMod));
}

bool xstring::MatchAllLines(const std::regex& FsRex) const
{
    xvector<xstring> lines = The.Split('\n');
    for (xvector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_match(*iter, FsRex)) {
            return false;
        }
    }
    return true;
}

bool xstring::MatchAllLines(const xstring& FsPattern, RXM::Type FeMod) const {
    return The.MatchAllLines(std::regex(FsPattern.c_str(), FeMod));
}

bool xstring::MatchAllLines(const char* str, RXM::Type FeMod) const {
    return The.MatchAllLines(std::regex(str, FeMod));
}

// =========================================================================================================================

bool xstring::Scan(const std::regex& FsRex) const {
    return bool(std::regex_search(The.c_str(), FsRex));
}

bool xstring::Scan(const char FsPattern, RXM::Type FeMod) const {
    return (std::find(The.begin(), The.end(), FsPattern) != The.end());
}

bool xstring::Scan(const xstring& FsPattern, RXM::Type FeMod) const {
    return bool(std::regex_search(c_str(), std::regex(FsPattern.c_str(), FeMod)));
}

bool xstring::Scan(const char* str, RXM::Type FeMod) const
{
    return bool(std::regex_search(c_str(), std::regex(str, FeMod)));
}

bool xstring::ScanLine(const std::regex& FsRex) const
{
    xvector<xstring> lines = The.Split('\n');
    for (xvector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++)
    {
        if (std::regex_search(*iter, FsRex))
            return true;
    }
    return false;
}

bool xstring::ScanLine(const xstring& FsPattern, RXM::Type FeMod) const
{
    std::regex FsRex(FsPattern.c_str(), FeMod);
    return The.ScanLine(FsRex);
}

bool xstring::ScanLine(const char* str, RXM::Type FeMod) const {
    return The.ScanLine(std::regex(str, FeMod));
}

bool xstring::ScanAllLines(const std::regex& FsRex) const
{
    xvector<xstring> lines = The.Split('\n');
    for (xvector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_search(*iter, FsRex))
            return false;
    }
    return true;
}

bool xstring::ScanAllLines(const xstring& FsPattern, RXM::Type FeMod) const {
    return The.ScanAllLines(std::regex(FsPattern.c_str(), FeMod));
}
bool xstring::ScanAllLines(const char* str, RXM::Type FeMod) const {
    return The.ScanAllLines(std::regex(str, FeMod));
}
// =========================================================================================================================


bool xstring::ScanList(const xvector<std::regex>& FvRex) const
{
    for (xvector<std::regex>::const_iterator iter = FvRex.begin(); iter != FvRex.end(); iter++)
    {
        if (std::regex_search(c_str(), *iter)) {
            return true;
        }
    }
    return false;
}

bool xstring::ScanList(const xvector<xstring>& lst, RXM::Type FeMod) const
{
    for (xvector<xstring>::const_iterator iter = lst.begin(); iter != lst.end(); iter++)
    {
        std::regex FsRex(*iter, FeMod);
        if (std::regex_search(c_str(), FsRex)) {
            return true;
        }
    }
    return false;
}


bool xstring::ScanList(const xvector<std::regex*>& FvRex) const
{
    for (xvector<std::regex*>::const_iterator iter = FvRex.begin(); iter != FvRex.end(); iter++)
    {
        if (std::regex_search(c_str(), **iter)) {
            return true;
        }
    }
    return false;
}

bool xstring::ScanList(const xvector<xstring*>& lst, RXM::Type FeMod) const
{
    for (xvector<xstring*>::const_iterator iter = lst.begin(); iter != lst.end(); iter++)
    {
        std::regex FsRex(**iter, FeMod);
        if (std::regex_search(c_str(), FsRex))
            return true;
    }
    return false;
}

// =========================================================================================================================

bool xstring::Is(const xstring& other) const {
    return (The) == other;
}

xint xstring::hash() const {
    return std::hash<std::string>()(The);
}

// =========================================================================================================================

int xstring::HasNonAscii(int front_skip, int end_skip, int threshold) const
{
    if (!The.size())
        return 0;

    for (xstring::const_iterator it = The.begin() + front_skip; it < The.end() - end_skip; it++) {
        if (!isascii(*it)) { // not always reliable but fast
            threshold--;
            if (threshold < 1)
                return static_cast<int>(*it);
        }
    }
    return 0;
}

bool xstring::HasNulls() const
{
    for (xstring::const_iterator it = The.begin(); it < The.end(); it++) {
        if (*it == '\0')
            return true;
    }
    return false;
}

bool xstring::HasDualNulls() const
{
    bool null_set = false;

    for (xstring::const_iterator it = The.begin(); it < The.end(); it++)
    {
        if (*it != '\0')
            null_set = false;

        else if (*it == '\0' && null_set == false)
            null_set = true;

        else if (*it == '\0' && null_set == true)
            return true;

    }
    return false;
}

void xstring::RemoveNulls()
{
    erase(std::find(begin(), end(), '\0'), end());
}

xstring xstring::RemoveNonAscii() const
{
    xstring clean_data;
    clean_data.reserve(The.size());
    for (xstring::const_iterator it = The.begin(); it < The.end(); it++) {
        if (isascii(*it))
            clean_data += *it;
    }
    return clean_data;
}

// =========================================================================================================================

xvector<xstring> xstring::Findall(const std::regex& FsRex) const
{
    xvector<xstring> retv;
    for (std::sregex_token_iterator iter(The.begin(), The.end(), FsRex, 1); iter != std::sregex_token_iterator(); ++iter)
        retv << xstring(*iter);
    return retv;
}

xvector<xstring> xstring::Findall(const xstring& FsPattern, RXM::Type FeMod) const {
    return The.Findall(std::regex(FsPattern, FeMod));
}

xvector<xstring> xstring::Findall(const char* FsPattern, RXM::Type FeMod) const {
    return The.Findall(std::regex(FsPattern, FeMod));
}

xvector<xstring> xstring::Findwalk(const std::regex& FsRex) const
{
    xvector<xstring> retv;
    xvector<xstring> lines = The.Split('\n');
    for (const auto& line : lines) {
        for (std::sregex_token_iterator iter(line.begin(), line.end(), FsRex, 1); iter != std::sregex_token_iterator(); ++iter)
            retv << xstring(*iter);
    }
    return retv;
}

xvector<xstring> xstring::Findwalk(const xstring& FsPattern, RXM::Type FeMod) const {
    return The.Findwalk(std::regex(FsPattern, FeMod));
}

xvector<xstring> xstring::Findwalk(const char* FsPattern, RXM::Type FeMod) const {
    return The.Findwalk(std::regex(FsPattern, FeMod));
}

xvector<xstring> xstring::Search(const std::regex& FsRex, int count) const
{
    xvector<xstring> retv;
    std::smatch matcher;
    if (std::regex_search(The, matcher, FsRex))
    {
        xint sz = matcher.size();
        std::smatch::iterator it = matcher.begin() + 1;
        while (it != matcher.end() && count != 0)
        {
            retv << xstring(*it);
            ++it;
            --count;
        }
    }
    return retv;
}

//xvector<xstring> xstring::Search(const re2::RE2& FsRex, int count) const{
//    return xvector<xstring>();
//}

xvector<xstring> xstring::Search(const xstring& FsPattern, RXM::Type FeMod, int count) const {
    return The.Search(std::regex(FsPattern, FeMod), count);
}

xvector<xstring> xstring::Search(const char* FsPattern, RXM::Type FeMod, int count) const {
    return The.Search(std::regex(FsPattern, FeMod), count);
}

// =========================================================================================================================

bool xstring::Has(const char var_char) const {
    if ((std::find(The.begin(), The.end(), var_char) != The.end()))
        return true;
    return false;
}

bool xstring::Lacks(const char var_char) const {

    if ((std::find(The.begin(), The.end(), var_char) != The.end()))
        return false;
    return true;
}

xint xstring::Count(const char var_char) const {
    xint n = std::count(The.begin(), The.end(), var_char);
    return n;
}

xint xstring::Count(const xstring& FsPattern) const {
    xint n = The.Split(FsPattern).size();
    return n;
}

// =========================================================================================================================

xstring xstring::Sub(const std::regex& FsRex, const std::string& FsReplacement) const {
    return std::regex_replace(c_str(), FsRex, FsReplacement);
}

xstring xstring::Sub(const std::string& FsPattern, const std::string& FsReplacement, RXM::Type FeMod) const {
    return std::regex_replace(c_str(), std::regex(FsPattern, FeMod), FsReplacement);
}

xstring xstring::Sub(const char* FsPattern, const std::string& FsReplacement, RXM::Type FeMod) const {
    return std::regex_replace(c_str(), std::regex(FsPattern, FeMod), FsReplacement);
}

xstring xstring::Sub(const char* FsPattern, const char* FsReplacement, RXM::Type FeMod) const {
    return std::regex_replace(c_str(), std::regex(FsPattern, FeMod), FsReplacement);
}


xstring xstring::IfFindReplace(const std::regex& LoFind, const std::regex& LoReplace, const std::string& LsReplacement) const
{
    if (std::regex_search(c_str(), LoFind))
        return std::regex_replace(c_str(), LoReplace, LsReplacement);
    return The;
}

xstring xstring::IfFindReplace(const std::string& LsFind, const std::string& LsReplace, const std::string& LsReplacement, RXM::Type FeMod) const
{
    if (std::regex_search(c_str(), std::regex(LsFind, FeMod)))
        return std::regex_replace(c_str(), std::regex(LsReplace, FeMod), LsReplacement);
    return The;
}

xstring xstring::IfFindReplace(const char* LsFind, const char* LsReplace, const std::string& LsReplacement, RXM::Type FeMod) const
{
    if (std::regex_search(c_str(), std::regex(LsFind, FeMod)))
        return std::regex_replace(c_str(), std::regex(LsReplace, FeMod), LsReplacement);
    return The;
}

xstring xstring::IfFindReplace(const char* LsFind, const char* LsReplace, const char* LsReplacement, RXM::Type FeMod) const
{
    if (std::regex_search(c_str(), std::regex(LsFind, FeMod)))
        return std::regex_replace(c_str(), std::regex(LsReplace, FeMod), LsReplacement);
    return The;
}


xstring& xstring::Trim()
{
    The.erase(0, The.find_first_not_of(" \t\r\n"));
    The.erase(The.find_last_not_of(" \t\r\n") + 1);
    return The;
}

xstring& xstring::LeftTrim()
{
    The.erase(0, The.find_first_not_of(" \t\r\n"));
    return The;
}

xstring& xstring::RightTrim()
{
    The.erase(The.find_last_not_of(" \t\r\n") + 1);
    return The;
}

xstring& xstring::Trim(const xstring& trim)
{
    The.erase(0, The.find_first_not_of(trim));
    The.erase(The.find_last_not_of(trim) + 1);
    return The;
}

xstring& xstring::LeftTrim(const xstring& trim)
{
    The.erase(0, The.find_first_not_of(trim));
    return The;
}

xstring& xstring::RightTrim(const xstring& trim)
{
    The.erase(The.find_last_not_of(trim) + 1);
    return The;
}

xstring xstring::operator()(const long long int x) const
{
    if (x == 0)
        return The;

    if (x > 0 && x >= The.size())
        return "";

    if (x < 0 && std::abs(x) > size())
        return The;

    if (x == 0)
        return The;
    else if (x > 0)
        return The.substr(x, size() - x);
    else
        return The.substr(size() + x, size() - (size() + x));
}

xstring xstring::operator()(
    const long long int x,
    const long long int y,
    const long long int z,
    const char removal_method) const
{
    const auto m_size = static_cast<long long int>(The.size());
    if (m_size <= 1)
        return The;

    xstring n_arr;
    n_arr.reserve(m_size + 4);

    if (z >= 0)
    {
        const auto tx = (x >= 0) ? x : m_size + x + 1;
        const auto ty = (y >= 0) ? y : m_size + y;

        typename xstring::const_iterator iter = The.begin() + tx;
        typename xstring::const_iterator stop = The.begin() + ty;

        if (z == 0) { // forward direction with no skipping
            for (; iter != stop; ++iter)
                n_arr.push_back(*iter);
        }
        else if (removal_method == 's') { // forward direction with skipping
            double iter_insert = 0;
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    n_arr.push_back(*iter);
                    iter_insert = z - 1;
                }
                else {
                    --iter_insert;
                }
            }
        }
        else {
            double iter_insert = 0;
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    iter_insert = z - 1;
                }
                else {
                    n_arr.push_back(*iter);
                    --iter_insert;
                }
            }
        }
    }
    else { // reverse direction
        const auto tz = z * (-1) - 1;
        const auto tx = (x >= 0) ? m_size - x - 1 : std::abs(x) - 1;
        const auto ty = (y >= 0) ? m_size - y - 1 : std::abs(y) - 1;

        typename xstring::const_reverse_iterator iter = The.rbegin() + tx;
        typename xstring::const_reverse_iterator stop = The.rbegin() + ty;

        double iter_insert = 0;

        if (z + 1 == 0) {
            for (; iter != stop; ++iter) {
                n_arr.push_back(*iter);
            }
        }
        else if (removal_method == 's') {
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    n_arr.push_back(*iter);
                    iter_insert = z + 1;
                }
                else {
                    --iter_insert;
                }
            }
        }
        else {
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    iter_insert = z + 1;
                }
                else {
                    n_arr.push_back(*iter);
                    --iter_insert;
                }
            }
        }
    }
    return n_arr;
}

// =========================================================================================================================

bool xstring::BxNumber() const
{
    if (!The.Size())
        return false;

    for (char Chr : The)
    {
        if (!(Chr == '0' || Chr == '1' || Chr == '2' || Chr == '3' || Chr == '4'
            || Chr == '5' || Chr == '6' || Chr == '7' || Chr == '8' || Chr == '9'))
            return false;
    }
    return true;
}


int xstring::ToInt() const
{
    return std::atoi(The.c_str());
}

long xstring::ToLong() const
{
    return std::atol(The.c_str());
}

long long xstring::ToLongLong() const
{
    return std::atoll(The.c_str());
}

xint xstring::To64() const
{
    return std::atoll(The.c_str());
}

double xstring::ToDouble() const
{
    return std::atof(The.c_str());
}

float xstring::ToFloat() const
{
    return static_cast<float>(std::atof(The.c_str()));
}

bool xstring::ToBool() const
{
    if (Size() == 0)
        return false;
    if (Size() == 1)
        return (bool)std::atoi(The.c_str());
    else if (ToLower() == "true")
        return true;
    else if (ToLower() == "false")
        return false;
    throw ("xstring::ToBool >> No Match");
}

// =================================================================================================================================

xstring xstring::ToBlack() const {
    return Color::Black + The;
}
xstring xstring::ToRed() const {
    return Color::Red + The;
}
xstring xstring::ToGreen() const {
    return Color::Green + The;
}
xstring xstring::ToYellow() const {
    return Color::Yellow + The;
}
xstring xstring::ToBlue() const {
    return Color::Blue + The;
}
xstring xstring::ToMegenta() const {
    return Color::Magenta + The;
}
xstring xstring::ToCyan() const {
    return Color::Cyan + The;
}
xstring xstring::ToGrey() const {
    return Color::Grey + The;
}
xstring xstring::ToWhite() const {
    return Color::White + The;
}
// --------------------------------------
xstring xstring::ToOnBlack() const {
    return Color::On::Black + The;
}
xstring xstring::ToOnRed() const {
    return Color::On::Red + The;
}
xstring xstring::ToOnGreen() const {
    return Color::On::Green + The;
}
xstring xstring::ToOnYellow() const {
    return Color::On::Yellow + The;
}
xstring xstring::ToOnBlue() const {
    return Color::On::Blue + The;
}
xstring xstring::ToOnMegenta() const {
    return Color::On::Magenta + The;
}
xstring xstring::ToOnCyan() const {
    return Color::On::Cyan + The;
}
xstring xstring::ToOnGrey() const {
    return Color::On::Grey + The;
}
xstring xstring::ToOnWhite() const {
    return Color::On::White + The;
}
// --------------------------------------
xstring xstring::ResetColor() const {
    return The + Color::Mod::Reset;
}
xstring xstring::ToBold() const {
    return Color::Mod::Bold + The;
}
xstring xstring::ToUnderline() const {
    return Color::Mod::Underline + The;
}
xstring xstring::ToInvertedColor() const {
    return Color::Mod::Reverse + The;
}

// =================================================================================================================================

xstring RA::WTXS(const wchar_t* wstr) {
    if (!wstr)
        return xstring::StaticClass;
    xstring str;
    const xint Max = wcslen(wstr);
    str.resize(Max);
    xint loc = 0;
    for (const wchar_t* ptr = wstr; Max > loc; ptr++)
        str[loc++] = static_cast<char>(*ptr);
    return str;
}

xstring RA::WTXS(const std::wstring& wstr) {
    if (!wstr.size())
        return xstring::StaticClass;
    xstring str;
    str.resize(wcslen(wstr.c_str()));
    xint loc = 0;
    for(auto Char : wstr)
        str[loc++] = static_cast<char>(Char); // can't use std::move between 2 different types
    return str;
}


// =================================================================================================================================

xstring operator+(const char First, const xstring& Second)
{
    return xstring(First) + Second;
}

xstring operator+(const char First, xstring&& Second)
{
    return xstring(First) + std::move(Second);
}

xstring operator+(const char* const First, const xstring& Second)
{
    return xstring(First) + Second;
}

xstring operator+(const char* const First, xstring&& Second)
{
    return xstring(First) + std::move(Second);
}
// ---------------------------------------------------------------
xstring operator+(const char First, const std::wstring& Second)
{
    return xstring(First) + RA::WTXS(Second);
}

xstring operator+(const char First, std::wstring&& Second)
{
    return xstring(First) + RA::WTXS(std::move(Second));
}

xstring operator+(const char* const First, const std::wstring& Second)
{
    return xstring(First) + RA::WTXS(Second);
}

xstring operator+(const char* const First, std::wstring&& Second)
{
    return xstring(First) + RA::WTXS(std::move(Second));
}

// ---------------------------------------------------------------
