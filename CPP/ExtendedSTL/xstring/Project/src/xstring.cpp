#pragma warning (disable : 26444) // allow anynomous objects
#pragma warning (disable : 26812) // allow normal enums from STL

#include "xstring.h"

#include <stdlib.h>

const xstring xstring::static_class;

xstring Color::Black = "\033[30m";
xstring Color::Red = "\033[31m";
xstring Color::Green = "\033[32m";
xstring Color::Yellow = "\033[33m";
xstring Color::Blue = "\033[34m";
xstring Color::Magenta = "\033[35m";
xstring Color::Cyan = "\033[36m";
xstring Color::Grey = "\033[37m";
xstring Color::White = "\033[39m";

xstring Color::On::Black = "\033[40m";
xstring Color::On::Red = "\033[41m";
xstring Color::On::Green = "\033[42m";
xstring Color::On::Yellow = "\033[43m";
xstring Color::On::Blue = "\033[44m";
xstring Color::On::Magenta = "\033[45m";
xstring Color::On::Cyan = "\033[46m";
xstring Color::On::Grey = "\033[47m";
xstring Color::On::White = "\033[49m";

xstring Color::Mod::Reset = "\033[00m";
xstring Color::Mod::Bold = "\033[01m";
xstring Color::Mod::Underline = "\033[04m";
xstring Color::Mod::Reverse = "\033[07m";

// Operates only on Linux 
xstring Color::Mod::Dark = "\033[02m";
xstring Color::Mod::Blink = "\033[05m";
xstring Color::Mod::Hide = "\033[08m";

xstring::xstring(const char chr)
{
    The.insert(The.begin(), chr);
}

xstring::xstring(const char* chrs)
{
    size_t len = strlen(chrs) + 1;
    The.resize(len);
    The.insert(The.begin(), chrs, &chrs[len]);
    RemoveNulls();
}

xstring::xstring(const unsigned char* chrs)
{
    //for (const unsigned char* ptr = chrs; static_cast<char>(*ptr) != '\0'; ptr++)
    //    The.insert(The.end(), static_cast<char>(*ptr));
    size_t len = strlen(reinterpret_cast<const char*>(chrs));
    The.resize(len);
    The.insert(The.begin(), chrs, &chrs[len]);
    // RemoveNulls(); // unsigned char is often uesd for byte-arrays
}

xstring::xstring(const unsigned char* chrs, size_t len)
{
    // Does not Terminate Correctly
    //The.reserve(len);
    //The.insert(The.begin(), chrs, &chrs[len]);

    // Does Not Size correctly (resize may add more size than you want)
    //The.resize(len);
    //size_t Start = 0;
    //const unsigned char* ptr = chrs;
    //for (size_t idx = 0; idx < len; idx++)
    //    The[idx] = static_cast<char>(*(ptr++));
    //The[len] = '\0';

    The.reserve(len);
    size_t Start = 0;
    const unsigned char* ptr = chrs;
    for (size_t idx = 0; idx < len; idx++)
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

void xstring::operator+=(std::string&& str){
    The.insert(The.end(), std::make_move_iterator(str.begin()), std::make_move_iterator(str.end()));
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

const char& xstring::At(size_t Idx) const
{
    return at(Idx);
}

char& xstring::At(size_t Idx)
{
    return at(Idx);
}

const char xstring::First(size_t Idx) const
{
    if (!(This.Size() - 1 >= Idx))
        throw "Index Out Of Range";
    return The.operator[](Idx);
}

char& xstring::First(size_t Idx)
{
    if (!(This.Size() - 1 >= Idx))
        throw "Index Out Of Range";
    return The.operator[](Idx);
}

const char xstring::Last(size_t Idx) const
{
    if (!(This.Size() - 1 >= Idx))
        throw "Index Out Of Range";
    return The.operator[](this->size() - Idx - 1);
}

char& xstring::Last(size_t Idx)
{
    if (!(This.Size() - 1 >= Idx))
        throw "Index Out Of Range";
    return The.operator[](this->size() - Idx - 1);
}

size_t xstring::Size() const
{
    return size();
}

const char* xstring::Ptr() const
{
    return c_str();
}

void xstring::Print() const
{
    std::cout << The << '\n';
}

void xstring::Print(int num) const
{
    std::cout << The;
    char* new_lines = static_cast<char*>(calloc(static_cast<size_t>(num) + 1, sizeof(char)));
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
    if(strlen(chr2))
        std::cout << chr1 << The << chr2 << '\n';
    else
        std::cout << The << chr1 << '\n';
}

std::string xstring::ToStdString() const {
    return std::string(The.c_str()); // todo: return only from the base class
}

std::wstring xstring::ToStdWString() const
{
    std::wstring LsWideStr(size(), L' ');
    for(size_t i = 0; i < size(); i++)
        LsWideStr[i] = static_cast<wchar_t>((The)[i]);

    return LsWideStr;
}

RA::SharedPtr<unsigned char[]> xstring::ToUnsignedChar() const
{
    const auto LnSize = The.Size();
    auto UnsignedChar = MKPA<unsigned char>(LnSize);
    for (int i = 0; i < LnSize; i++)
        UnsignedChar[i] = The[i];
    return UnsignedChar;
}

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

bool xstring::IsByteCode() const
{
    static const RE2 ByteStringRex(R"(^(\\x[A-Fa-f0-9]{2})*$)");
    if (The.Match(ByteStringRex))
        return true;
    return false;
}

xstring xstring::FromByteCodeToASCII() const
{
    xstring Ascii = "";
    xstring part;
    part.resize(2);
    for (size_t i = 0; i < The.length() - 2; i += 4)
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
        }else
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

xstring xstring::Remove(const char val) const
{
    xstring str;
    str.reserve(The.size());
    for (xstring::const_iterator it = The.begin(); it != The.end(); it++)
    {
        if (val != *it)
            str += *it;
    }
    return str;
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

xvector<xstring> xstring::SingleSplit(size_t loc) const
{
    xvector<xstring> ret_vec;
    ret_vec.reserve(4);
    ret_vec << The.substr(0, loc);
    ret_vec << The.substr(loc, The.size() - loc);

    return ret_vec;
}

xvector<xstring> xstring::Split(size_t SegSize) const
{
    if (!SegSize)
        throw "Seg Size is Zero";
    xvector<xstring> Vec;
    Vec.reserve(This.Size() / SegSize + 1);
    xstring CurrentStr;
    CurrentStr.resize(SegSize + 1);
    size_t Idx = 0;
    for (xstring::const_iterator It = This.cbegin(); It != This.cend(); It++)
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
    if(This.Size() % SegSize > 0)
        Vec << CurrentStr.substr(0, Idx);
    return Vec;
}

xvector<xstring> xstring::Split(const std::regex& rex) const
{
    xvector<xstring> split_content;
    for (std::sregex_token_iterator iter(The.begin(), The.end(), rex, -1); iter != std::sregex_token_iterator(); ++iter)
        split_content << xstring(*iter);

    return split_content;
}

xvector<xstring> xstring::Split(const xstring& pattern, rxm::type mod) const {
    return The.Split(std::regex(pattern.c_str(), rxm::ECMAScript | mod));
}

xvector<xstring> xstring::Split(const char splitter, rxm::type mod) const {
    return The.Split(xstring({ splitter }), mod);
}

xvector<xstring> xstring::InclusiveSplit(const std::regex& rex, bool single) const
{
    xvector<xstring> retv;
    for (std::sregex_token_iterator iter(The.begin(), The.end(), rex, { -1, 0 }); iter != std::sregex_token_iterator(); ++iter)
        retv << xstring(*iter);

    if (!single) {
        if (retv.size() == 1)
            return xvector<xstring>();
    }
    return retv;
}


xvector<xstring> xstring::InclusiveSplit(const xstring& splitter, rxm::type mod, bool single) const
{
    std::regex rex(splitter.c_str(), rxm::ECMAScript | mod);
    // -1       grab NOT regex
    //  0       grab regex
    //  1       grab blank
    return The.InclusiveSplit(rex, single);
}

xvector<xstring> xstring::InclusiveSplit(const char* splitter, rxm::type mod, bool aret) const {
    return The.InclusiveSplit(xstring(splitter), mod, aret);
}

xvector<xstring> xstring::InclusiveSplit(const char splitter, rxm::type mod, bool aret) const {
    return The.InclusiveSplit(xstring({ splitter }), mod, aret);
}

// =========================================================================================================================

bool xstring::Match(const std::regex& rex) const {
    return bool(std::regex_match(c_str(), rex));
}

bool xstring::Match(const re2::RE2& rex) const {
    return RE2::FullMatch(c_str(), rex);
}

bool xstring::Match(const xstring& pattern, rxm::type mod) const {
    return bool(std::regex_match(c_str(), std::regex(pattern.c_str(), rxm::ECMAScript | mod)));
}

bool xstring::Match(const char* str, rxm::type mod) const {
    return bool(std::regex_match(c_str(), std::regex(str, mod)));
}

bool xstring::MatchLine(const std::regex& rex) const
{
    std::vector<xstring> lines = The.Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (std::regex_match(*iter, rex)) {
            return true;
        }
    }
    return false;
}

bool xstring::MatchLine(const re2::RE2& rex) const
{
    std::vector<xstring> lines = The.Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (RE2::FullMatch(*iter, rex)) {
            return true;
        }
    }
    return false;
}

bool xstring::MatchLine(const xstring& pattern, rxm::type mod) const {
    return The.MatchLine(std::regex(pattern.c_str(), rxm::ECMAScript | mod));
}

bool xstring::MatchLine(const char* str, rxm::type mod) const {
    return The.MatchLine(std::regex(str, mod));
}

bool xstring::MatchAllLines(const std::regex& rex) const
{
    std::vector<xstring> lines = The.Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_match(*iter, rex)) {
            return false;
        }
    }
    return true;
}

bool xstring::MatchAllLines(const re2::RE2& rex) const
{
    std::vector<xstring> lines = The.Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!RE2::FullMatch(*iter, rex)) {
            return false;
        }
    }
    return true;
}

bool xstring::MatchAllLines(const xstring& pattern, rxm::type mod) const {
    return The.MatchAllLines(std::regex(pattern.c_str(), rxm::ECMAScript | mod));
}

bool xstring::MatchAllLines(const char* str, rxm::type mod) const {
    return The.MatchAllLines(std::regex(str, rxm::ECMAScript | mod));
}

// =========================================================================================================================

bool xstring::Scan(const std::regex& rex) const {
    return bool(std::regex_search(The.c_str(), rex));
}

bool xstring::Scan(const re2::RE2& rex) const {
    return re2::RE2::PartialMatch(The.c_str(), rex);
}

bool xstring::Scan(const char pattern, rxm::type mod) const {
    return (std::find(The.begin(), The.end(), pattern) != The.end());
}

bool xstring::Scan(const xstring& pattern, rxm::type mod) const {
    return bool(std::regex_search(c_str(), std::regex(pattern.c_str(), rxm::ECMAScript | mod)));
}

bool xstring::Scan(const char* str, rxm::type mod) const
{
    return bool(std::regex_search(c_str(), std::regex(str, rxm::ECMAScript | mod)));
}

bool xstring::ScanLine(const std::regex& rex) const
{
    std::vector<xstring> lines = The.Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++)
    {
        if (std::regex_search(*iter, rex))
            return true;
    }
    return false;
}

bool xstring::ScanLine(const re2::RE2& rex) const
{
    std::vector<xstring> lines = The.Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++)
    {
        if (re2::RE2::PartialMatch(*iter, rex))
            return true;
    }
    return false;
}

bool xstring::ScanLine(const xstring& pattern, rxm::type mod) const
{
    std::regex rex(pattern.c_str(), rxm::ECMAScript | mod);
    return The.ScanLine(rex);
}

bool xstring::ScanLine(const char* str, rxm::type mod) const {
    return The.ScanLine(std::regex(str, rxm::ECMAScript | mod));
}

bool xstring::ScanAllLines(const std::regex& rex) const
{
    std::vector<xstring> lines = The.Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_search(*iter, rex))
            return false;
    }
    return true;
}

bool xstring::ScanAllLines(const re2::RE2& rex) const
{
    std::vector<xstring> lines = The.Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!re2::RE2::PartialMatch(*iter, rex))
            return false;
    }
    return true;
}

bool xstring::ScanAllLines(const xstring& pattern, rxm::type mod) const {
    return The.ScanAllLines(std::regex(pattern.c_str(), rxm::ECMAScript | mod));
}
bool xstring::ScanAllLines(const char* str, rxm::type mod) const {
    return The.ScanAllLines(std::regex(str, rxm::ECMAScript | mod));
}
// =========================================================================================================================


bool xstring::ScanList(const xvector<std::regex>& rex_lst) const
{
    for (std::vector<std::regex>::const_iterator iter = rex_lst.begin(); iter != rex_lst.end(); iter++)
    {
        if (std::regex_search(c_str(), *iter)) {
            return true;
        }
    }
    return false;
}

bool xstring::ScanList(const xvector<re2::RE2>& rex_lst) const
{
    for (std::vector<re2::RE2>::const_iterator iter = rex_lst.begin(); iter != rex_lst.end(); iter++)
    {
        if (re2::RE2::PartialMatch(c_str(), *iter)) {
            return true;
        }
    }
    return false;
}

bool xstring::ScanList(const xvector<xstring>& lst, rxm::type mod) const
{
    for (std::vector<xstring>::const_iterator iter = lst.begin(); iter != lst.end(); iter++)
    {
        std::regex rex(*iter, rxm::ECMAScript | mod);
        if (std::regex_search(c_str(), rex)) {
            return true;
        }
    }
    return false;
}


bool xstring::ScanList(const xvector<std::regex*>& rex_lst) const
{
    for (std::vector<std::regex*>::const_iterator iter = rex_lst.begin(); iter != rex_lst.end(); iter++)
    {
        if (std::regex_search(c_str(), **iter)) {
            return true;
        }
    }
    return false;
}

bool xstring::ScanList(const xvector<re2::RE2*>& rex_lst) const
{
    for (std::vector<re2::RE2*>::const_iterator iter = rex_lst.begin(); iter != rex_lst.end(); iter++)
    {
        if (re2::RE2::PartialMatch(c_str(), **iter)) {
            return true;
        }
    }
    return false;
}

bool xstring::ScanList(const xvector<xstring*>& lst, rxm::type mod) const
{
    for (std::vector<xstring*>::const_iterator iter = lst.begin(); iter != lst.end(); iter++)
    {
        std::regex rex(**iter, rxm::ECMAScript | mod);
        if (std::regex_search(c_str(), rex))
            return true;
    }
    return false;
}

// =========================================================================================================================

bool xstring::Is(const xstring& other) const {
    return (The) == other;
}

size_t xstring::hash() const {
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

xvector<xstring> xstring::Findall(const std::regex& rex) const
{
    xvector<xstring> retv;
    for (std::sregex_token_iterator iter(The.begin(), The.end(), rex, 1); iter != std::sregex_token_iterator(); ++iter)
        retv << xstring(*iter);
    return retv;
}

xvector<xstring> xstring::Findall(const re2::RE2& rex) const
{
    xvector<xstring> retv;
    re2::StringPiece data_cpy = The;
    xstring out;
    while (re2::RE2::FindAndConsume(&data_cpy, rex, reinterpret_cast<std::string*>(&out)))
        retv << out;

    return retv;
}

xvector<xstring> xstring::Findall(const xstring& pattern, rxm::type mod) const {
    return The.Findall(std::regex(pattern, rxm::ECMAScript | mod));
}

xvector<xstring> xstring::Findall(const char* pattern, rxm::type mod) const {
    return The.Findall(std::regex(pattern, mod));
}

xvector<xstring> xstring::Findwalk(const std::regex& rex) const
{
    xvector<xstring> retv;
    xvector<xstring> lines = The.Split('\n');
    for (const auto& line : lines) {
        for (std::sregex_token_iterator iter(line.begin(), line.end(), rex, 1); iter != std::sregex_token_iterator(); ++iter)
            retv << xstring(*iter);
    }
    return retv;
}

xvector<xstring> xstring::Findwalk(const re2::RE2& rex) const
{
    xvector<xstring> retv;
    xvector<xstring> lines = The.Split('\n');
    for (const auto& line : lines) {
        for (const xstring& val : line.Findall(rex))
            retv << val;
    }
    return retv;
}

xvector<xstring> xstring::Findwalk(const xstring& pattern, rxm::type mod) const {
    return The.Findwalk(std::regex(pattern, rxm::ECMAScript | mod));
}

xvector<xstring> xstring::Findwalk(const char* pattern, rxm::type mod) const {
    return The.Findwalk(std::regex(pattern, rxm::ECMAScript | mod));
}

xvector<xstring> xstring::Search(const std::regex& rex, int count) const
{
    xvector<xstring> retv;
    std::smatch matcher;
    if (std::regex_search(The, matcher, rex))
    {
        size_t sz = matcher.size();
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

//xvector<xstring> xstring::Search(const re2::RE2& rex, int count) const{
//    return xvector<xstring>();
//}

xvector<xstring> xstring::Search(const xstring& pattern, rxm::type mod, int count) const {
    return The.Search(std::regex(pattern, mod), count);
}

xvector<xstring> xstring::Search(const char* pattern, rxm::type mod, int count) const {
    return The.Search(std::regex(pattern, mod), count);
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

size_t xstring::Count(const char var_char) const {
    size_t n = std::count(The.begin(), The.end(), var_char);
    return n;
}

size_t xstring::Count(const xstring& pattern) const {
    size_t n = The.Split(pattern).size();
    return n;
}

// =========================================================================================================================

xstring xstring::Sub(const std::regex& rex, const std::string& replacement) const {
    return std::regex_replace(c_str(), rex, replacement);
}

xstring xstring::Sub(const RE2& rex, const std::string& replacement) const
{
    xstring ret = The;
    RE2::GlobalReplace(reinterpret_cast<std::string*>(&ret), rex, replacement);
    return ret;
}

xstring xstring::Sub(const RE2& rex, const re2::StringPiece& replacement) const
{
    xstring ret = The;
    RE2::GlobalReplace(reinterpret_cast<std::string*>(&ret), rex, replacement);
    return ret;
}

xstring xstring::Sub(const RE2& rex, const char* replacement) const
{
    xstring ret = The;
    RE2::GlobalReplace(reinterpret_cast<std::string*>(&ret), rex, re2::StringPiece(replacement));
    return ret;
}

xstring xstring::Sub(const std::string& pattern, const std::string& replacement, rxm::type mod) const {
    return std::regex_replace(c_str(), std::regex(pattern, rxm::ECMAScript | mod), replacement);
}

xstring xstring::Sub(const char* pattern, const std::string& replacement, rxm::type mod) const {
    return std::regex_replace(c_str(), std::regex(pattern, rxm::ECMAScript | mod), replacement);
}

xstring xstring::Sub(const char* pattern, const char* replacement, rxm::type mod) const {
    return std::regex_replace(c_str(), std::regex(pattern, rxm::ECMAScript | mod), replacement);
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

xstring xstring::operator()(long double x, long double y, long double z, const char removal_method) const
{
    size_t m_size = The.size();
    xstring n_arr;
    n_arr.reserve(m_size + 4);

    double n_arr_size = static_cast<double>(m_size) - 1;

    if (z >= 0) {

        if (x < 0) { x += n_arr_size; }

        if (!y) { y = n_arr_size; }
        else if (y < 0) { y += n_arr_size; }
        ++y;

        if (x > y) { return n_arr; }

        typename xstring::const_iterator iter = The.begin();
        typename xstring::const_iterator stop = The.begin() + static_cast<ull>(y);

        if (z == 0) { // forward direction with no skipping
            for (iter += static_cast<ull>(x); iter != stop; ++iter)
                n_arr.push_back(*iter);
        }
        else if (removal_method == 's') { // forward direction with skipping
            double iter_insert = 0;
            --z;
            for (iter += static_cast<unsigned long long>(x); iter != stop; ++iter) {
                if (!iter_insert) {
                    n_arr.push_back(*iter);
                    iter_insert = z;
                }
                else {
                    --iter_insert;
                }
            }
        }
        else {
            double iter_insert = 0;
            --z;
            for (iter += static_cast<ull>(x); iter != stop; ++iter) {
                if (!iter_insert) {
                    iter_insert = z;
                }
                else {
                    n_arr.push_back(*iter);
                    --iter_insert;
                }
            }
        }
    }
    else { // reverse direction
        z = z * -1 - 1;
        //z = static_cast<size_t>(z = z * -1 - 1);
        if (!x) { x = n_arr_size; }
        else if (x < 0) { x += n_arr_size; }

        if (!y) { y = 0; }
        else if (y < 0) { y += n_arr_size; }

        if (y > x) { return n_arr; }

        // x = static_cast<size_t>(x);
        // y = static_cast<size_t>(y);

        typename xstring::const_reverse_iterator iter = The.rend() - static_cast<ull>(x) - 1;
        typename xstring::const_reverse_iterator stop = The.rend() - static_cast<ull>(y);

        double iter_insert = 0;

        if (z == 0) {
            for (; iter != stop; ++iter) {
                if (!iter_insert)
                    n_arr.push_back(*iter);
            }
        }
        else if (removal_method == 's') {
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    n_arr.push_back(*iter);
                    iter_insert = z;
                }
                else {
                    --iter_insert;
                }
            }
        }
        else {
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    iter_insert = z;
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

size_t xstring::To64() const
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
        return xstring::static_class;
    xstring str;
    const size_t Max = wcslen(wstr);
    str.resize(Max);
    size_t loc = 0;
    for (const wchar_t* ptr = wstr; Max > loc; ptr++) {
        str[loc] = static_cast<char>(*ptr);
        loc++;
    }
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