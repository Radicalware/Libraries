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
    this->insert(this->begin(), chr);
}

xstring::xstring(const char* chrs)
{
    size_t len = strlen(chrs) + 1;
    this->resize(len);
    this->insert(this->begin(), chrs, &chrs[len]);
    RemoveNulls();
}

xstring::xstring(const unsigned char* chrs)
{
    size_t len = strlen(reinterpret_cast<const char*>(chrs));
    this->resize(len);
    this->insert(this->begin(), chrs, &chrs[len]);
    RemoveNulls();
}

void xstring::operator+=(const char chr)
{
    this->insert(this->end(), chr);
}

xstring xstring::operator+(const char chr)
{
    xstring rstr = *this;
    rstr.insert(rstr.end(), chr);
    return rstr;
}

void xstring::operator+=(const char* chr)
{
    *this += xstring(chr);
}

xstring xstring::operator+(const char* chr)
{
    xstring retr;
    retr.reserve(this->size() + strlen(chr));
    retr += *this;
    retr += chr;
    return retr;
}

void xstring::operator+=(const unsigned char* chr)
{
    *this += xstring(chr);
}

xstring xstring::operator+(const unsigned char* chr)
{
    xstring retr;
    retr.reserve(this->size() + strlen(reinterpret_cast<const char*>(chr)));
    retr += *this;
    retr += chr;
    return retr;
}

void xstring::operator+=(const std::string& str) {
    this->insert(this->end(), str.begin(), str.end());
}

xstring xstring::operator+(const std::string& str)
{
    xstring rstr;
    rstr.reserve(this->size() + str.size());
    rstr += *this;
    rstr += str;
    return rstr;
}

void xstring::operator+=(std::string&& str)
{
    this->insert(this->end(), std::make_move_iterator(str.begin()), std::make_move_iterator(str.end()));
}

xstring xstring::operator+(std::string&& str)
{
    xstring rstr;
    rstr.reserve(this->size() + str.size());
    rstr += *this;
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
    std::cout << *this << '\n';
}

void xstring::Print(int num) const
{
    std::cout << *this;
    char* new_lines = static_cast<char*>(calloc(static_cast<size_t>(num) + 1, sizeof(char)));
    // calloc was used instead of "new" because "new" would give un-wanted after-effects.
    for (int i = 0; i < num; i++)
#pragma warning(suppress:6011) // we are derferencing a pointer, but assigning it a value at the same time
        new_lines[i] = '\n';
    std::cout << new_lines;
    free(new_lines);
}

void xstring::Print(const xstring& front, const xstring& end) const {
    std::cout << front << *this << end << '\n';
}

void xstring::Print(const char chr1, const char chr2) const
{
    std::cout << chr1 << *this << chr2 << '\n';
}

void xstring::Print(const char* chr1, const char* chr2) const
{
    std::cout << chr1 << *this << chr2 << '\n';
}

std::string xstring::ToStdString() const {
    return std::string(this->c_str()); // todo: return only from the base class
}

std::wstring xstring::ToStdWString() const
{
    std::wstring LsWideStr(size(), L' ');
    for(size_t i = 0; i < size(); i++)
        LsWideStr[i] = static_cast<wchar_t>((*this)[i]);

    return LsWideStr;
}

xstring xstring::ToUpper() const
{
    xstring ret_str;
    std::locale loc;
    for (xstring::const_iterator it = this->begin(); it != this->end(); it++)
        ret_str += std::toupper((*it), loc);
    return ret_str;
}

xstring xstring::ToLower() const
{
    xstring ret_str;
    std::locale loc;
    for (xstring::const_iterator it = this->begin(); it != this->end(); it++)
        ret_str += std::tolower((*it), loc);
    return ret_str;
}

xstring xstring::operator*(int total) const
{
    xstring ret_str = *this;
    for (int i = 0; i < total; i++)
        ret_str.insert(ret_str.end(), this->begin(), this->end());
    return ret_str;
}

void xstring::operator*=(int total)
{
    xstring str_copy = *this;
    for (int i = 0; i < total; i++)
        this->insert(this->end(), str_copy.begin(), str_copy.end());
}

xstring xstring::Remove(const char val) const
{
    xstring str;
    str.reserve(this->size());
    for (xstring::const_iterator it = this->begin(); it != this->end(); it++)
    {
        if (val != *it)
            str += *it;
    }
    return str;
}

xstring xstring::Reverse() const
{
    xstring str;
    str.resize(this->size());

    xstring::const_reverse_iterator from_it = this->rbegin();
    xstring::iterator to_it = str.begin();

    for (; from_it != this->rend();)
    {
        *to_it = *from_it;
        from_it++;
        to_it++;
    }
    return str;
}

// ---------------------------------------------------------------

xvector<xstring> xstring::Split(size_t loc) const
{
    xvector<xstring> ret_vec;
    ret_vec.reserve(4);
    ret_vec << this->substr(0, loc);
    ret_vec << this->substr(loc, this->size() - loc);

    return ret_vec;
}

xvector<xstring> xstring::Split(const std::regex& rex) const
{
    xvector<xstring> split_content;
    for (std::sregex_token_iterator iter(this->begin(), this->end(), rex, -1); iter != std::sregex_token_iterator(); ++iter)
        split_content << xstring(*iter);

    return split_content;
}

xvector<xstring> xstring::Split(const xstring& pattern, rxm::type mod) const {
    return this->Split(std::regex(pattern.c_str(), rxm::ECMAScript | mod));
}

xvector<xstring> xstring::Split(const char splitter, rxm::type mod) const {
    return this->Split(xstring({ splitter }), mod);
}

xvector<xstring> xstring::InclusiveSplit(const std::regex& rex, bool single) const
{
    xvector<xstring> retv;
    for (std::sregex_token_iterator iter(this->begin(), this->end(), rex, { -1, 0 }); iter != std::sregex_token_iterator(); ++iter)
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
    return this->InclusiveSplit(rex, single);
}

xvector<xstring> xstring::InclusiveSplit(const char* splitter, rxm::type mod, bool aret) const {
    return this->InclusiveSplit(xstring(splitter), mod, aret);
}

xvector<xstring> xstring::InclusiveSplit(const char splitter, rxm::type mod, bool aret) const {
    return this->InclusiveSplit(xstring({ splitter }), mod, aret);
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
    std::vector<xstring> lines = this->Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (std::regex_match(*iter, rex)) {
            return true;
        }
    }
    return false;
}

bool xstring::MatchLine(const re2::RE2& rex) const
{
    std::vector<xstring> lines = this->Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (RE2::FullMatch(*iter, rex)) {
            return true;
        }
    }
    return false;
}

bool xstring::MatchLine(const xstring& pattern, rxm::type mod) const {
    return this->MatchLine(std::regex(pattern.c_str(), rxm::ECMAScript | mod));
}

bool xstring::MatchLine(const char* str, rxm::type mod) const {
    return this->MatchLine(std::regex(str, mod));
}

bool xstring::MatchAllLines(const std::regex& rex) const
{
    std::vector<xstring> lines = this->Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_match(*iter, rex)) {
            return false;
        }
    }
    return true;
}

bool xstring::MatchAllLines(const re2::RE2& rex) const
{
    std::vector<xstring> lines = this->Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!RE2::FullMatch(*iter, rex)) {
            return false;
        }
    }
    return true;
}

bool xstring::MatchAllLines(const xstring& pattern, rxm::type mod) const {
    return this->MatchAllLines(std::regex(pattern.c_str(), rxm::ECMAScript | mod));
}

bool xstring::MatchAllLines(const char* str, rxm::type mod) const {
    return this->MatchAllLines(std::regex(str, rxm::ECMAScript | mod));
}

// =========================================================================================================================

bool xstring::Scan(const std::regex& rex) const {
    return bool(std::regex_search(this->c_str(), rex));
}

bool xstring::Scan(const re2::RE2& rex) const {
    return re2::RE2::PartialMatch(this->c_str(), rex);
}

bool xstring::Scan(const char pattern, rxm::type mod) const {
    return (std::find(this->begin(), this->end(), pattern) != this->end());
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
    std::vector<xstring> lines = this->Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++)
    {
        if (std::regex_search(*iter, rex))
            return true;
    }
    return false;
}

bool xstring::ScanLine(const re2::RE2& rex) const
{
    std::vector<xstring> lines = this->Split('\n');
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
    return this->ScanLine(rex);
}

bool xstring::ScanLine(const char* str, rxm::type mod) const {
    return this->ScanLine(std::regex(str, rxm::ECMAScript | mod));
}

bool xstring::ScanAllLines(const std::regex& rex) const
{
    std::vector<xstring> lines = this->Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_search(*iter, rex))
            return false;
    }
    return true;
}

bool xstring::ScanAllLines(const re2::RE2& rex) const
{
    std::vector<xstring> lines = this->Split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!re2::RE2::PartialMatch(*iter, rex))
            return false;
    }
    return true;
}

bool xstring::ScanAllLines(const xstring& pattern, rxm::type mod) const {
    return this->ScanAllLines(std::regex(pattern.c_str(), rxm::ECMAScript | mod));
}
bool xstring::ScanAllLines(const char* str, rxm::type mod) const {
    return this->ScanAllLines(std::regex(str, rxm::ECMAScript | mod));
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
    return (*this) == other;
}

size_t xstring::hash() const {
    return std::hash<std::string>()(*this);
}

// =========================================================================================================================

int xstring::HasNonAscii(int front_skip, int end_skip, int threshold) const
{
    if (!this->size())
        return 0;

    for (xstring::const_iterator it = this->begin() + front_skip; it < this->end() - end_skip; it++) {
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
    for (xstring::const_iterator it = this->begin(); it < this->end(); it++) {
        if (*it == '\0')
            return true;
    }
    return false;
}

bool xstring::HasDualNulls() const
{
    bool null_set = false;

    for (xstring::const_iterator it = this->begin(); it < this->end(); it++)
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
    clean_data.reserve(this->size());
    for (xstring::const_iterator it = this->begin(); it < this->end(); it++) {
        if (isascii(*it))
            clean_data += *it;
    }
    return clean_data;
}

// =========================================================================================================================

xvector<xstring> xstring::Findall(const std::regex& rex) const
{
    xvector<xstring> retv;
    for (std::sregex_token_iterator iter(this->begin(), this->end(), rex, 1); iter != std::sregex_token_iterator(); ++iter)
        retv << xstring(*iter);
    return retv;
}

xvector<xstring> xstring::Findall(const re2::RE2& rex) const
{
    xvector<xstring> retv;
    re2::StringPiece data_cpy = *this;
    xstring out;
    while (re2::RE2::FindAndConsume(&data_cpy, rex, reinterpret_cast<std::string*>(&out)))
        retv << out;

    return retv;
}

xvector<xstring> xstring::Findall(const xstring& pattern, rxm::type mod) const {
    return this->Findall(std::regex(pattern, rxm::ECMAScript | mod));
}

xvector<xstring> xstring::Findall(const char* pattern, rxm::type mod) const {
    return this->Findall(std::regex(pattern, mod));
}

xvector<xstring> xstring::Findwalk(const std::regex& rex) const
{
    xvector<xstring> retv;
    xvector<xstring> lines = this->Split('\n');
    for (const auto& line : lines) {
        for (std::sregex_token_iterator iter(line.begin(), line.end(), rex, 1); iter != std::sregex_token_iterator(); ++iter)
            retv << xstring(*iter);
    }
    return retv;
}

xvector<xstring> xstring::Findwalk(const re2::RE2& rex) const
{
    xvector<xstring> retv;
    xvector<xstring> lines = this->Split('\n');
    for (const auto& line : lines) {
        for (const xstring& val : line.Findall(rex))
            retv << val;
    }
    return retv;
}

xvector<xstring> xstring::Findwalk(const xstring& pattern, rxm::type mod) const {
    return this->Findwalk(std::regex(pattern, rxm::ECMAScript | mod));
}

xvector<xstring> xstring::Findwalk(const char* pattern, rxm::type mod) const {
    return this->Findwalk(std::regex(pattern, rxm::ECMAScript | mod));
}

xvector<xstring> xstring::Search(const std::regex& rex, int count) const
{
    xvector<xstring> retv;
    std::smatch matcher;
    if (std::regex_search(*this, matcher, rex))
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
    return this->Search(std::regex(pattern, mod), count);
}

xvector<xstring> xstring::Search(const char* pattern, rxm::type mod, int count) const {
    return this->Search(std::regex(pattern, mod), count);
}

// =========================================================================================================================

bool xstring::Has(const char var_char) const {
    if ((std::find(this->begin(), this->end(), var_char) != this->end()))
        return true;
    return false;
}

bool xstring::Lacks(const char var_char) const {

    if ((std::find(this->begin(), this->end(), var_char) != this->end()))
        return false;
    return true;
}

size_t xstring::Count(const char var_char) const {
    size_t n = std::count(this->begin(), this->end(), var_char);
    return n;
}

size_t xstring::Count(const xstring& pattern) const {
    size_t n = this->Split(pattern).size();
    return n;
}

// =========================================================================================================================

xstring xstring::Sub(const std::regex& rex, const std::string& replacement) const {
    return std::regex_replace(c_str(), rex, replacement);
}

xstring xstring::Sub(const RE2& rex, const std::string& replacement) const
{
    xstring ret = *this;
    RE2::GlobalReplace(reinterpret_cast<std::string*>(&ret), rex, replacement);
    return ret;
}

xstring xstring::Sub(const RE2& rex, const re2::StringPiece& replacement) const
{
    xstring ret = *this;
    RE2::GlobalReplace(reinterpret_cast<std::string*>(&ret), rex, replacement);
    return ret;
}

xstring xstring::Sub(const RE2& rex, const char* replacement) const
{
    xstring ret = *this;
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
    this->erase(0, this->find_first_not_of(" \t\r\n"));
    this->erase(this->find_last_not_of(" \t\r\n") + 1);
    return *this;
}

xstring& xstring::LeftTrim()
{
    this->erase(0, this->find_first_not_of(" \t\r\n"));
    return *this;
}

xstring& xstring::RightTrim()
{
    this->erase(this->find_last_not_of(" \t\r\n") + 1);
    return *this;
}

xstring& xstring::Trim(const xstring& trim)
{
    this->erase(0, this->find_first_not_of(trim));
    this->erase(this->find_last_not_of(trim) + 1);
    return *this;
}

xstring& xstring::LeftTrim(const xstring& trim)
{
    this->erase(0, this->find_first_not_of(trim));
    return *this;
}

xstring& xstring::RightTrim(const xstring& trim)
{
    this->erase(this->find_last_not_of(trim) + 1);
    return *this;
}

xstring xstring::operator()(long double x, long double y, long double z, const char removal_method) const
{
    size_t m_size = this->size();
    xstring n_arr;
    n_arr.reserve(m_size + 4);

    double n_arr_size = static_cast<double>(m_size) - 1;

    if (z >= 0) {

        if (x < 0) { x += n_arr_size; }

        if (!y) { y = n_arr_size; }
        else if (y < 0) { y += n_arr_size; }
        ++y;

        if (x > y) { return n_arr; }

        typename xstring::const_iterator iter = this->begin();
        typename xstring::const_iterator stop = this->begin() + static_cast<ull>(y);

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

        typename xstring::const_reverse_iterator iter = this->rend() - static_cast<ull>(x) - 1;
        typename xstring::const_reverse_iterator stop = this->rend() - static_cast<ull>(y);

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
    return std::atoi(this->c_str());
}

long xstring::ToLong() const
{
    return std::atol(this->c_str());
}

long long xstring::ToLongLong() const
{
    return std::atoll(this->c_str());
}

size_t xstring::To64() const
{
    return std::atoll(this->c_str());
}

double xstring::ToDouble() const
{
    return std::atof(this->c_str());
}

float xstring::ToFloat() const
{
    return static_cast<float>(std::atof(this->c_str()));
}

// =================================================================================================================================

xstring xstring::ToBlack() const {
    return Color::Black + *this;
}
xstring xstring::ToRed() const {
    return Color::Red + *this;
}
xstring xstring::ToGreen() const {
    return Color::Green + *this;
}
xstring xstring::ToYellow() const {
    return Color::Yellow + *this;
}
xstring xstring::ToBlue() const {
    return Color::Blue + *this;
}
xstring xstring::ToMegenta() const {
    return Color::Magenta + *this;
}
xstring xstring::ToCyan() const {
    return Color::Cyan + *this;
}
xstring xstring::ToGrey() const {
    return Color::Grey + *this;
}
xstring xstring::ToWhite() const {
    return Color::White + *this;
}
// --------------------------------------
xstring xstring::ToOnBlack() const {
    return Color::On::Black + *this;
}
xstring xstring::ToOnRed() const {
    return Color::On::Red + *this;
}
xstring xstring::ToOnGreen() const {
    return Color::On::Green + *this;
}
xstring xstring::ToOnYellow() const {
    return Color::On::Yellow + *this;
}
xstring xstring::ToOnBlue() const {
    return Color::On::Blue + *this;
}
xstring xstring::ToOnMegenta() const {
    return Color::On::Magenta + *this;
}
xstring xstring::ToOnCyan() const {
    return Color::On::Cyan + *this;
}
xstring xstring::ToOnGrey() const {
    return Color::On::Grey + *this;
}
xstring xstring::ToOnWhite() const {
    return Color::On::White + *this;
}
// --------------------------------------
xstring xstring::ResetColor() const {
    return *this + Color::Mod::Reset;
}
xstring xstring::ToBold() const {
    return Color::Mod::Bold + *this;
}
xstring xstring::ToUnderline() const {
    return Color::Mod::Underline + *this;
}
xstring xstring::ToInvertedColor() const {
    return Color::Mod::Reverse + *this;
}

// =================================================================================================================================

xstring WTXS(const wchar_t* wstr) {
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
