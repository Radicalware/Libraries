#pragma warning (disable : 26444) // allow anynomous objects
#pragma warning (disable : 26812) // allow normal enums from STL

#include "xstring.h"

xstring Color::Black   = "\033[30m";
xstring Color::Red     = "\033[31m";
xstring Color::Green   = "\033[32m";
xstring Color::Yellow  = "\033[33m";
xstring Color::Blue    = "\033[34m";
xstring Color::Magenta = "\033[35m";
xstring Color::Cyan    = "\033[36m";
xstring Color::Grey    = "\033[37m";
xstring Color::White   = "\033[39m";

xstring Color::On::Black   = "\033[40m";
xstring Color::On::Red     = "\033[41m";
xstring Color::On::Green   = "\033[42m";
xstring Color::On::Yellow  = "\033[43m";
xstring Color::On::Blue    = "\033[44m";
xstring Color::On::Magenta = "\033[45m";
xstring Color::On::Cyan    = "\033[46m";
xstring Color::On::Grey    = "\033[47m";
xstring Color::On::White   = "\033[49m";

xstring Color::Mod::Reset     = "\033[00m";
xstring Color::Mod::Bold      = "\033[01m";
xstring Color::Mod::Underline = "\033[04m";
xstring Color::Mod::Reverse   = "\033[07m";

// Operates only on Linux 
xstring Color::Mod::Dark  = "\033[02m";
xstring Color::Mod::Blink = "\033[05m";
xstring Color::Mod::Hide  = "\033[08m";


xstring::xstring()
{
}


xstring::xstring(std::string&& str) noexcept : std::string(str)
{
}

xstring::xstring(const std::string& str): std::string(str)
{
    
}

xstring::xstring(xstring&& str) noexcept
{
    this->insert(this->begin(), str.begin(), str.end());
}

xstring::xstring(const xstring& str)
{
    this->insert(this->begin(),str.begin(), str.end());
}

xstring::xstring(const char* str): std::string(str)
{
}

xstring::xstring(const unsigned char* str): std::string((char*)str)
{
}

xstring::xstring(const char i_char, const int i_int) :
    std::string(i_int, i_char)
{
}

xstring::xstring(const int i_int, const char i_char) :
    std::string(i_int, i_char)
{
}

xstring::xstring(std::initializer_list<char> lst) : std::string(lst)
{
}


void xstring::operator=(const xstring& other)
{
    this->clear();
    if (other.size()) {
        this->reserve(other.size());
        this->insert(this->begin(), other.begin(), other.end());
    }
}

void xstring::operator=(xstring&& other) noexcept
{
    this->clear();
    if (other.size()) {
        this->reserve(other.size());
        this->insert(this->begin(), other.begin(), other.end());
    }
}

void xstring::operator=(const char* other)
{
    *this = std::string(other);
}

//void xstring::operator+=(const size_t& num)
//{
//    xstring val = to_xstring(num);
//    (*this).insert(this->end(), val.begin(), val.end());
//}

xstring xstring::operator+(const char* str) const
{
    xstring rstr = *this;
    rstr += str;
    return rstr;
}

void xstring::operator+=(const char* str)
{
    //this->insert(this->end(), &str, &str+str[strlen(str)]);
    *this += xstring(str);
}

void xstring::operator+=(const xstring& str)
{
    this->insert(this->end(), str.begin(), str.end());
}

bool xstring::operator==(const char* str) const
{
    if (strcmp(this->c_str(), str) == 0)
        return true;
    else
        return false;
}

void xstring::print() const
{
    std::cout << *this << '\n';
}

void xstring::print(int num) const
{
    std::cout << *this;
    char* new_lines = static_cast<char*>(calloc(static_cast<size_t>(num)+1, sizeof(char)));
    // calloc was used instead of "new" because "new" would give un-wanted after-effects.
    for (int i = 0; i < num; i++)
#pragma warning(suppress:6011) // we are derferencing a pointer, but assigning it a value at the same time
        new_lines[i] = '\n';
    std::cout << new_lines;
    free(new_lines);
}

void xstring::print(const xstring& front, const xstring& end) const
{
    std::cout << front << *this << end << '\n';
}

std::string xstring::to_string() const
{
    return std::string(this->c_str()); // todo: return only from the base class
}

xstring xstring::upper() const
{
    xstring ret_str;
    std::locale loc;
    for(xstring::const_iterator it = this->begin(); it != this->end(); it++)
        ret_str += std::toupper((*it), loc);
    return ret_str;
}

xstring xstring::lower() const
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

xstring xstring::remove(const char val) const
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

// ---------------------------------------------------------------

xvector<xstring> xstring::split(size_t loc) const
{
    xvector<xstring> ret_vec;
    ret_vec.reserve(4);
    ret_vec << this->substr(0, loc);
    ret_vec << this->substr(loc, this->size() - loc);

    return ret_vec;
}

xvector<xstring> xstring::split(const std::regex& rex) const
{
    xvector<xstring> split_content;
    for (std::sregex_token_iterator iter(this->begin(), this->end(), rex, -1); iter != std::sregex_token_iterator(); ++iter)
        split_content << xstring(*iter);

    return split_content;
}

xvector<xstring> xstring::split(const xstring& pattern, rxm::type mod) const
{
    std::regex rex(pattern, rxm::ECMAScript | mod);
    return this->split(rex);
}

xvector<xstring> xstring::split(const char splitter, rxm::type mod) const
{
    return this->split(xstring(splitter), mod);
}

xvector<xstring> xstring::inclusive_split(const char splitter, rxm::type mod, bool aret) const{
    return this->inclusive_split(xstring(splitter), mod, aret);
}

xvector<xstring> xstring::inclusive_split(const char* splitter, rxm::type mod, bool aret) const {
    return this->inclusive_split(xstring(splitter), mod, aret);
}

xvector<xstring> xstring::inclusive_split(const std::regex& rex, bool single) const
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


xvector<xstring> xstring::inclusive_split(const xstring& splitter, rxm::type mod, bool single) const
{
    std::regex rex(splitter, rxm::ECMAScript | mod);
    // -1       grab NOT regex
    //  0       grab regex
    //  1       grab blank
    return this->inclusive_split(rex, single);
}


// =========================================================================================================================

bool xstring::match(const std::regex& rex) const
{
    return bool(std::regex_match(*this, rex));
}

bool xstring::match(const xstring& pattern, rxm::type mod) const
{
    std::regex rex(pattern.c_str(), rxm::ECMAScript | mod);
    return bool(std::regex_match(*this, rex));
}

bool xstring::match_line(const std::regex& rex) const
{
    std::vector<xstring> lines = this->split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (std::regex_match(*iter, rex)) {
            return true;
        }
    }
    return false;
}

bool xstring::match_line(const xstring& pattern, rxm::type mod) const
{
    std::regex rex(pattern.c_str(), rxm::ECMAScript | mod);
    return this->match_line(rex);
}

bool xstring::match_lines(const std::regex& rex) const
{
    std::vector<xstring> lines = this->split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_match(*iter, rex)) {
            return false;
        }
    }
    return true;
}

bool xstring::match_lines(const xstring& pattern, rxm::type mod) const
{
    std::regex rex(pattern, rxm::ECMAScript | mod);
    return this->match_lines(rex);
}

// =========================================================================================================================

bool xstring::scan(const std::regex& rex) const
{
    return bool(std::regex_search(*this, rex));
}

bool xstring::scan(const char pattern, rxm::type mod) const
{
    return (std::find(this->begin(), this->end(), pattern) != this->end());
}

bool xstring::scan(const xstring& pattern, rxm::type mod) const
{
    return bool(std::regex_search(*this, std::regex(pattern, rxm::ECMAScript | mod)));
}

bool xstring::scan_line(const std::regex& rex) const
{
    std::vector<xstring> lines = this->split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) 
    {
        if (std::regex_search(*iter, rex)) 
            return true;
    }
    return false;
}

bool xstring::scan_line(const xstring& pattern, rxm::type mod) const
{
    std::regex rex(pattern, rxm::ECMAScript | mod);
    return this->scan_line(rex);
}

bool xstring::scan_lines(const std::regex& rex) const
{
    std::vector<xstring> lines = this->split('\n');
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_search(*iter, rex)) {
            return false;
        }
    }
    return true;
}

bool xstring::scan_lines(const xstring& pattern, rxm::type mod) const
{
    std::regex rex(pattern, rxm::ECMAScript | mod);
    return this->scan_lines(rex);
}
// =========================================================================================================================


bool xstring::scan_list(const xvector<std::regex>& rex_lst) const
{
    for (std::vector<std::regex>::const_iterator iter = rex_lst.begin(); iter != rex_lst.end(); iter++)
    {
        if (std::regex_search(*this, *iter)) {
            return true;
        }
    }
    return false;
}

bool xstring::scan_list(const xvector<xstring>& lst, rxm::type mod) const
{
    for (std::vector<xstring>::const_iterator iter = lst.begin(); iter != lst.end(); iter++)
    {
        std::regex rex(*iter, rxm::ECMAScript | mod);
        if (std::regex_search(*this, rex)) {
            return true;
        }
    }
    return false;
}


bool xstring::scan_list(const xvector<std::regex*>& rex_lst) const
{
    for (std::vector<std::regex*>::const_iterator iter = rex_lst.begin(); iter != rex_lst.end(); iter++)
    {
        if (std::regex_search(*this, **iter)) {
            return true;
        }
    }
    return false;
}

bool xstring::scan_list(const xvector<xstring*>& lst, rxm::type mod) const
{
    for (std::vector<xstring*>::const_iterator iter = lst.begin(); iter != lst.end(); iter++)
    {
        std::regex rex(**iter, rxm::ECMAScript | mod);
        if (std::regex_search(*this, rex))
            return true;
    }
    return false;
}

// =========================================================================================================================

bool xstring::is(const xstring& other) const
{
    return (*this) == other;
}

size_t xstring::hash() const
{
    return std::hash<std::string>()(*this);
}

// =========================================================================================================================

int xstring::has_non_ascii(int front_skip , int end_skip, int threshold) const
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

bool xstring::has_nulls() const
{
    for (xstring::const_iterator it = this->begin(); it < this->end(); it++) {
        if (*it == '\0')
            return true;
    }
    return false;
}

bool xstring::has_dual_nulls() const
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

xstring xstring::remove_non_ascii() const 
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

xvector<xstring> xstring::findall(const std::regex& rex) const
{
    xvector<xstring> retv;
    for (std::sregex_token_iterator iter(this->begin(), this->end(), rex, 1); iter != std::sregex_token_iterator(); ++iter)
        retv << xstring(*iter);
    return retv;
}

xvector<xstring> xstring::findall(const xstring& pattern, rxm::type mod) const
{
    std::regex rex(pattern, rxm::ECMAScript | mod);
    return this->findall(rex);
}

xvector<xstring> xstring::findwalk(const std::regex& rex) const
{
    xvector<xstring> retv;
    xvector<xstring> lines = this->split('\n');
    for (const auto& line : lines) {
        for (std::sregex_token_iterator iter(line.begin(), line.end(), rex, 1); iter != std::sregex_token_iterator(); ++iter)
            retv << xstring(*iter);
    }
    return retv;
}

xvector<xstring> xstring::findwalk(const xstring& pattern, rxm::type mod) const
{
    std::regex rex(pattern, rxm::ECMAScript | mod);
    return this->findwalk(rex);
}

xvector<xstring> xstring::search(const std::regex& rex) const
{
    xvector<xstring> retv;
    std::smatch matcher;
    if (std::regex_search(*this, matcher, rex)) {
        size_t sz = matcher.size();
        for (std::smatch::const_iterator it = matcher.begin() + 1; it != matcher.end(); it++)
            retv << xstring(*it);
    }
    return retv;
}

xvector<xstring> xstring::search(const xstring& pattern, int depth, rxm::type mod) const
{
    std::regex rex(pattern, rxm::ECMAScript | mod);
    return this->search(rex);
}

// =========================================================================================================================

bool xstring::has(const char var_char) const {
    if ((std::find(this->begin(), this->end(), var_char) != this->end()))
        return true;
    return false;
}

bool xstring::lacks(const char var_char) const {

    if ((std::find(this->begin(), this->end(), var_char) != this->end()))
        return false;
    return true;
}

size_t xstring::count(const char var_char) const {
    size_t n = std::count(this->begin(), this->end(), var_char);
    return n;
}

size_t xstring::count(const xstring& pattern) const {
    size_t n = this->split(pattern).size();
    return n;
}

// =========================================================================================================================

xstring xstring::sub(const std::regex& rex, const std::string& replacement) const
{
    return std::regex_replace(*this, rex, replacement);
}

xstring xstring::sub(const std::string& pattern, const std::string& replacement, rxm::type mod) const {
    return std::regex_replace(*this, std::regex(pattern, rxm::ECMAScript | mod), replacement);
}

xstring& xstring::trim()
{
    this->erase(0, this->find_first_not_of(" \t\r\n"));
    this->erase(this->find_last_not_of(" \t\r\n") + 1);
    return *this;
}

xstring& xstring::ltrim()
{
    this->erase(0, this->find_first_not_of(" \t\r\n"));
    return *this;
}

xstring& xstring::rtrim()
{
    this->erase(this->find_last_not_of(" \t\r\n") + 1);
    return *this;
}

xstring& xstring::trim(const xstring& trim)
{
    this->erase(0, this->find_first_not_of(trim));
    this->erase(this->find_last_not_of(trim) + 1);
    return *this;
}

xstring& xstring::ltrim(const xstring& trim)
{
    this->erase(0, this->find_first_not_of(trim));
    return *this;
}

xstring& xstring::rtrim(const xstring& trim)
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

int xstring::to_int() const
{
    return std::atoi(this->c_str());
}

long xstring::to_long() const
{
    return std::atol(this->c_str());
}

long long xstring::to_ll() const
{
    return std::atoll(this->c_str());
}

size_t xstring::to_64() const
{
    return std::atoll(this->c_str());
}

double xstring::to_double() const
{
    return std::atof(this->c_str());
}

float xstring::to_float() const
{
    return static_cast<float>(std::atof(this->c_str()));
}

// =================================================================================================================================

xstring xstring::black() const {
    return Color::Black + *this;
}
xstring xstring::red() const {
    return Color::Red + *this;
}
xstring xstring::green() const {
    return Color::Green + *this;
}
xstring xstring::yellow() const {
    return Color::Yellow + *this;
}
xstring xstring::blue() const {
    return Color::Blue + *this;
}
xstring xstring::megenta() const {
    return Color::Magenta + *this;
}
xstring xstring::cyan() const {
    return Color::Cyan + *this;
}
xstring xstring::grey() const {
    return Color::Grey + *this;
}
xstring xstring::white() const {
    return Color::White + *this;
}
// --------------------------------------
xstring xstring::on_black() const {
    return Color::On::Black + *this;
}
xstring xstring::on_red() const {
    return Color::On::Red + *this;
}
xstring xstring::on_green() const {
    return Color::On::Green + *this;
}
xstring xstring::on_yellow() const {
    return Color::On::Yellow + *this;
}
xstring xstring::on_blue() const {
    return Color::On::Blue + *this;
}
xstring xstring::on_megenta() const {
    return Color::On::Magenta + *this;
}
xstring xstring::on_cyan() const {
    return Color::On::Cyan + *this;
}
xstring xstring::on_grey() const {
    return Color::On::Grey + *this;
}
xstring xstring::on_white() const {
    return Color::On::White + *this;
}
// --------------------------------------
xstring xstring::reset() const{
    return *this + Color::Mod::Reset;
}
xstring xstring::bold() const {
    return Color::Mod::Bold + *this;
}
xstring xstring::underline() const {
    return Color::Mod::Underline + *this;
}
xstring xstring::reverse() const {
    return Color::Mod::Reverse + *this;
}
// =================================================================================================================================
