#pragma warning (disable : 26444) // allow anynomous objects
#pragma warning (disable : 26812) // allow normal enums from STL

#include "xstring.h"

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

xstring xstring::operator*(int total)
{
    xstring original = *this;
    for (int i = 0; i < total; i++) {
        this->insert(this->end(), original.begin(), original.end());
    }
    xstring ret_copy = *this;
    *this = original;
    return ret_copy;
}

xstring xstring::operator*=(int total)
{
    xstring str_copy = *this;
    for (int i = 0; i < total; i++) {
        this->insert(this->end(), str_copy.begin(), str_copy.end());
    };
    return *this;
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

xvector<xstring> xstring::split(const xstring& in_pattern, rxm::type mod) const
{
    xvector<xstring> split_content;
    std::regex rpattern(in_pattern, rxm::ECMAScript | mod);
    for (std::sregex_token_iterator iter(this->begin(), this->end(), rpattern, -1); iter != std::sregex_token_iterator(); ++iter)
        split_content << xstring(*iter);

    return split_content;
}

xvector<xstring> xstring::split(xstring&& in_pattern, rxm::type mod) const
{
    return this->split(in_pattern, mod);
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


xvector<xstring> xstring::inclusive_split(const xstring& splitter, rxm::type mod, bool aret) const
{
    xvector<xstring> retv;
    std::regex rpattern(splitter, rxm::ECMAScript | mod);
    for (std::sregex_token_iterator iter(this->begin(), this->end(), rpattern, { -1, 1 }); iter != std::sregex_token_iterator(); ++iter)
        retv << xstring(*iter);

    if (!aret) {
        if (retv.size() == 1)
            return xvector<xstring>();
    }
    return retv;
}


// =========================================================================================================================

bool xstring::match(const xstring& in_pattern, rxm::type mod) const
{
    std::regex pattern(in_pattern.c_str(), rxm::ECMAScript | mod);
    return bool(std::regex_match(*this, pattern));
}

bool xstring::match_line(const xstring& in_pattern, rxm::type mod) const
{
    std::vector<xstring> lines = this->split('\n');
    std::regex pattern(in_pattern.c_str(), rxm::ECMAScript | mod);
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (std::regex_match(*iter, pattern)) {
            return true;
        }
    }
    return false;
}

bool xstring::match_lines(const xstring& in_pattern, rxm::type mod) const
{
    std::vector<xstring> lines = this->split('\n');
    std::regex pattern(in_pattern, rxm::ECMAScript | mod);
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_match(*iter, pattern)) {
            return false;
        }
    }
    return true;
}

// =========================================================================================================================

bool xstring::scan(const char in_pattern, rxm::type mod) const
{
    return (std::find(this->begin(), this->end(), in_pattern) != this->end());
}

bool xstring::scan(const xstring& in_pattern, rxm::type mod) const
{
    return bool(std::regex_search(*this, std::regex(in_pattern, rxm::ECMAScript | mod)));
}

bool xstring::scan_line(const xstring& in_pattern, rxm::type mod) const
{
    std::vector<xstring> lines = this->split('\n');
    std::regex pattern(in_pattern, rxm::ECMAScript | mod);
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (std::regex_search(*iter, pattern)) {
            return true;
        }
    }
    return false;
}

bool xstring::scan_lines(const xstring& in_pattern, rxm::type mod) const
{
    std::vector<xstring> lines = this->split('\n');
    std::regex pattern(in_pattern, rxm::ECMAScript | mod);
    for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        if (!std::regex_search(*iter, pattern)) {
            return false;
        }
    }
    return true;
}

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

    for (xstring::const_iterator it = this->begin() + front_skip; it < this->end() - end_skip; it++) {
        if (!isascii(*it)) {
            threshold--;
            if(threshold < 1)
                return static_cast<int>(*it);
        }
    }
    return 0;
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

xvector<xstring> xstring::findall(const xstring& in_pattern, rxm::type mod, const bool group /*false*/) const
{
    xvector<xstring> retv;
    std::regex rpattern(in_pattern, rxm::ECMAScript | mod);

    for (std::sregex_token_iterator iter(this->begin(), this->end(), rpattern, 1); iter != std::sregex_token_iterator(); ++iter) 
        retv << xstring(*iter);

    return retv;
}

xvector<xstring> xstring::findwalk(const xstring& in_pattern, rxm::type mod, const bool group) const
{
    xvector<xstring> retv;
    std::regex rpattern(in_pattern, rxm::ECMAScript | mod);

    xvector<xstring> lines = this->split('\n');
    for (const auto& line : lines) {
        for (std::sregex_token_iterator iter(line.begin(), line.end(), rpattern, 1); iter != std::sregex_token_iterator(); ++iter)
            retv << xstring(*iter);
    }

    return retv;
}

xvector<xstring> xstring::search(const xstring& in_pattern, int depth, rxm::type mod, const bool group) const
{
    xvector<xstring> retv;
    std::regex rpattern(in_pattern, rxm::ECMAScript | mod);

    std::smatch matcher;
    if (std::regex_search(*this, matcher, rpattern)) {
        size_t sz = matcher.size();
        for (std::smatch::const_iterator it = matcher.begin()+1; it != matcher.end(); it++)
            retv << xstring(*it);
    }

    return retv;
}

// =========================================================================================================================

bool xstring::has(const char var_char, rxm::type mod) const {
    if ((std::find(this->begin(), this->end(), var_char) != this->end()))
        return true;
    return false;
}

bool xstring::lacks(const char var_char, rxm::type mod) const {

    if ((std::find(this->begin(), this->end(), var_char) != this->end()))
        return false;
    return true;
}

unsigned long long xstring::count(const char var_char, rxm::type mod) const {
    unsigned long long n = std::count(this->begin(), this->end(), var_char);
    return n;
}

unsigned long long xstring::count(const xstring& in_pattern, rxm::type mod) const {
    unsigned long long n = this->split(in_pattern).size();
    return n;
}

// =========================================================================================================================

xstring xstring::sub(const std::string& in_pattern, const std::string& replacement, rxm::type mod) const {
    return std::regex_replace(*this, std::regex(in_pattern, rxm::ECMAScript | mod), replacement);
}

xstring xstring::strip() {
    this->erase(this->begin(), std::find_if(this->begin(), this->end(),
        std::not1(std::ptr_fun<int, int>(std::isspace))));
    this->erase(std::find_if(this->rbegin(), this->rend(),
        std::not1(std::ptr_fun<int, int>(std::isspace))).base(), this->end());

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

