#pragma warning (disable : 26444) // allow anynomous objects

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

xvector<xstring> xstring::split(const xstring& in_pattern, rxm mod) const
{
	xvector<xstring> split_content;
	std::regex pattern(in_pattern, static_cast<rexmod>(mod));
	std::sregex_token_iterator iter(this->begin(), this->end(), pattern, -1);

	for ( ; iter != std::sregex_token_iterator(); ++iter)
		split_content << xstring(*iter);

	return split_content;
}

xvector<xstring> xstring::split(xstring&& in_pattern, rxm mod) const
{
	return this->split(in_pattern);
}

xvector<xstring> xstring::split(const char splitter, rxm mod) const
{
	xvector<xstring> all_sections;
	xstring current_section;
	for (xstring::const_iterator it = this->begin(); it != this->end(); it++) {
		if (*it == splitter) {
			if (current_section.size()) {
				all_sections.push_back(current_section);
				current_section.clear();
			}
		}
		else {
			current_section += *it;
		}
	}
	if (current_section.size())
		all_sections.push_back(current_section);
	return all_sections;
}


// =========================================================================================================================

bool xstring::match(const xstring& in_pattern, rxm mod) const
{
	std::regex pattern(in_pattern.c_str(), static_cast<rexmod>(mod));
	return bool(std::regex_match(*this, pattern));
}

bool xstring::match_line(const xstring& in_pattern, rxm mod) const
{
	std::vector<xstring> lines = this->split('\n');
	std::regex pattern(in_pattern.c_str(), static_cast<rexmod>(mod));
	for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
		if (std::regex_match(*iter, pattern)) {
			return true;
		}
	}
	return false;
}

bool xstring::match_lines(const xstring& in_pattern, rxm mod) const
{
	std::vector<xstring> lines = this->split('\n');
	std::regex pattern(in_pattern, static_cast<rexmod>(mod));
	for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
		if (!std::regex_match(*iter, pattern)) {
			return false;
		}
	}
	return true;
}

// =========================================================================================================================

bool xstring::scan(const char in_pattern, rxm mod) const
{
	return (std::find(this->begin(), this->end(), in_pattern) != this->end());
}

bool xstring::scan(const xstring& in_pattern, rxm mod) const
{
	return bool(std::regex_search(*this, std::regex(in_pattern)));
}

bool xstring::scan_line(const xstring& in_pattern, rxm mod) const
{
	std::vector<xstring> lines = this->split('\n');
	std::regex pattern(in_pattern, static_cast<rexmod>(mod));
	for (std::vector<xstring>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
		if (std::regex_search(*iter, pattern)) {
			return true;
		}
	}
	return false;
}

bool xstring::scan_lines(const xstring& in_pattern, rxm mod) const
{
	std::vector<xstring> lines = this->split('\n');
	std::regex pattern(in_pattern, static_cast<rexmod>(mod));
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

bool xstring::has_non_ascii() const {
	for (char c : (*this)) {
		if (!((static_cast<unsigned char>(c) > 0) || (static_cast<unsigned char>(c) < 128))) {
			return true;
		}
	}
	return false;
}

xstring xstring::remove_non_ascii() const {
	xstring clean_data;
	clean_data.reserve(this->size());
	for (xstring::const_iterator it = this->begin(); it < this->end(); it++) {
		if (int(*it) > 0 && int(*it) < 128)
			clean_data += *it;
	}
	return clean_data;
}

// =========================================================================================================================

xvector<xstring> xstring::grouper(const xstring& content, xvector<xstring>& ret_vector, const std::regex& pattern)  const {
	std::smatch match_array;
	xstring::const_iterator searchStart(content.cbegin());
	xstring::const_iterator prev(content.cbegin());
	while (regex_search(searchStart, content.cend(), match_array, pattern)) {
		for (int i = 0; i < match_array.size(); i++) {
			ret_vector.push_back(xstring(match_array[i]));
		}

		searchStart += match_array.position() + match_array.length();
		if (searchStart == prev) {
			break;
		}
		else { prev = searchStart; }
	}
	return ret_vector;
}


xvector<xstring> xstring::iterate(const xstring& content, xvector<xstring>& ret_vector, const std::regex& pattern) const {
	//std::smatch match_array;
	int start_iter = 1;
	if (bool(std::regex_search(R"(^\(\?\:)", pattern)) == true)
		start_iter = 2;
	
	std::sregex_iterator iter_index = std::sregex_iterator(content.begin(), content.end(), pattern);
	for (iter_index; iter_index != std::sregex_iterator(); ++iter_index) {
		std::match_results<xstring::const_iterator> match_array(*iter_index);
		//match_array = *iter_index;
		//std::sregex_iterator match_array = iter_index;
		for (int index = start_iter; index < match_array.size(); ++index) {
			if (!match_array[index].str().empty()) {
				ret_vector.push_back(xstring(match_array[index]));
			}
		}
	}
	return ret_vector;
}


std::vector<xstring> xstring::findall(const std::string& in_pattern, rxm mod, const bool group /*false*/) const
{
	xvector<xstring> ret_vector;
	xvector<xstring> split_string;

	ull new_line_count = std::count(this->begin(), this->end(), '\n');
	size_t split_loc = 0;
	xstring tmp_content = *this;
	// if/else: set each line to an element of the split_string vector
	if (new_line_count) {

		for (ull i = 0; i < new_line_count; i++) {
			split_loc = tmp_content.find('\n');
			split_string.push_back(tmp_content.substr(0, split_loc));
			tmp_content = tmp_content.substr(split_loc + 1, tmp_content.length() - split_loc - 1);
		}
	}
	else {
		new_line_count = 1;
		split_string.push_back(*this);
	}

	std::smatch match_array;
	const std::regex pattern(in_pattern, static_cast<rexmod>(mod));
	// now iterate through each line (now each element of the array)
	if (group == false) { // grouping is set to false by default
		for (ull index = 0; index < new_line_count; index++) {
			this->iterate(split_string[index], ret_vector, pattern);
		}
	}
	else { // If you chose grouping, you have more controle but more work. (C++ not Python style)
		for (ull i = 0; i < new_line_count; i++) {
			this->grouper(split_string[i], ret_vector, pattern);
		}
	}

	xvector<xstring> filtered;
	for (ull i = 0; i < ret_vector.size(); i++) {
		if (!(ret_vector[i].size() == 1 && ret_vector[i].c_str()[0] == ' '))
			filtered.push_back(ret_vector[i]);
	}

	return filtered;
}

// =========================================================================================================================

bool xstring::has(const char var_char, rxm mod) const {
	if ((std::find(this->begin(), this->end(), var_char) != this->end()))
		return true;
	return false;
}

bool xstring::lacks(const char var_char, rxm mod) const {

	if ((std::find(this->begin(), this->end(), var_char) != this->end()))
		return false;
	return true;
}

unsigned long long xstring::count(const char var_char, rxm mod) const {
	unsigned long long n = std::count(this->begin(), this->end(), var_char);
	return n;
}

unsigned long long xstring::count(const xstring& in_pattern, rxm mod) const {
	unsigned long long n = this->split(in_pattern).size();
	return n;
}

// =========================================================================================================================

xstring xstring::sub(const std::string& in_pattern, const std::string& replacement, rxm mod) const {
	return std::regex_replace(*this, std::regex(in_pattern), replacement);
}

xstring xstring::strip() {
	this->erase(this->begin(), std::find_if(this->begin(), this->end(),
		std::not1(std::ptr_fun<int, int>(std::isspace))));
	this->erase(std::find_if(this->rbegin(), this->rend(),
		std::not1(std::ptr_fun<int, int>(std::isspace))).base(), this->end());

	return *this;
}

xstring xstring::operator()(double x, double y, double z, const char removal_method) const
{
	size_t m_size = this->size();
	xstring n_arr;
	n_arr.reserve(m_size + 4);

	double n_arr_size = static_cast<double>(m_size - 1);

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

// =========================================================================================================================

#if defined(__unix__)
template <class _Kty>
_NODISCARD size_t std::hash_xstring_vals::_Hash_array_representation(
    const _Kty* const _First, const size_t _Count) const noexcept {
    // bitwise hashes the representation of an array
    static_assert(is_trivial_v<_Kty>, "Only trivial types can be directly hashed.");
    return _Fnv1a_append_bytes(
        _FNV_offset_basis, reinterpret_cast<const unsigned char*>(_First), _Count * sizeof(_Kty)
    );
}

_NODISCARD inline size_t std::hash_xstring_vals::_Fnv1a_append_bytes(
    size_t _Val, const unsigned char* const _First, const size_t _Count) const noexcept {
   // accumulate range [_First, _First + _Count) into partial FNV-1a hash _Val
    for (size_t _Idx = 0; _Idx < _Count; ++_Idx) {
        _Val ^= static_cast<size_t>(_First[_Idx]);
        _Val *= _FNV_prime;
    }
    return _Val;
}
#endif
// STRUCT TEMPLATE SPECIALIZATION hash
size_t std::hash<xstring>::operator()(const xstring& _Keyval) const noexcept {
    return _Hash_array_representation(_Keyval.c_str(), _Keyval.size());
};

size_t std::hash<xstring*>::operator()(const xstring* _Keyval) const noexcept {
    return _Hash_array_representation(_Keyval->c_str(), _Keyval->size());
};

size_t std::hash<const xstring*>::operator()(const xstring* _Keyval) const noexcept {
    return _Hash_array_representation(_Keyval->c_str(), _Keyval->size());
};
