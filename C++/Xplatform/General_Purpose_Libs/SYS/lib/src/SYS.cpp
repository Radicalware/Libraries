
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

#include "../include/SYS.h"


// logical lok at how the KVP system works under 2 keys going to the same values
// char key map ---| -- int map --- | --- values
// str  key map ---|

SYS::SYS(int c_argc, char** c_argv) :
	m_argc(c_argc), m_argv(c_argv) {
	(this)->set_args(m_argc, m_argv);
};


SYS::SYS() {};

SYS::~SYS() {};

void SYS::add_value_section(size_t idx_start, size_t idx_end)
{
	size_t size = m_values.size();
	m_values.resize(size + 1);

	for (size_t i = idx_start; i < idx_end; i++) 
		m_values[size] << &m_all_args[i];
	
}

// ======================================================================================================================
// >>>> args

SYS SYS::set_args(int argc, char** argv) {

	m_full_path = xstring(argv[0]);
	size_t program_split_loc = m_full_path.find_last_of("/\\");
	m_path = m_full_path.substr(0, program_split_loc + 1);
	m_file = m_full_path.substr(program_split_loc + 1, m_full_path.size() - 1);

	m_argc = argc;

	m_alias.allocate_reverse_map(); // rev goes str>char
	m_all_args.add_char_strings(argc, argv);

	size_t last_args_idx = 0;
	int vec_values_idx = 0;

	// program arg (stage 0)
	short int stage = 0;
	m_str_kvps.add_pair(&m_file, vec_values_idx);

	size_t i = 1;
	for (; i < argc; i++)
	{
		if (m_all_args[i][0] == '-' && m_all_args[i][1] == '-') // str key
		{
			m_key_used = true;
			vec_values_idx++;
			this->add_value_section(last_args_idx, i); 
			m_str_kvps.add_pair(&m_all_args[i], vec_values_idx);

			if (m_alias.cached_rev_map().has(m_all_args[i]))  // if we have a char for the string add it
				m_chr_kvps.add_pair(m_alias.cached_rev_map()[m_all_args[i]], vec_values_idx);
			
			last_args_idx = i + 1; // +1 to avoid key
			stage = 1;
		}
		else if(m_all_args[i][0] == '-') // char key
		{
			m_key_used = true;
			vec_values_idx++;
			this->add_value_section(last_args_idx, i);
			for (size_t char_idx = 1; char_idx < m_all_args[i].size(); char_idx++) { // starts at 1 to avoid the '-'
				m_chr_kvps.add_pair(m_all_args[i][char_idx], vec_values_idx);
				if (m_alias.has(m_all_args[i][char_idx])) { // if the alias has a str for the char key, add the value
					m_str_kvps.add_pair(&m_alias.find(m_all_args[i][char_idx])->second, vec_values_idx); 
				}
			}
			last_args_idx = i + 1; // +1 to avoid key 
			stage = 2;
		}
	}
	this->add_value_section(last_args_idx, i);

	m_str_kvps.allocate_keys();
	m_chr_kvps.keys().proc([this](char chr) { m_chr_lst += chr; });

	// for debugging
	//m_str_kvps.print();
	//int i = 0;
	//m_values.proc([&i](auto& vec) { std::cout << i << " >> " << vec.join(' ') << '\n'; i++; });

	return *this;
}


void SYS::alias(const char c_arg, const xstring& s_arg) {
	m_alias.add_pair(c_arg, s_arg);
}


// -------------------------------------------------------------------------------------------------------------------
int SYS::argc() { return m_argc; }
xvector<xstring> SYS::argv() const { return m_all_args; }
xvector<const xstring*> SYS::str_keys() const { return m_str_kvps.cache(); }
xstring SYS::chr_keys() const { return m_chr_lst; }
// -------------------------------------------------------------------------------------------------------------------
xstring SYS::full_path() { return m_full_path; }
xstring SYS::path() { return m_path; }
xstring SYS::file() { return m_file; }
// -------------------------------------------------------------------------------------------------------------------
xvector<xstring*> SYS::key(const xstring& key) { return m_values[m_str_kvps[key]]; }
xvector<xstring*> SYS::key(const char key) { return m_values[m_chr_kvps[key]]; }
bool SYS::key_used() const { return m_key_used; }
// -------------------------------------------------------------------------------------------------------------------

bool SYS::has(const xstring& key) {
	return this->m_str_kvps.cache().has(key);
};

bool SYS::has(const xstring* key) {
	return this->m_str_kvps.cache().has(*key);
}

bool SYS::has(xstring&& key)
{
	return this->m_str_kvps.cache().has(key);
}

bool SYS::has(const char* key)
{
	return this->m_str_kvps.cache().has(xstring(key));
}

bool SYS::has(const char key)
{
	return this->m_chr_kvps.cache().has(key);
}

// -------------------------------------------------------------------------------------------------------------------

xvector<xstring*> SYS::operator[](const xstring& key) { return m_values[m_str_kvps[key]]; }

xvector<xstring*> SYS::operator[](const char key) { return m_values[m_chr_kvps[key]]; }

xstring SYS::operator[](int key) {
	return m_all_args[key];
}

bool SYS::operator()(const xstring& key)
{
	return m_str_kvps.has(key);
}

bool SYS::operator()(const xstring& key, const xstring& value) {
	return m_values[m_str_kvps[key]].has(value);
}

bool SYS::operator()(xstring&& key)
{
	return m_str_kvps.has(key);
}

bool SYS::operator()(xstring&& key, xstring&& value)
{
	return m_values[m_str_kvps[key]].has(value);
}

bool SYS::operator()(const char key)
{
	return this->m_chr_lst.has(key);
}

bool SYS::operator()(const char key, const xstring& value) {
	return m_values[m_chr_kvps[key]].has(value);

}

// -------------------------------------------------------------------------------------------------------------------
bool SYS::help() {
	return m_str_kvps.keys().match_one(R"(^[-]{1,2}[hH]((elp)?)$)");
}
// ======================================================================================================================

