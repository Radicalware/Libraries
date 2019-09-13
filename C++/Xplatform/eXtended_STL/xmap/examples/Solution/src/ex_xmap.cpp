﻿#include <iostream>


#include "xvector.h"
#include "xstring.h"
#include "xmap.h"

#include <utility>

using std::cout;
using std::endl;
using std::string;

using wrap = std::pair<xstring, xstring>;


int main()
{

	xvector<xstring> single_vec{ "one","two","three","four","five","six" };
	xvector<xvector<xstring>> double_vec = single_vec.split(3);

	single_vec.sub_all("(o)", "0").join(' ').print();

	single_vec.remove("(^f)").join(' ').print();
	single_vec.take("(^f)").join(' ').print();

	cout << "\n========================================\n";
	cout << "testing map discovery\n\n";

	xmap<xstring, xvector<xstring>> map_s_v = { 
		{"V-First", {"1-one","1-two","1-three"}},
		{"V-Second",{"2-one","2-two","2-three"}}
	};

	xmap<xstring, xstring> smap = {
		{ "hash1111", "AAAA"},
		{ "hash222", "BBBB"},
		{ "hash33", "CCCC"},
		{ "hash4", ""}
	};


	map_s_v.keys().join(' ').print();
	smap.keys().join(' ').print();

	// the following three are the same method but with different verbage
	cout << "1.) map_ss_1 value for 'hash222' = " << smap.at("hash222") << endl;
	cout << "2.) map_ss_1 value for 'hash222' = " << smap.key("hash222") << endl;
	cout << "3.) map_ss_1 value for 'hash222' = " << smap.value_for("hash222") << endl;

	cout << "its value is BBBB = " << smap.value_for("hash222").is("BBBB") << endl; // true
	cout << "its value is BBBB = " << smap.key("hash222").is("BBBB") << endl; // true
	cout << "its value is BBBB = " << smap("hash222", "BBBB") << endl; // true, alternate method
	cout << "its value is BBBB = " << smap("I don't exist", "BBBB") << endl; // false
	
	cout << "no cache allocated; size = " << smap.cached_keys().size() << endl;

	cout << "========================================\n";
	cout << "starting order of keys \n" << smap.keys().join("\n") << endl;
	cout << "========================================\n";
	cout << "orderd by key size \n";
	smap.allocate_keys()->sort([](const auto* first, const auto* second) { return first->size() < second->size(); });
	smap.cached_keys().join('\n').print();
	cout << "========================================\n";
	
	xstring key_values;
	smap.proc(key_values, [](auto& key_values, const auto& iter) {
		key_values += iter.second + " ";
	});
	key_values.print();
	cout << "========================================\n";

	xvector<xstring> str_lst = { "one","two","three","four","five" };
	xvector<xstring*> ptr_lst = str_lst.ptrs();

	xmap<xstring*, xstring> map_ptrs;
	int count = 1;
	for (xstring* ptr : ptr_lst) {
		map_ptrs.add_pair(ptr, xstring(count, '='));
		count++;
	}

	map_ptrs.allocate_keys()->join('\n').print(2);

	cout << "cached truth: " << map_ptrs.cache().has(str_lst[0]) << endl; // faster (if cached) than
	cout << "slow   truth: " << map_ptrs.has(str_lst[0]) << endl;

	cout << "========================================\n";

	smap.print(2);
	smap.allocate_reverse_map()->print();

	return 0;
}

