#pragma once

#include<iostream>
#include<vector>
#include<string>
#include<map>
#include<unordered_map>

#include "ord.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;


int ex_order() {
	cout << "========================================\n";
	cout << "testing vector discovery\n\n";
	vector<string> vec_str{ "zero","one","two","three","four","five" };
	vector<char>   vec_char{ 'v','a','r','c','h','a','r' };

	char v1[] = "var1";
	char v2[] = "var2";
	char v3[] = "var3";
	char v4[] = "var4";
	vector<char*> vec_chara{ v1, v2, v3, v4 };

	vector<int>   vec_int{ 0,1,2,3,4,5,6 };

	cout << ord::join(vec_str) << endl;
	cout << ord::join(vec_char) << endl;
	cout << ord::join(vec_chara) << endl;
	cout << ord::join(vec_int) << endl;

	std::string three = "three";
	cout << "true  => " << ord::in(three, vec_str) << endl;
	cout << "true  => " << ord::in("two", vec_str) << endl;
	cout << "false = " << ord::in("twenty", vec_str) << endl;

	cout << "true  => " << ord::in(5, vec_int) << endl;
	cout << "false = " << ord::in(7, vec_int) << endl;

	// ----------------------------------------------------------------
	cout << "========================================\n";
	cout << "testing regex\n\n";
	vector<string> emails{
		"first.last@subdomain.domain.com",
		"realuser@gmail.com",
		"trap@attack.hack.info"
	};

	cout << "false email = " << ord::match_one("[\\w\\d_]*@[\\w\\d_]*", emails) << endl;
	// that failed because it does not match any one email start to finish
	cout << "true  email = " << ord::match_one("^([\\w\\d_]*(\\.?)){1,2}@([\\w\\d_]*\\.){1,2}[\\w\\d_]*$", emails) << endl;;
	// this one matches the first email start to finish so it replies true


	cout << "true email  = " << ord::scan_one("[\\w\\d_]*@[\\w\\d_]*", emails) << endl;
	// this time the partial email returns true because the regex matches at least a portion
	// of one of the elements in the array.

	std::string found_email = ord::ret_scans("^([\\w\\d_]*(\\.?)){1,2}@([\\w\\d_]*\\.){1,2}[\\w\\d_]*$", emails)[0];
	cout << found_email << endl;
	emails[0] = "new.appended@email.com";
	cout << found_email << endl; // passing of data gives you a deep copy from the *iterator

	cout << ord::ret_scans("[\\w\\d_]*@[\\w\\d_]*", emails)[2] << endl;


	cout << "========================================\n";
	cout << "testing map discovery\n\n";

	std::map<string, vector<string>> omap = { {"first", {"1-one","1-two","1-three"}},
												{"second",{"2-one","2-two","2-three"}}
	};

	std::unordered_map<std::string, std::string> umap1 = { { "hash1", "AAAA"},
														{ "hash2", "BBBB"},
														{ "hash3", "CCCC"},
														{ "hash4", ""}
	};

	std::unordered_map<std::string, std::string> umap2 = { { "AAAA", "1111"},
														 { "BBBB", "2222"},
														 { "CCCC", "3333"}
	};


	cout << ord::join(ord::keys(umap1)) << endl;
	// notice that the key hash4 is present becaused it list all of them
	// however hash4 really doesn't exist becaues it has no data so

	cout << "true  => " << ord::has_key(umap1, "hash2") << endl;
	cout << "False = " << ord::has_key(umap1, "hash4") << endl; // ret false sense hash4 contains nothing
	cout << "False = " << ord::has_key(umap1, "no key") << endl;

	cout << "true  => " << ord::has(omap.at("first"), "1-two") << endl;
	cout << "False = " << ord::has(omap.at("first"), "2-two") << endl;



	cout << ord::join(ord::keys<std::string, std::vector<std::string>>(omap), " <> ") << endl;
	cout << ord::join(ord::keys(omap), " <> ") << endl;

	cout << ord::join(ord::key_values<std::string, std::string>(umap1), " <> ", true) << endl;

	cout << "true  => " << ord::scan_one("(con)", ord::keys(omap)) << endl;

	ord::relational_copy(umap1, "C-EXT", umap2, "hash3");
	// umap1 gets the new key "C-EXT" and it's value is umap2.at("hash3")


	cout << ord::join(ord::keys(umap1)) << endl;

	cout << "C-EXT == " << umap1.at("C-EXT") << endl;

	cout << "========================================\n";
	cout << "generators\n";

	cout << ord::join(ord::range(0, 10), " - ") << endl;

	cout << "12345" << ord::ditto("0", 5) << endl;


	return 0;
}
