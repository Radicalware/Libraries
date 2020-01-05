
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
#include<vector>
#include<string>
#include<map>
#include<unordered_map>

#include "mc.h"
#include "ac.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;


int main() {

    cout << "========================================\n";
    cout << "testing map discovery\n\n";

    std::map<string, vector<string>> omap = { {"first", {"1-one","1-two","1-three"}},
                                                {"second",{"2-one","2-two","2-three"}}
    };

    std::unordered_map<std::string, std::string> umap1 = {    { "hash1", "AAAA"},
                                                            { "hash2", "BBBB"},
                                                            { "hash3", "CCCC"},
                                                            { "hash4", ""}
    };

    std::unordered_map<std::string, std::string> umap2 = {    { "AAAA", "1111"},
                                                            { "BBBB", "2222"},
                                                            { "CCCC", "3333"}
    };

    
    cout << ac::join(mc::all_keys(umap1), " ") << endl;
    // notice that the key hash4 is present becaused it list all of them
    // however hash4 really doesn't exist becaues it has no data so

    cout << "key_data for omap first = " << ac::join(mc::key_data("first", omap), " ") << endl;

    cout << "true  => " << mc::has_key("hash2", umap1) << endl;
    cout << "False = " << mc::has_key("hash4", umap1) << endl; // ret false sense hash4 contains nothing
    cout << "False = " << mc::has_key("no key", umap1) << endl;

    cout << "true  => " << mc::has_key_value("first","1-two",omap) << endl;
    cout << "False = " << mc::has_key_value("first", "2-two", omap) << endl;
    
    cout << ac::join(mc::all_keys<std::string, std::vector<std::string>>(omap), " <> ", true) << endl;
    cout << ac::join(mc::all_keys(omap), " <> ") << endl;

    cout << ac::join(mc::all_values(umap2), " <> ") << endl;

    cout << "true  => " << ac::scan_one("(con)", mc::all_keys(omap)) << endl;

    mc::xcopy(umap1, "C-EXT", umap2, "hash3");
    // umap1 gets the new key "C-EXT" and it's value is umap2.at("hash3")


    cout << ac::join(mc::all_keys(umap1), " ") << endl;

    cout << "C-EXT == " << umap1.at("C-EXT") << endl;

    cout << "========================================\n";

    return 0;
}
