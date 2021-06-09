
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
#include<vector>
#include<string>
#include<map>
#include<unordered_map>

#include "MC.h"
#include "AC.h"

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

    
    cout << AC::Join(MC::GetAllKeys(umap1), " ") << endl;
    // notice that the key hash4 is present becaused it list all of them
    // however hash4 really doesn't exist becaues it has no data so

    cout << "key_data for omap first = " << AC::Join(MC::GetKeyData("first", omap), " ") << endl;

    cout << "true  => " << MC::HasKey("hash2", umap1) << endl;
    cout << "False = " << MC::HasKey("hash4", umap1) << endl; // ret false sense hash4 contains nothing
    cout << "False = " << MC::HasKey("no key", umap1) << endl;

    cout << "true  => " << MC::HasKeyValue("first","1-two",omap) << endl;
    cout << "False = " << MC::HasKeyValue("first", "2-two", omap) << endl;
    
    cout << AC::Join(MC::GetAllKeys<std::string, std::vector<std::string>>(omap), " <> ", true) << endl;
    cout << AC::Join(MC::GetAllKeys(omap), " <> ") << endl;

    cout << AC::Join(MC::GetAllValues(umap2), " <> ") << endl;

    cout << "true  => " << AC::ScanOne("(con)", MC::GetAllKeys(omap)) << endl;

    MC::XCopy(umap1, "C-EXT", umap2, "hash3");
    // umap1 gets the new key "C-EXT" and it's value is umap2.At("hash3")


    cout << AC::Join(MC::GetAllKeys(umap1), " ") << endl;

    cout << "C-EXT == " << umap1["C-EXT"] << endl;

    cout << "========================================\n";

    return 0;
}
