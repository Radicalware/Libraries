
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
#include<vector>
#include<string>

#include "AC.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;


int main() {
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

    cout << AC::Join(vec_str, " ") << endl;
    cout << AC::Join(vec_char, " ") << endl;
    cout << AC::Join(vec_chara, " ") << endl;
    cout << AC::Join(vec_int, " ") << endl;

    std::string three = "three";
    cout << "true  => " << AC::Has(three, vec_str) << endl;
    cout << "true  => " << AC::Has("two", vec_str) << endl;
    cout << "false = " << AC::Has("twenty", vec_str) << endl;

    cout << "true  => " << AC::Has(5, vec_int) << endl;
    cout << "false = " << AC::Has(7, vec_int) << endl;

    // ----------------------------------------------------------------
    cout << "========================================\n";
    cout << "testing regex\n\n";
    vector<string> emails{
        "first.last@subdomain.domain.com",
        "realuser@gmail.com",
        "trap@attack.hack.info"
    };

    cout << "false email = " << AC::MatchOne("[\\w\\d_]*@[\\w\\d_]*", emails) << endl;
    // that failed because it does not match any one email start to finish
    cout << "true  email = " << AC::MatchOne("^([\\w\\d_]*(\\.?)){1,2}@([\\w\\d_]*\\.){1,2}[\\w\\d_]*$", emails) << endl;;
    // this one matches the first email start to finish so it replies true


    cout << "true email  = " << AC::ScanOne("[\\w\\d_]*@[\\w\\d_]*", emails) << endl;
    // this time the partial email returns true because the regex matches at least a portion
    // of one of the elements in the array.

    std::string found_email = AC::GetScans("^([\\w\\d_]*(\\.?)){1,2}@([\\w\\d_]*\\.){1,2}[\\w\\d_]*$", emails)[0];
    cout << found_email << endl;
    emails[0] = "new.appended@email.com";
    cout << found_email << endl; // passing of data gives you a deep copy from the *iterator

    cout << AC::GetScans("[\\w\\d_]*@[\\w\\d_]*", emails)[2] << endl;

    cout << "========================================\n";
    cout << "generators\n";

    cout << AC::Join(AC::Range(0, 10), " - ") << endl;

    cout << "12345" << AC::Ditto("0", 5) << endl;


    return 0;
}
