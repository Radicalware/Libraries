
#include<iostream>
#include "xvector.h"
#include "xstring.h"

using std::cout;
using std::endl;

int main(int argc, char** argv) {


    xvector<xstring> vec_str{ "zero","one","two","three","four","five" };

    vec_str.add("six", "seven", "eight"); // as many as you want.

    vec_str.join(' ').print('\n','\n');

    cout << "========================================\n";
    cout << "testing xvector discovery\n\n";
    xvector<char>    vec_char{ 'v','a','r','c','h','a','r' };

    const char v1[] = "var1";
    const char v2[] = "var2";
    const char v3[] = "var3";
    const char v4[] = "var4";
    xvector<const char*> vec_chara{ v1, v2, v3, v4 };

    xvector<int>   vec_int{ 0,1,2,3,4,5,6 };

    cout << "xstring : " << vec_str.join(' ') << endl;
    cout << "chars   : " << vec_char.convert<xstring>().join(' ') << endl;
    cout << "char arr: " << vec_chara.convert<xstring>([](const char* val) {return xstring(val); }).join(' ') << endl;
    cout << "ints    : " << vec_int.convert<xstring>([](int val) {return to_xstring(val); }).join(' ') << endl;

    xstring three = "three";

    xvector<xstring*> vec_str_ptrs = vec_str.ptrs();
    cout << "\nprint ptrs: " << vec_str_ptrs.join(' ') << '\n';

    xvector<xstring> vec_str_vals = vec_str_ptrs.vals();
    cout << "print vals: " << vec_str_vals.join(' ') << "\n\n";

    cout << "true  => " << vec_str.has(three) << endl;
    cout << "true  => " << vec_str.has("two") << endl;
    cout << "false => " << vec_str.has("twenty") << endl;

    cout << "========================================\n";
    cout << "testing regex\n\n";
    xvector<xstring> emails{
       "first.last@subdomain.domain.com",
       "realuser@gmail.com",
       "trap@attack.hack.info"
    };

    cout << "false email = " << emails.match_one(R"([\w\d_]*@[\w\d_]*)") << endl;
    // that failed because it does not match any one email start to finish
    cout << "true  email = " << emails.match_one(R"(^([\w\d_]*(\.?)){1,2}@([\w\d_]*\.){1,2}[\w\d_]*$)") << endl;;
    // this one matches the first email start to finish so it replies true

    cout << "true email  = " << emails.scan_one("[\\w\\d_]*@[\\w\\d_]*") << endl;
    // this time the partial email returns true because the regex matches at least a portion
    // of one of the elements in the array.

    xvector<xstring> found_email = emails.take("^([\\w\\d_]*(\\.?)){1,2}@([\\w\\d_]*\\.){1,2}[\\w\\d_]*$");
    // take the first email that matches that regex

    cout << "\nemails found\n" << found_email.join('\n') << endl;

    cout << "\nafter email removed\n" << emails.remove("trap@[\\w\\d_]*").join('\n') << "\n\n";


    xvector<xstring> tripple = { "one","two","three" };
    tripple *= 4;
    tripple.split(5).proc([](auto& xvec) {xvec.join(' ').print(); return false; /* (false means never break) */});
    cout << '\n';


    auto sub_function_val = [](xstring& elem, xstring& str) -> xstring {
        return elem.sub("gmail", str);
    };
    auto sub_function_val2 = [](xstring& elem, xstring& junk, xstring& str) -> xstring {
        return elem.sub("gmail", str);
    };
    auto sub_function_ptr = [](xstring* elem, xstring& str) -> xstring {
        return elem->sub("gmail", str);
    };

    xvector<xstring>arr = { "one@gmail.com","two@gmail.com","three@gmail.com" };

    arr.xrender(sub_function_val2, xstring("junk"), xstring("render")).join("\n").print(2);

    arr.xrender(
        [](auto& elem, auto& val) 
        {
            return elem.sub("gmail", val); 
        },
        "new_val"
    ).join('\n').print(2);

    xstring x_new = "new_ref";
    arr.xrender([&x_new](auto& elem) {return elem.sub("gmail", x_new); }).join('\n').print(2);

    arr.xrender([](auto& elem, xstring def_replace = "default") {return elem.sub("gmail", def_replace); }).join('\n').print(2);

    xvector<xstring*> arr_ptr = arr.ptrs();
    arr_ptr.render([](xstring* elem, xstring str = "mod_ptr1") {return elem->sub("gmail", str); }).join('\n').print();
    arr_ptr.render(sub_function_ptr, xstring("mod_ptr2")).join('\n').print();

    arr_ptr.xrender([](xstring& elem, xstring str = "mod_ptr3") {return elem.sub("gmail", str); }).join('\n').print();
    arr_ptr.xrender(sub_function_val, xstring("mod_ptr4")).join('\n').print();
    xstring mod5 = "mod_ptr5";
    arr_ptr.xrender([&mod5](xstring& elem) {return elem.sub("gmail", mod5); }).join('\n').print();


    arr_ptr.xproc([&mod5](xstring& elem) {return elem.sub("gmail", mod5); }); // executes but returns nothing

    cout << "============================================\n";

    xvector<xstring> four_five_six = { "seven","eight","nine","nine" };
    cout << "vec 1: " << vec_str.join(' ') << endl;
    cout << "vec 2: " << four_five_six.join(' ') << endl;

    xvector<xstring> vec_common_values = vec_str.common(four_five_six);
    cout << "common values: " << vec_common_values.join(' ') << endl;


    xvector<xstring> first = { "one" , "two", "three" };
    xvector<xstring> second = { "four", "five", "six" };

    xvector<xvector<xstring>> nested_vecs = { first, second };
    xvector<xvector<xstring>*> nested_vec_ptrs = nested_vecs.ptrs();
    xvector<xstring*> un_nested_vec_ptrs = nested_vec_ptrs.expand();
    cout << "ptrs: " << un_nested_vec_ptrs.join(' ') << endl;
    
    
    return 0;
}
