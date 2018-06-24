#include<iostream>
#include<vector>
#include<string>

#include "./re.h" // Found on "https://github.com/Radicalware"

#include <ctime>


using std::cout;
using std::endl;
using std::string;
using std::vector;


// g++ -g $file.cpp -o $file -std=c++17 -Wfatal-errors

void benchmark_slice_string(std::string& str);

int main()
{


    cout << "\n===(SLICE)======================================================\n";
    cout << "based on the python string[int:int:int] mechanics\n\n";


    std::string str =  "<123*abc^456*def>"; // odd  len 
    std::string loc1 = "^   ^   ^   ^   ^";
    std::string loc2 = "0   4   8   12  16";

    std::string str0 = "";
    int L = str.length();
    std::string underline = "_____________________________________________________\n\n";
    std::string d_underline = "_____________________________________________________\n_____________________________________________________\n\n";
    cout << "String                     = " << str << "\n";
    cout << "string item locator        = " << loc1 << "\n";
    cout << "                             " << loc2 << "\n\n";
    // NOTE: 0 will default to max_leng

    // [starting (inclusive): ending (non-inclusive): iterator]
    cout << underline;
    // +, + , +     x < y
    cout << "from arr[1] to arry[4 - 1]\n\n";
    cout << str << "   1,   4,   0 = " << re::slice(str,   1,   4,   0) << '\n'; // 123
    cout << str << "   5,  12,   0 = " << re::slice(str,   5,  12,   0) << '\n'; // abc^456
    cout << str << "   0,  str.length(),   4 = " << re::slice(str,   0,  str.length(),   4) << '\n'; // 

    cout << "\nstart from str[3] and go to str[str.length()]\n";
    cout << str << "   3,  str.length(),   0 = " << re::slice(str,   3,  str.length(),   0) << '\n'; // 
    
    cout << "\nSame as the last slice. '0' will subsitute for the end of the string\n";
    cout << "If 'z' is positive, '0' = str.length(); else z = '0'\n";
    cout << str << "   3,   0,   0 = " << re::slice(str,   3,  0,   0) << '\n'; // 
    cout << str << "   0,   4,  -1 = " << re::slice(str,   0,  4,  -1) << '\n'; // 
    // above we start at the beginning (x = 0) and go to str[4+1] position


    cout << underline;

    // +, +, -      x > y
    cout << "Specify reverse direction 'z' is -1' then\nfrom arr[7] >> arr[4 + 1]\n\n";
    cout << str << "   7,   4,  -1 = " << re::slice(str,   7,   4,  -1) << '\n'; // cba
    cout << str << "  12,   3,  -1 = " << re::slice(str,  12,   3,  -1) << '\n'; // *654^cba*
    cout << str << "  12,   3,  -3 = " << re::slice(str,  12,   3,  -3) << '\n'; // 
    cout << d_underline;

    // --------------------------------------------------------------------------------
    // -, +, +      x < y
    cout << "from arr[arr.length() - 8] >> arr[ 1 2 -1]\n\n";
    cout << str << "  -8,  12,   0 = " << re::slice(str,  -8,  12,   0) << '\n'; // 456
    cout << str << " -15,  11,   0 = " << re::slice(str, -15,  11,   0) << '\n'; // 23*abc^45
    cout << str << " -15,  11,   3 = " << re::slice(str, -15,  11,   3) << '\n'; // 
    cout << underline;

    // -, +, -      x < y
    cout << "Specify reverse direction 'z' is -1' then\nfrom arr[arr.length() - 6] >> arr[8 + 1]\n\n";
    cout << str << "  -6,   8,  -1 = " << re::slice(str,  -6,   8,  -1) << '\n'; // 654
    cout << str << "  -8,   2,  -1 = " << re::slice(str,  -8,   2,  -1) << '\n'; // 4^cba*3
    cout << str << "  -8,   2,  -2 = " << re::slice(str,  -8,   2,  -2) << '\n'; // 
    cout << d_underline;

    // --------------------------------------------------------------------------------
    // +, -, +
    cout << "from arr[5] >> arr[arr.length() - 4 - 1]\n\n";
    cout << str << "   5,  -4,   0 = " << re::slice(str,   5,  -4,   0) << '\n'; // abc^456*
    cout << str << "   3,  -3,   0 = " << re::slice(str,   3,  -3,   0) << '\n'; // 3*abc^456*d
    cout << str << "   3,  -3,   4 = " << re::slice(str,   3,  -3,   4) << '\n'; // 
    cout << underline;

    // +, -, -
    cout << "Specify reverse direction 'z' is -1' then\nfrom arr[8] >> arr[arr.length()-15 + 1]\n\n";
    cout << str << "   8, -15,  -1 = " << re::slice(str,   8, -15,   -1) << '\n'; // ^cba*3
    cout << str << "  12, -14,  -1 = " << re::slice(str,  12, -14,   -1) << '\n'; // *654^cba*
    cout << str << "  12, -14,  -2 = " << re::slice(str,  12, -14,   -2) << '\n'; // 
    cout << d_underline;

    // --------------------------------------------------------------------------------
    // -, -, +
    cout << "from arr[5] >> arr[arr.length()-1]\n\n";
    cout << str << "  -5,  -1,   0 = " << re::slice(str,  -5,  -1,   0) << '\n'; // *def
    cout << str << " -12,  -5,   0 = " << re::slice(str, -12,  -5,   0) << '\n'; // abc^456
    cout << str << " -12,  -5,   2 = " << re::slice(str, -12,  -5,   2) << '\n'; //
    cout << underline;

    // -, -, -
    cout << "from arr[arr.length() - 5] >> arr[arr.length() - 13 + 1]\n\n";
    cout << str << "  -5, -13,  -1 = " << re::slice(str,  -5, -13,  -1) << '\n'; // *654^cba
    cout << str << " -12, -15,  -1 = " << re::slice(str, -12, -15,  -1) << '\n'; // 4^cba*3
    cout << str << "  12, -15,  -4 = " << re::slice(str,  12, -15,  -4) << '\n'; //
    cout << underline;
    // --------------------------------------------------------------------------------

    benchmark_slice_string(str);

}



void benchmark_slice_string(std::string& str){


    double ishort = 32;
    double ilong  = 9999999;

    double used_size = ilong;


    std::clock_t    start;
    start = std::clock();

    std::string strx;
    for(int i = 0; i < used_size; i ++){
        strx = re::slice(str, 4, -5, -2);
        // trim off the last 4 chars and print in reverse order
    }

    double result = (std::clock() - start) / (double)(CLOCKS_PER_SEC);

    cout << "_____________________________________________________\n";
    cout << "Slice Benchmark";
    cout << "\nLooped benchmark Count: " << used_size << "\n\n";
    cout << "Elapsed Time in Seconds\n" << result << endl;
    cout << '\n';
}  
 

// g++ -g $file.cpp -o $file -std=c++17 -Wfatal-errors
