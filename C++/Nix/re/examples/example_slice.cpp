#include<iostream>
#include<vector>
#include<string>
#include<typeinfo> // debugging
#include<chrono>


#include "./slice.h" // Found on "https://github.com/Radicalware"



using std::cout;
using std::endl;
using std::string;
using std::vector;


// g++ -O2 $file.cpp -o $file -std=c++17 -Wfatal-errors


class Timer // class by learncpp
{
private:
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double, std::ratio<1> >;
    std::chrono::time_point<clock_t> m_beg;

public:
    Timer() : m_beg(clock_t::now())
    {   }

    void reset(){ 
        m_beg = clock_t::now(); 
    }
    double elapsed() const  { 
        return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count();    
    }
};

void benchmark_slice_string(std::string& str);

int main()
{


    cout << "\n===(SLICE)======================================================\n";
    cout << "based on the python string[int:int:int] mechanics\n\n";


    std::string str =  "<123*abc^456*def>"; // odd  len 
    vector<char> vec {'<','1','2','3','*','a','b','c','^','4','5','6','*','d','e','f','>'}  ;

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

    // for(auto& i :re::slice<std::string>(vec,   1,   4,   0))
    //     cout << i << ' ';

    // [starting (inclusive): ending (non-inclusive): iterator]
    cout << underline;
    // +, + , +     x < y
    cout << "from arr[1] to arry[4 - 1]\n\n";
    cout << str << "   1,   4,   0 = " << re::slice<std::string>(str,   1,   4,   0) << '\n'; // 123
    cout << str << "   5,  12,   0 = " << re::slice<std::string>(str,   5,  12,   0) << '\n'; // abc^456
    cout << str << "   0,  str.length(),   4 = " << re::slice<std::string>(str,   0,  str.length(),   4) << '\n'; // 

    cout << "\nstart from str[3] and go to str[str.length()]\n";
    cout << str << "   3,  str.length(),   0 = " << re::slice<std::string>(str,   3,  str.length(),   0) << '\n'; // 
    
    cout << "\nSame as the last slice. '0' will subsitute for the end of the string\n";
    cout << "If 'z' is positive, '0' = str.length(); else z = '0'\n";
    cout << str << "   3,   0,   0 = " << re::slice<std::string>(str,   3,  0,   0) << '\n'; // 3*abc^456*def>
    cout << str << "   0,   4,  -1 = " << re::slice<std::string>(str,   0,  4,  -1) << '\n'; // >fed*654^cba
    // above we start at the beginning (x = 0) and go to str[4+1] position


    cout << underline;

    // +, +, -      x > y
    cout << "Specify reverse direction 'z' is -1' then\nfrom arr[7] >> arr[4 + 1]\n\n";
    cout << str << "   7,   4,  -1 = " << re::slice<std::string>(str,   7,   4,  -1) << '\n'; // cba
    cout << str << "  12,   3,  -1 = " << re::slice<std::string>(str,  12,   3,  -1) << '\n'; // *654^cba*
    cout << str << "  12,   3,  -3 = " << re::slice<std::string>(str,  12,   3,  -3) << '\n'; // *4b
    cout << d_underline;

    // --------------------------------------------------------------------------------
    // -, +, +      x < y
    cout << "from arr[arr.length() - 8] >> arr[ 1 2 -1]\n\n";
    cout << str << "  -8,  12,   0 = " << re::slice<std::string>(str,  -8,  12,   0) << '\n'; // 456
    cout << str << " -15,  11,   0 = " << re::slice<std::string>(str, -15,  11,   0) << '\n'; // 23*abc^45
    cout << str << " -15,  11,   3 = " << re::slice<std::string>(str, -15,  11,   3) << '\n'; // 2a^
    cout << underline;

    // -, +, -      x < y
    cout << "Specify reverse direction 'z' is -1' then\nfrom arr[arr.length() - 6] >> arr[8 + 1]\n\n";
    cout << str << "  -6,   8,  -1 = " << re::slice<std::string>(str,  -6,   8,  -1) << '\n'; // 654
    cout << str << "  -8,   2,  -1 = " << re::slice<std::string>(str,  -8,   2,  -1) << '\n'; // 4^cba*3
    cout << str << "  -8,   2,  -2 = " << re::slice<std::string>(str,  -8,   2,  -2) << '\n'; // 4ca3
    cout << d_underline;

    // --------------------------------------------------------------------------------
    // +, -, +
    cout << "from arr[5] >> arr[arr.length() - 4 - 1]\n\n";
    cout << str << "   5,  -4,   0 = " << re::slice<std::string>(str,   5,  -4,   0) << '\n'; // abc^456*
    cout << str << "   3,  -3,   0 = " << re::slice<std::string>(str,   3,  -3,   0) << '\n'; // 3*abc^456*d
    cout << str << "   3,  -3,   4 = " << re::slice<std::string>(str,   3,  -3,   4) << '\n'; // 3c6
    cout << underline;

    // +, -, -
    cout << "Specify reverse direction 'z' is -1' then\nfrom arr[8] >> arr[arr.length()-15 + 1]\n\n";
    cout << str << "   8, -15,  -1 = " << re::slice<std::string>(str,   8, -15,   -1) << '\n'; // ^cba*3
    cout << str << "  12, -14,  -1 = " << re::slice<std::string>(str,  12, -14,   -1) << '\n'; // *654^cba*
    cout << str << "  12, -14,  -2 = " << re::slice<std::string>(str,  12, -14,   -2) << '\n'; // *5^b*
    cout << d_underline;

    // --------------------------------------------------------------------------------
    // -, -, +
    cout << "from arr[5] >> arr[arr.length()-1]\n\n";
    cout << str << "  -5,  -1,   0 = " << re::slice<std::string>(str,  -5,  -1,   0) << '\n'; // *def
    cout << str << " -12,  -5,   0 = " << re::slice<std::string>(str, -12,  -5,   0) << '\n'; // abc^456
    cout << str << " -12,  -5,   2 = " << re::slice<std::string>(str, -12,  -5,   2) << '\n'; // ac46
    cout << underline;

    // -, -, -
    cout << "from arr[arr.length() - 5] >> arr[arr.length() - 13 + 1]\n\n";
    cout << str << "  -5, -13,  -1 = " << re::slice<std::string>(str,  -5, -13,  -1) << '\n'; // *654^cba
    cout << str << " -12, -15,  -1 = " << re::slice<std::string>(str, -12, -15,  -1) << '\n'; // a*3
    cout << str << "  12, -15,  -4 = " << re::slice<std::string>(str,  12, -15,  -4) << '\n'; // *^*
    cout << underline;
    // --------------------------------------------------------------------------------

    benchmark_slice_string(str);

}



void benchmark_slice_string(std::string& str){

    size_t size = 9;
    char* char_string = new char[size]{'0','1','2','3','4','5','6','7','8'};

    double ishort = 32;
    double ilong  = 9999999;

    double used_size = ilong;


    Timer t;


    std::string strx;
    for(int i = 0; i < used_size; i ++){
        //re::slice<std::string>(str, 4, -5, -2);
        strx = re::slice<string>(str,0,8,0,size).ret();
        // trim off the last 4 chars and print in reverse order
    }

    cout << "_____________________________________________________\n";
    cout << "Slice Benchmark";
    cout << "\nLooped benchmark Count: " << used_size << "\n\n";
    cout << "Elapsed Time in Seconds\n" << t.elapsed() << endl;
    cout << '\n';
}  
 

// g++ -g $file.cpp -o $file -std=c++17 -Wfatal-errors
