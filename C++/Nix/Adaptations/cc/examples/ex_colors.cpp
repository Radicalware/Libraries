#include<iostream>
#include<string>
#include<vector>

// ----------------------------------------------------
// The following can be downloaded from 
// https://github.com/Radicalware/Libraries

#include "OS.h"  // by: Joel leagues aka Scouge
#include "SYS.h" // by: Joel leagues aka Scouge
extern OS os;
extern SYS sys;

#include "ord.h" // by: Joel leagues aka Scouge
#include "re.h"  // by: Joel leagues aka Scouge
#include "cc.h"  // by: Ihor Kalnytskyi
// ----------------------------------------------------

using std::cout;
using std::endl;
using std::string;
using std::vector;


void help(int ret_err = 0);

int main(int argc, char** argv){

	cout << "Test Colors\n";
	cout << cc::grey << "Test grey\n";
	cout << cc::black  << "Test black\n";
	cout << cc::red   << "Test red\n";
	cout << cc::green << "Test green\n";
	cout << cc::white << "Test white\n";

	cout << cc::on_grey << cc::red << "white background & red text" << cc::reset << '\n' ;

    return 0;
}
