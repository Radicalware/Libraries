#include <dirent.h> 
#include <iostream> 
#include <vector>
#include <string>

#include "dir_scan.h"
// Programed by Scourge
// Apache License v2.0

using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::vector;

// g++ example_dir_scan.cpp dir_scan.hpp -o ex --std=c++17 -Wfatal-errors && ./ex
int main(){

	vector<string> rec_vec;
	vector<string> pwd_vec;

	// './' as arg 1 sets the target folder to your PWD

	scan_dir("./","-r","-f", rec_vec); // "scope"          = '-r' =  target_folder recursive search
	// recursive vector from pwd       // "include_folder" = '-f' = include folders in the results

	scan_dir("./","-n","-n", pwd_vec); // "scope"          = '-n' = use target_folder but not recursive  
	// pwd vector           	       // "include_folder  = '-n' = not folder included


	for(string& i: rec_vec){cout << i << endl;} // loop through and print the vector's elements
	cout << "---------------------\n";
	for(string& i: pwd_vec){cout << i << endl;} // loop through and print the vector's elements

	return 0;
}

 
