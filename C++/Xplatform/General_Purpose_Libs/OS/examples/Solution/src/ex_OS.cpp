#include<iostream>
#include<vector>
#include<string>

#include "OS.h"
//extern OS os;

#include "ex_open_n_delete.h"
#include "ex_file_managment.h"
#include "ex_bash_style.h"

using std::cout;
using std::endl;
using std::string;

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
	#include<vld.h>
#endif

OS os;

int main() {

	ex_open_n_delete();
	ex_file_managment();
	ex_bash_style();


	return 0;
}
