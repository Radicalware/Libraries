#include "stdafx.h"

#include<iostream>
#include<vector>
#include<string>

#include "ord.h"
#include "OS.h"

#include "ex_open_n_delete.h"
#include "ex_file_managment.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

extern OS os;

int main() {

	ex_open_n_delete();
	ex_file_managment();

	return 0;
}