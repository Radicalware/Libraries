#pragma once

#include "os.h"

#include<iostream>

using std::cout;
using std::endl;
using std::string;

extern OS os;

void ex_open_n_delete() {
#if defined(NIX_base)

	// clear_file()
	// remove_file()
	// move_file()
	// copy_file()

	// popen()/read()
	//  open()/read()

	os.mkdir("./test_folder1/test_folder2/test_folder3");
	os.mkdir("./test_folder1/test_folder2a/");
	os.mkdir("./test_folder1/test_folder2b/");

	os.open("./test_folder1/test_folder2/test_folder3/file1", 'w').write("test data");
	os.open("./test_folder1/test_folder2/test_folder3/file2", 'w').write("test data");
	os.open("./test_folder1/test_folder2/test_folder3/file3", 'w').write("test data");

	os.open("./test_folder1/test_folder2a//file1", 'w').write("test data");
	os.open("./test_folder1/test_folder2a//file2", 'w').write("test data");
	os.open("./test_folder1/test_folder2a//file3", 'w').write("test data");

	os.open("./test_folder1/test_folder2b//file1", 'w').write("test data");
	os.open("./test_folder1/test_folder2b//file2", 'w').write("test data");
	os.open("./test_folder1/test_folder2b//file3", 'w').write("test data");

	os.open("./test_folder1///file1", 'w').write("test data");
	os.open("./test_folder1///file2", 'w').write("test data");
	os.open("./test_folder1///file3", 'w').write("test data");

	for (auto&i : os.dir("./test_folder1", "files", "folders", "recursive")) cout << i << endl;

	os.rmdir("./test_folder1");

	cout << '\n';

#elif defined(WIN_BASE)

	os.mkdir(".\\test_folder1\\test_folder2\\test_folder3");
	os.mkdir(".\\test_folder1\\test_folder2a\\");
	os.mkdir(".\\test_folder1\\test_folder2b\\");

	os.open(".\\test_folder1\\test_folder2\\test_folder3\\file1", 'w').write("test data");
	os.open(".\\test_folder1\\test_folder2\\test_folder3\\file2", 'w').write("test data");
	os.open(".\\test_folder1\\test_folder2\\test_folder3\\file3", 'w').write("test data");

	os.open(".\\test_folder1\\test_folder2a\\file1", 'w').write("test data");
	os.open(".\\test_folder1\\test_folder2a\\file2", 'w').write("test data");
	os.open(".\\test_folder1\\test_folder2a\\file3", 'w').write("test data");

	os.open(".\\test_folder1\\test_folder2b\\file1", 'w').write("test data");
	os.open(".\\test_folder1\\test_folder2b\\file2", 'w').write("test data");
	os.open(".\\test_folder1\\test_folder2b\\file3", 'w').write("test data");

	os.open(".\\test_folder1\\file1", 'w').write("test data");
	os.open(".\\test_folder1\\file2", 'w').write("test data");
	os.open(".\\test_folder1\\file3", 'w').write("test data"); 

	cout << "\n------------------ current dirs -----------------------\n";
	for (auto&i : os.dir(".\\test_folder1", "files", "folders", "recursive")) cout << "item = " << i << endl;

	os.rmdir(".\\test_folder1");

	cout << "\n------------------ dirs gone --------------------------\n";
	for (auto&i : os.dir(".\\test_folder1", "files", "folders", "recursive")) cout << "item = " << i << endl;

#endif	
	cout << '\n';
}




