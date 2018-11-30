#include "OS.h"

#include<iostream>
#include<vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;

extern OS os;

int main() {
	// clear_file()
	// remove_file()
	// move_file()
	// copy_file()

	// popen()/read()
	//  open()/read()

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

	cout << "\n------------------ Current Dirs -----------------------\n";
	for (auto&i : os.dir("./test_folder1", 'r','d','f'))
		cout << "item = " << i << endl;

	os.move_dir("./test_folder1", "./test_folder2");

	cout << "\n------------------ Moved Dirs -------------------------\n";
	for (auto&i : os.dir("./test_folder2", 'r','d','f'))
		cout << "item = " << i << endl;

	os.delete_dir("./test_folder2");

	cout << "\n------------------ Removed Dirs 1 ---------------------\n";
	vector<string> items = os.dir("./test_folder1", 'r','d','f');
	for (auto&i : items)
		cout << "item = " << i << endl;

	cout << "\n------------------ Deleted Dirs 2 ---------------------\n";
	for (auto&i : os.dir("./test_folder2", 'r','d','f'))
		cout << "item = " << i << endl;
	cout << '\n';
}

// 'r','d','f'
// order does not matter
// 'r' = recursive
// 'd' = directories
// 'f' = files
