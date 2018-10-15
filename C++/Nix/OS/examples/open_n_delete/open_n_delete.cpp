#include <iostream>

#include "OS.h"

int main(){

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

	for(auto&i : os.dir("./test_folder1","files","folders", "star", "recursive")) cout << i << endl;

	os.rmdir("./test_folder1");

	cout << '\n';
	return 0;
}