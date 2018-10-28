#pragma once

#include<iostream>
#include<vector>
#include<string>

#include "ord.h"
#include "OS.h"
extern OS os;

using std::cout;
using std::endl;
using std::string;
using std::vector;


void ex_bash_style() {

	os.open("./test_file.txt", 'w').touch(); // 'a' would append; 'w' overWrites

	if (os.has_file("./test_file.txt") && os.has("./test_file.txt")) {
		cout << "test_data.txt was created\n" << endl;
	} else {
		cout << "error: test_data.txt should have been created" << endl; exit(1);
	}

	os.write("random data\n");

	if (os.read() == "random data\n") {
		cout << "data was written\n";
	} else {
		cout << "error: data didn't write" << endl; exit(1);
	}

	os.mkdir("./tmp_dir");
	os.cp("./test_file.txt", "./tmp_dir/test_file1.txt");
	os.mv("./test_file.txt", "./tmp_dir/test_file2.txt");


	if (os.has_file("./test_file.txt")) {
		cout << "error: data was not removed\n"; exit(1);
	} else {
		cout << "./test_file.txt was removed\n";
	}

	if (os.has_file("./tmp_dir/test_file1.txt") && os.has_file("./tmp_dir/test_file2.txt")) {
		cout << "data was copied & moved correctly" << endl;
	} else {
		cout << "error: data was not coppied or moved\n"; exit(1);
	}

	if (os.open("./tmp_dir/test_file1.txt").read() == "random data\n" && \
		os.open("./tmp_dir/test_file2.txt").read() == "random data\n") {

		cout << "data was coppied correctly\n";
	} else {
		cout << "error: corrupted copy\n"; exit(1);
	}

	os.rm("./tmp_dir");

	if (os.has("./tmp_dir")) {
		// note: has will return true for either 
		// "has_file()" or  "has_folder()"
		cout << "error: tmp_dir was not delted\n"; exit(1);
	} else {
		cout << "tmp_dir was deleted\n";
	}

	cout << "All Bash File Managment was SUCCESSFUL!!\n";
}