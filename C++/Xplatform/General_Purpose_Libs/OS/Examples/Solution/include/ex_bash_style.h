#pragma once
#pragma warning ( disable : 26444) // Allow un-named objects

#include<iostream>
using std::cout;
using std::endl;

#include "xvector.h"
#include "OS.h"
extern OS os;

void ex_bash_style() 
{
    // all static OS functions start with an Upper_Case

    os.touch("test_file.txt"); 
    // has will work as both "has_file" and "has_dir"
    if (os.Has_File("./test_file.txt") && os.Has("./test_file.txt")) {
        cout << "test_data.txt was created\n";
    } else {
        cout << "error: test_data.txt should have been created\n"; exit(1);
    }
    os.write("random data\n");

    if (os.read() == "random data\n") {
        cout << "data was written\n";
    } else {
        cout << "error: data didn't write\n"; exit(1);
    }

    os.MKDIR("./tmp_dir");
    os.CP("./test_file.txt", "./tmp_dir/test_file1.txt");
    // could have used
    //os.file.copy("./tmp_dir/test_file1.txt");

    os.file.close(); // test_file.txt will not delete if we don't close it first
    os.MV("./test_file.txt", "./tmp_dir/test_file2.txt"); 
    // could have used
    //os.file.move("./tmp_dir/test_file2.txt");

    if (os.Has("./test_file.txt")) {
        cout << "error: data was not removed\n"; exit(1);
    } else {
        cout << "./test_file.txt was removed\n";
    }

    if (os.Has("./tmp_dir/test_file1.txt") && os.Has("./tmp_dir/test_file2.txt")) {
        cout << "data was copied & moved correctly\n";
    } else {
        cout << "error: data was not coppied or moved\n"; exit(1);
    }

    if (os.open("./tmp_dir/test_file1.txt").read() == "random data\n" && \
        os.open("./tmp_dir/test_file2.txt").read() == "random data\n") {

        cout << "data was coppied correctly\n";
    } else {
        cout << "error: corrupted copy\n"; exit(1);
    }

    os.close(); // same as os.file.close();
    os.RM("./tmp_dir");

    if (os.Has("./tmp_dir")) {
        // note: has will return true for either 
        // "file()" or  "folder()"
        cout << "error: tmp_dir was not delted\n"; exit(1);
    } else {
        cout << "tmp_dir was deleted\n";
    }

    cout << "All Bash File Managment was SUCCESSFUL!!\n\n";
}
