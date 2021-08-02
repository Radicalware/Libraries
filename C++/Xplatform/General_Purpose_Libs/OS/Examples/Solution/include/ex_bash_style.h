#pragma once
#pragma warning ( disable : 26444) // Allow un-named objects

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
using std::cout;
using std::endl;

#include "xvector.h"
#include "OS.h"
extern OS os;

void ex_bash_style() 
{
    // all static OS functions start with an Upper_Case

    os.Touch("test_file.txt"); 
    // has will work as both "has_file" and "has_dir"
    if (os.HasFile("./test_file.txt") && os.Has("./test_file.txt")) {
        cout << "test_data.txt was created\n";
    } else {
        cout << "error: test_data.txt should have been created\n"; exit(1);
    }
    os.Write("random data\n", 'w').Close();

    if (os.Read() == "random data\n") {
        cout << "data was written\n";
    } else {
        cout << xstring("error: data didn't write\n").ToRed(); exit(1);
    }

    os.MKDIR("./tmp_dir");
    os.CP("./test_file.txt", "./tmp_dir/test_file1.txt");
    // could have used
    //os.file.Copy("./tmp_dir/test_file1.txt");

    os.File.Close(); // test_file.txt will not delete if we don't close it first
    os.MV("./test_file.txt", "./tmp_dir/test_file2.txt"); 
    // could have used
    //os.file.Move("./tmp_dir/test_file2.txt");

    if (os.Has("./test_file.txt")) {
        cout << xstring("error: data was not removed\n").ToRed(); exit(1);
    } else {
        cout << "./test_file.txt was removed\n";
    }

    if (os.Has("./tmp_dir/test_file1.txt") && os.Has("./tmp_dir/test_file2.txt")) {
        cout << "data was copied & moved correctly\n";
    } else {
        cout << xstring("error: data was not coppied or moved\n").ToRed(); exit(1);
    }

    if (os.Open("./tmp_dir/test_file1.txt").Read() == "random data\n" && \
        os.Open("./tmp_dir/test_file2.txt").Read() == "random data\n") {

        cout << "data was coppied correctly\n";
    } else {
        cout << xstring("error: corrupted copy\n").ToRed(); exit(1);
    }

    os.Close(); // same as os.file.Close();
    os.RM("./tmp_dir");

    if (os.Has("./tmp_dir")) {
        // note: has will return true for either 
        // "File()" or  "folder()"
        cout << xstring("error: tmp_dir was not delted\n").ToRed(); exit(1);
    } else {
        cout << "tmp_dir was deleted\n";
    }

    cout << xstring("All Bash File Managment was SUCCESSFUL!!\n").ToGreen();
}
