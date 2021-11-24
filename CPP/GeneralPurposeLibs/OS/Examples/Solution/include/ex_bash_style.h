#pragma once
#pragma warning ( disable : 26444) // Allow un-named objects

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
using std::cout;
using std::endl;

#include "xvector.h"
#include "OS.h"

void ex_bash_style() 
{
    Begin();
    // all static OS functions start with an Upper_Case
    RA::OS OS;
    OS.Touch("test_file.txt"); 
    // has will work as both "has_file" and "has_dir"
    if (OS.HasFile("./test_file.txt") && OS.Has("./test_file.txt")) {
        cout << "test_data.txt was created\n";
    } else {
        cout << "error: test_data.txt should have been created\n"; exit(1);
    }
    OS.Write("random data\n", true).Close();

    if (OS.Read() == "random data\n") {
        cout << "data was written\n";
    } else {
        cout << xstring("error: data didn't write\n").ToRed(); exit(1);
    }

    OS.MKDIR("./tmp_dir");
    OS.CP("./test_file.txt", "./tmp_dir/test_file1.txt");
    // could have used
    //OS.file.Copy("./tmp_dir/test_file1.txt");

    OS.File.Close(); // test_file.txt will not delete if we don't close it first
    OS.MV("./test_file.txt", "./tmp_dir/test_file2.txt"); 
    // could have used
    //OS.file.Move("./tmp_dir/test_file2.txt");

    if (OS.Has("./test_file.txt")) {
        cout << xstring("error: data was not removed\n").ToRed(); exit(1);
    } else {
        cout << "./test_file.txt was removed\n";
    }

    if (OS.Has("./tmp_dir/test_file1.txt") && OS.Has("./tmp_dir/test_file2.txt")) {
        cout << "data was copied & moved correctly\n";
    } else {
        cout << xstring("error: data was not coppied or moved\n").ToRed(); exit(1);
    }

    if (OS.Open("./tmp_dir/test_file1.txt").Read() == "random data\n" && \
        OS.Open("./tmp_dir/test_file2.txt").Read() == "random data\n") {

        cout << "data was coppied correctly\n";
    } else {
        cout << xstring("error: corrupted copy\n").ToRed(); exit(1);
    }

    OS.Close(); // same as OS.file.Close();
    OS.RM("./tmp_dir");

    if (OS.Has("./tmp_dir")) {
        // note: has will return true for either 
        // "File()" or  "folder()"
        cout << xstring("error: tmp_dir was not delted\n").ToRed(); exit(1);
    } else {
        cout << "tmp_dir was deleted\n";
    }

    cout << xstring("All Bash File Managment was SUCCESSFUL!!\n").ToGreen();
    RescueThrow();
}
