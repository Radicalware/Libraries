#pragma once
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence
#pragma warning ( disable : 26444) // Allow un-named objects

#include "OS.h"

#include "xvector.h"
#include "xstring.h"

#include<iostream>
using std::cout;
using std::endl;


void ex_open_n_delete() 
{    
    Begin();
	// all static OS functions start with an Upper_Case
    RA::OS OS;

    OS.RM("./test_folder1");
    OS.RM("./test_folder2");

    OS.MKDIR("./test_folder1/nested_folder_a");
    OS.MKDIR("./test_folder1/nested_folder_b");
    OS.MKDIR("./test_folder1/nested_folder/double_nested_folder");

    OS.Open("./test_folder1/nested_folder/double_nested_folder/file1", 'w').Write("test data");
    OS.Open("./test_folder1/nested_folder/double_nested_folder/file2", 'w').Write("test data");
    OS.Open("./test_folder1/nested_folder/double_nested_folder/file3", 'w').Write("test data");

    OS.Open("./test_folder1/nested_folder_a//file1", 'w').Write("test data");
    OS.Open("./test_folder1/nested_folder_a//file2", 'w').Write("test data");
    OS.Open("./test_folder1/nested_folder_a//file3", 'w').Write("test data");

    OS.Open("./test_folder1/nested_folder_b//file1", 'w').Write("test data");
    OS.Open("./test_folder1/nested_folder_b//file2", 'w').Write("test data");
    OS.Open("./test_folder1/nested_folder_b//file3", 'w').Write("test data");

    OS.Open("./test_folder1///file1", 'w').Write("test data");
    OS.Open("./test_folder1///file2", 'w').Write("test data");
    OS.Open("./test_folder1///file3", 'w').Write("test data");
    
    cout << "******************************************************" << endl;
    cout << "file   = " << OS.Has("./test_folder1///file3") << endl; // true
    cout << "folder = " << OS.Has("./test_folder1/nested_folder_b/") << endl; // true
    cout << "zilch  = " << OS.Has("./asdfasdf/") << endl; // false
    cout << "******************************************************" << endl;

    cout << "\n------------------ Current Dirs -----------------------\n";
    for (auto&i : OS.Dir("./test_folder1", 'r', 'd', 'f'))
        cout << "item = " << i << endl;

    OS.Close();
    OS.MoveDir("./test_folder1", "./test_folder2");
    
    cout << "\n------------------ Moved Dirs -------------------------\n";
    for (auto&i : OS.Dir("./test_folder2", 'r', 'd', 'f'))
        cout << "item = " << i << endl;

    OS.RemoveDir("./test_folder2");

    cout << "\n------------------ Removed Dirs 1 ---------------------\n";
    for (auto&i : OS.Dir("./test_folder1", 'r', 'd', 'f'))
        cout << "item = " << i << endl;

    cout << "\n------------------ Deleted Dirs 2 ---------------------\n";
    for (auto&i : OS.Dir("./test_folder2", 'r', 'd', 'f'))
        cout << "item = " << i << endl;

    cout << "\n\n\n";
    RescueThrow();
}




