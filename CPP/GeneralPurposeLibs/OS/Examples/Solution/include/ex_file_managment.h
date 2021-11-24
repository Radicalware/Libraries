#pragma once

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#include "OS.h"

void ex_file_managment() 
{
    Begin();
    // all static OS functions start with an Upper_Case
    RA::OS OS;
    cout << "Present Working Directory = " << RA::OS::PWD() << endl;
    cout << "Binary Working Directory  = " << RA::OS::BWD() << endl;
    cout << "users home dir            = " << RA::OS::Home() << endl;

    cout << OS.RunConsoleCommand("echo testing popen").Read() << endl;

    cout << OS("echo testing popen SHORTHAND") << endl;

    RA::OS::MKDIR("./os_test");

    OS.Open("./os_test/tmp.txt", 'w').Write("hello world\n");
    auto& file = OS.File;

#if defined(NIX_BASE)
    cout << "1st time write >> " << OS.RunConsoleCommand("cat ./os_test/tmp.txt").Read() << endl;
    OS.cmd.GetOutput().Print();
#elif defined(WIN_BASE)
    cout << "1st time write >> " << OS.RunConsoleCommand("TYPE .\\os_test\\tmp.txt").Read() << endl;
    OS.CMD.GetOutput().Print();
#endif

    file.Clear();
    try {
        cout << "after clear() >> "; 
        cout << OS.Open("./os_test/tmp.txt").Read() << endl;
    }
    catch (std::runtime_error&) {
        cout << "File was deleted\n"; 
        exit(1);
    }
    cout << '\n';
    OS.Open("./os_test/tmp.txt", 'w').Write("hello world\n");

    file.Close(); // same as OS.Close();
    file.Move("./os_test/tmp2.txt");
    try {
        cout << "MoveFile() print from >> " << OS.Open("./os_test/tmp.txt").Read();
    }
    catch (std::runtime_error&) {
        cout << ": file does not exist!\n";
    }

     cout << "MoveFile() print to   >> " << OS.Open("./os_test/tmp2.txt").Read() << endl;

     file.Remove();
    try {
        cout << "Remove() >> "; 
        cout << OS.Open(file.GetName()).Read();
    }
    catch (std::runtime_error&) {
        cout << ": file does not exist!\n";
    }
    try {
        file.Remove();
    }
    catch (std::runtime_error& err) {
        cout << err.what() << endl;
    }

     OS.Open("./os_test/tmp.txt", 'w').Write("hello world\n");
     file.Copy("./os_test/tmp2.txt");
     cout << "CopyFile() print from >> " << OS.Open("./os_test/tmp.txt").Read();
     cout << "CopyFile() print to   >> " << OS.Open("./os_test/tmp2.txt").Read() << endl;

     OS.MKDIR("./os_test/os_test2/os_test3");
     xvector<xstring> paths = OS.Dir("./os_test", 'r', 'd', 'f');
     // Tthe order of the followinig does not matter >> "recursive", "files", "directories"

     cout << "-----------------\n";
     paths.Join('\n').Print();
     cout << "-----------------\n";

     file.Close();
     OS.MoveDir("./os_test", "./test");


     // note to be more cross platform you could have used "OS.dir()"
     // I did the following for demonstrative purposes
 #if defined (NIX_BASE)
     cout << OS("tree ./test") << endl;
     OS.file.Close();
     OS.RemoveDir("./test");
     cout << "\n\n";
     cout << OS("tree ./test") << endl;
 #elif defined (WIN_BASE)
     cout << OS("tree /F .\\test") << endl;
     OS.File.Close();
     OS.RemoveDir("./test");
     cout << "\n\n"; 
     cout << OS("tree /F .\\test") << endl;
 #endif
     cout << "\n\n";
     RescueThrow();
}
