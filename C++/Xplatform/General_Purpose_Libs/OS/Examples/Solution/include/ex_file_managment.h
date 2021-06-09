#pragma once

// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
using std::cout;
using std::endl;

#include "xvector.h"
#include "OS.h"
extern OS os;

void ex_file_managment() 
{
    // all static OS functions start with an Upper_Case

    cout << "Present Working Directory = " << OS::PWD() << endl;
    cout << "Binary Working Directory  = " << os.BWD() << endl;
    cout << "users home dir            = " << os.Home() << endl;

    cout << os.RunConsoleCommand("echo testing popen").Read() << endl;

    cout << os("echo testing popen SHORTHAND") << endl;

    OS::MKDIR("./os_test");

    os.Open("./os_test/tmp.txt", 'w').Write("hello world\n");
    auto& file = os.file;

#if defined(NIX_BASE)
    cout << "1st time write >> " << os.RunConsoleCommand("cat ./os_test/tmp.txt").Read() << endl;
    os.cmd.GetOutput().Print();
#elif defined(WIN_BASE)
    cout << "1st time write >> " << os.RunConsoleCommand("TYPE .\\os_test\\tmp.txt").Read() << endl;
    os.cmd.GetOutput().Print();
#endif

    file.Clear();
    try {
        cout << "after clear() >> "; 
        cout << os.Open("./os_test/tmp.txt").Read() << endl;
    }
    catch (std::runtime_error&) {
        cout << "File was deleted\n"; 
        exit(1);
    }
    cout << '\n';
    os.Open("./os_test/tmp.txt", 'w').Write("hello world\n");

    file.Close(); // same as os.Close();
    file.Move("./os_test/tmp2.txt");
    try {
        cout << "MoveFile() print from >> " << os.Open("./os_test/tmp.txt").Read();
    }
    catch (std::runtime_error&) {
        cout << ": file does not exist!\n";
    }

     cout << "MoveFile() print to   >> " << os.Open("./os_test/tmp2.txt").Read() << endl;

     file.Remove();
    try {
        cout << "Remove() >> "; 
        cout << os.Open(file.GetName()).Read();
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

     os.Open("./os_test/tmp.txt", 'w').Write("hello world\n");
     file.Copy("./os_test/tmp2.txt");
     cout << "CopyFile() print from >> " << os.Open("./os_test/tmp.txt").Read();
     cout << "CopyFile() print to   >> " << os.Open("./os_test/tmp2.txt").Read() << endl;

     os.MKDIR("./os_test/os_test2/os_test3");
     xvector<xstring> paths = os.Dir("./os_test", 'r', 'd', 'f');
     // Tthe order of the followinig does not matter >> "recursive", "files", "directories"

     cout << "-----------------\n";
     paths.Join('\n').Print();
     cout << "-----------------\n";

     file.Close();
     os.MoveDir("./os_test", "./test");


     // note to be more cross platform you could have used "os.dir()"
     // I did the following for demonstrative purposes
 #if defined (NIX_BASE)
     cout << os("tree ./test") << endl;
     os.file.Close();
     os.RemoveDir("./test");
     cout << "\n\n";
     cout << os("tree ./test") << endl;
 #elif defined (WIN_BASE)
     cout << os("tree /F .\\test") << endl;
     os.file.Close();
     os.RemoveDir("./test");
     cout << "\n\n"; 
     cout << os("tree /F .\\test") << endl;
 #endif
     cout << "\n\n";

}
