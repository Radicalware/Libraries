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

    cout << os.popen("echo testing popen").read() << endl;

    cout << os("echo testing popen SHORTHAND") << endl;

    os.MKDIR("./os_test");

    os.open("./os_test/tmp.txt", 'w').write("hello world\n");
    auto& file = os.file;

#if defined(NIX_BASE)
    cout << "1st time write >> " << os.popen("cat ./os_test/tmp.txt").read() << endl;
    os.cmd.out().print();
#elif defined(WIN_BASE)
    cout << "1st time write >> " << os.popen("TYPE .\\os_test\\tmp.txt").read() << endl;
    os.cmd.out().print();
#endif

    file.clear();
    try {
        cout << "after clear() >> "; 
        cout << os.open("./os_test/tmp.txt").read() << endl;
    }
    catch (std::runtime_error&) {
        cout << "File was deleted\n"; 
        exit(1);
    }
    cout << '\n';
    os.open("./os_test/tmp.txt", 'w').write("hello world\n");

    file.close(); // same as os.close();
    file.move("./os_test/tmp2.txt");
    try {
        cout << "move_file() print from >> " << os.open("./os_test/tmp.txt").read();
    }
    catch (std::runtime_error&) {
        cout << ": file does not exist!\n";
    }

     cout << "move_file() print to   >> " << os.open("./os_test/tmp2.txt").read() << endl;

     file.remove();
    try {
        cout << "remove() >> "; 
        cout << os.open(file.name()).read();
    }
    catch (std::runtime_error&) {
        cout << ": file does not exist!\n";
    }
    try {
        file.remove();
    }
    catch (std::runtime_error& err) {
        cout << err.what() << endl;
    }

     os.open("./os_test/tmp.txt", 'w').write("hello world\n");
     file.copy("./os_test/tmp2.txt");
     cout << "copy_file() print from >> " << os.open("./os_test/tmp.txt").read();
     cout << "copy_file() print to   >> " << os.open("./os_test/tmp2.txt").read() << endl;

     os.MKDIR("./os_test/os_test2/os_test3");
     xvector<xstring> paths = os.Dir("./os_test", 'r', 'd', 'f');
     // Tthe order of the followinig does not matter >> "recursive", "files", "directories"

     cout << "-----------------\n";
     paths.join('\n').print();
     cout << "-----------------\n";

     file.close();
     os.Move_Dir("./os_test", "./test");


     // note to be more cross platform you could have used "os.dir()"
     // I did the following for demonstrative purposes
 #if defined (NIX_BASE)
     cout << os("tree ./test") << endl;
     os.file.close();
     os.Remove_Dir("./test");
     cout << "\n\n";
     cout << os("tree ./test") << endl;
 #elif defined (WIN_BASE)
     cout << os("tree /F .\\test") << endl;
     os.file.close();
     os.Remove_Dir("./test");
     cout << "\n\n"; 
     cout << os("tree /F .\\test") << endl;
 #endif
     cout << "\n\n";

}
