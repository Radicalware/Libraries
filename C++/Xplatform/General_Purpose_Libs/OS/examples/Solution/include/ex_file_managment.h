#pragma once

#include "OS.h"

#include<iostream>
#include "xvector.h"

using std::cout;
using std::endl;
using std::string;


extern OS os;

void replenish() {
	os.open("./fm_open/tmp.txt", 'w').write("hello world\n");
}

int ex_file_managment() {


    cout << "users pwd      = " << os.pwd() << endl;
    cout << "users home dir = " << os.home() << endl;
    cout << "binary's pwd   = " << os.bpwd() << endl;

	cout << os.popen("echo testing popen").read() << endl;

	cout << os("echo testing popen SHORTHAND") << endl;

	os.mkdir("./fm_open");

	replenish();
#if defined(NIX_BASE)
	cout << "1st time write >> " << os.popen("cat ./fm_open/tmp.txt").read() << endl;
#elif defined(WIN_BASE)
	cout << "1st time write >> " << os.popen("TYPE .\\fm_open\\tmp.txt").read() << endl;
#endif

	os.clear_file("./fm_open/tmp.txt");
	cout << "clear_file() >> " << os.open("./fm_open/tmp.txt").read() << endl;
	cout << '\n';
	replenish();


	os.move_file("./fm_open/tmp.txt", "./fm_open/tmp2.txt");
	cout << "move_file() print from >> " << os.open("./fm_open/tmp.txt").read() << endl;
	cout << "move_file() print to   >> " << os.open("./fm_open/tmp2.txt").read() << endl;


	os.delete_file("./fm_open/tmp2.txt");
	cout << "delete_file() >> " << os.open("./fm_open/tmp2.txt").read() << endl;
	os.delete_file("./fm_open/tmp2.txt");

	replenish();
	os.copy_file("./fm_open/tmp.txt", "./fm_open/tmp2.txt");
	cout << "copy_file() print from >> " << os.open("./fm_open/tmp.txt").read();
	cout << "copy_file() print to   >> " << os.open("./fm_open/tmp2.txt").read() << endl;

	os.mkdir("./fm_open/fm_open2/fm_open3");
	xvector<xstring> paths = os.dir("./fm_open", 'r', 'd', 'f');
	// Tthe order of the followinig does not matter >> "recursive", "files", "directories"

	cout << "-------------\n";
	for (string& i : paths)
		cout << i << endl;
	cout << "-------------\n";

	os.move_dir("./fm_open", "./test");


	// note to be more cross platform you could have used "os.dir()"
	// I did the following for demonstrative purposes
#if defined (NIX_BASE)
	cout << os("tree ./test") << endl;
	os.delete_dir("./test");
	cout << "\n\n";
	cout << os("tree ./test") << endl;
#elif defined (WIN_BASE)
	cout << os("tree /F .\\test") << endl;
	os.delete_dir("./test");
	cout << "\n\n"; 
	cout << os("tree /F .\\test") << endl;
#endif

	cout << "\n\n";


	return 0;
}
