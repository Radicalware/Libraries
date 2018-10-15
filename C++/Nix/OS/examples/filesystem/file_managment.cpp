#include <iostream>

#include "OS.h"
#include "re.h"

void replenish(){
	os.open("./ex_open/tmp.txt",'w').write("hello world\n");
}

int main(){

	// clear_file()
	// delete_file()
	// move_file()
	// copy_file()

	// popen()/read()
	//  open()/read()

	// rmdir()
	
	os.mkdir("./ex_open");

	replenish();
	cout << "1st time write >> " << os.popen("cat ./ex_open/tmp.txt").read() << endl;

	os.clear_file("./ex_open/tmp.txt");
	cout << "clear_file() >> " << os.open("./ex_open/tmp.txt").read() << endl;
	cout << '\n';
	replenish();


	os.move_file("./ex_open/tmp.txt","./ex_open/tmp2.txt");
	cout << "move_file() print from >> " << os.open("./ex_open/tmp.txt").read() << endl;
	cout << "move_file() print to   >> " << os.open("./ex_open/tmp2.txt").read()  << endl;


	os.delete_file("./ex_open/tmp2.txt");
	cout << "remove() >> " << os.open("cat ./ex_open/tmp2.txt").read() << endl;
	os.delete_file("./ex_open/tmp2.txt"); // doesn't crash
	
	replenish();
	os.copy_file("./ex_open/tmp.txt","./ex_open/tmp2.txt"); // doesn't crash

	// using open (not popen) below
	cout << "copy_file() print from >> " << os.open("./ex_open/tmp.txt").read();
	cout << "copy_file() print to   >> " << os.open("./ex_open/tmp2.txt").read()  << endl;


	string path = "./ex_open/ex_open2/ex_open3";

	os.mkdir(path);
	vector<string> paths = os.dir("./ex_open", "recursive", "files", "folders");
	// Tthe order of the followinig does not matter >> "recursive", "files", "folders"

	cout << "-------------\n";
	for(string& i : paths)
		cout << i << endl;
	cout << "-------------\n";


	cout << "move_file() print from >> " << os.open("./ex_open/tmp.txt").read() << endl;
	cout << "move_file() print to   >> " << os.open("./ex_open/tmp2.txt").read()  << endl;

	os.rmdir("./ex_open");
	cout << '\n';
	return 0;
}