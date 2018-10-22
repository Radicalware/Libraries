#include <iostream>

#include "OS.h"
#include "re.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;


void replenish(){
	os.open("./fm_open/tmp.txt",'w').write("hello world\n");
}

int main(){

	// clear_file()
	// delete_file()
	// move_file()
	// copy_file()

	// popen()/read()
	//  open()/read()

	// rmdir()

	cout << "pwd = " << os.pwd() << endl;

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
	vector<string> paths = os.dir("./fm_open", "recursive", "files", "folders");
	// Tthe order of the followinig does not matter >> "recursive", "files", "folders"

	cout << "-------------\n";
	for (string& i : paths)
		cout << i << endl;
	cout << "-------------\n";

	os.move_dir("./fm_open", "./test");
	cout << os("tree ./test") << endl;

	os.rmdir("./test");
	cout << '\n';

	return 0;
}
