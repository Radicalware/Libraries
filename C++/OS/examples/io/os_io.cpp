#include<iostream>
#include<vector>
#include<string>
#include<ctime>

#include "OS.h"   // Found on "https://github.com/Radicalware"
#include "ord.h"  // Found on "https://github.com/Radicalware"
using std::cout;
using std::endl;
using std::string;
using std::vector;



int main(int argc, char** argv){

	os.set_args(argc, argv);

	// return all the args via
	cout << ord::join(os.argv(), " ") << endl;

	// and see the count via
	cout << "argv count = " << os.argc() << endl;
	cout << "program start the count at 1, then add 1 for every arg.\n";

	// if we are using a small script use the [] operator
	cout << "os[5] = " << os[5] << endl;

	// that is all good for small scripts but is bad for any tools of any size
	// nobody wants to remember what arg goes before or after another.
	// that is why we use maps. (like python dictionaries or ruby hashes)


	cout << "keys   = " << ord::join(os.keys(), " ") << endl;
	//cout << "values = " << ord::join(os.keyValues(), " ") << endl;
	
	cout << "os values for key '-key-B' = " << ord::join(os.keyValues("-key-B")) << endl;
	
	if(os.findKey("-key-B")){
		cout << "-key-B exists\n";
	}else{
		cout << "-key-B does NOT exists\n";
	}

	if(os.findKey("-key-C")){
		cout << "-key-C exists\n";
	}else{
		cout << "-key-C does NOT exist\n";
	}

	if(os.findKeyValue("-key-A","sub-B-2")){
		cout << "value sub-B-2 exist in -key-A\n";
	}else{
		cout << "value sub-B-2 does NOT exist in -key-A\n";
	}


	if(os.findKeyValue("-key-B","sub-B-2")){
		cout << "value sub-B-2 exist in -key-B\n";
	}else{
		cout << "value sub-B-2 does NOT exist in -key-B\n";
	}

	if(os.findKeyValue("-key-B","sub-B-3")){
		cout << "value sub-B-3 exist in -key-B\n";
	}else{
		cout << "value sub-B-3 does NOT exist in -key-B\n";
	}

	return 0;
}
