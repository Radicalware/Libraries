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
	//cout << "values = " << ord::join(os.key_values(), " ") << endl;
	
	cout << "os values for key '-key-B' = " << ord::join(os.key_values("-key-B")) << endl;
	
	if(os.has_key("-key-B")){
		cout << "-key-B exists\n";
	}else{
		cout << "-key-B does NOT exists\n";
	}

	if(os.has_key("-key-C")){
		cout << "-key-C exists\n";
	}else{
		cout << "-key-C does NOT exist\n";
	}

	if(os.key_value("-key-A","sub-B-2")){
		cout << "value sub-B-2 exist in -key-A\n";
	}else{
		cout << "value sub-B-2 does NOT exist in -key-A\n";
	}


	if(os.has_key_value("-key-B","sub-B-2")){ 
		// note key_value function could get confusing because one is
		// bool that references the string you give it and the other is
		// string that returns the integer locatio you give it.
		// to remove ambiguity, just use has_key_value so you know you are returning bool
		cout << "value sub-B-2 exist in -key-B\n";
	}else{
		cout << "value sub-B-2 does NOT exist in -key-B\n";
	}

	cout << "'-key-A's second value = " <<os.key_value("-key-A", 1) << endl; 

	if(os.key_value("-key-B","sub-B-3")){ // does -key-B have the value sub-B-3
		cout << "value sub-B-3 exist in -key-B\n";
	}else{
		cout << "value sub-B-3 does NOT exist in -key-B\n";
	}

	cout << endl;

	// here is the shorthand for
	// 1. returning  key-values
	// 2. validating key-values
	cout << "Shorthand for returning and validating KVPs\n";
	cout << R"(os["-key-A"] = )" << ord::join(os["-key-A"], " ") << endl;
	cout << R"(os("-key-A", "sub-A-1") =  )" << os("-key-A", "sub-A-1") << endl;
	cout << R"(os("-key-A", "sub-A-5") =  )" << os("-key-A", "sub-A-5") << endl;

	return 0;
}
