#pragma once


#include "sys.h"
#include "ord.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

void ex_sys() {
	// based on the folowing arguments you pass
	//  -key-A sub-A-1 sub-A-2 sub-A-3 -key-B sub-B-1 sub-B-2

	//  -key-A 
	//		sub-A-1 
	//		sub-A-2 
	//		sub-A-3 
	// -key-B 
	//		sub-B-1 
	//		sub-B-2 


	cout << ord::join(sys.argv(), " ") << endl;

	// and see the count via
	cout << "argv count = " << sys.argc() << endl;
	cout << "program start the count at 1, then add 1 for every arg.\n";

	// if we are using a small script use the [] operator
	cout << "sys[5] = " << sys[5] << endl;

	// that is all good for small scripts but is bad for any tools of any size
	// nobody wants to remember what arg goes before or after another.
	// that is why we use maps. (like python dictionaries or ruby hashes)


	cout << "keys   = " << ord::join(sys.keys(), " ") << endl;
	//cout << "values = " << ord::join(sys.key_values(), " ") << endl;

	cout << "os values for key '-key-B' = " << ord::join(sys.key_values("-key-B")) << endl;


	cout << "Use operator() for booleans\n";
	cout << "Use operator[] for returning vectors\n";

	int count = 0; // This is used to make sure we hit all the correct targets and cmp at the end
	int Target = 6;

	if (sys.has_key("-key-B") && sys.has("-key-B") && sys("-key-B")) {
		cout << "-key-B exists\n"; ++count;
	}else {
		cout << "-key-B does NOT exists\n";
	}

	if (sys.has_key("-key-C") || sys.has("-key-C") || sys("-key-C")) {
		cout << "-key-C exists\n";
	}else {
		cout << "-key-C does NOT exist\n";  ++count;
	}

	if (sys.has_key_value("-key-A", "sub-B-2") || sys.key_value("-key-A", "sub-B-2") || sys("-key-A", "sub-B-2")) {
		cout << "value sub-B-2 exist in -key-A\n";
	}else {
		cout << "value sub-B-2 does NOT exist in -key-A\n"; ++count;
	}


	if (sys.has_key_value("-key-B", "sub-B-2") && sys.key_value("-key-B", "sub-B-2") && sys("-key-B", "sub-B-2")) {
		cout << "value sub-B-2 exist in -key-B\n"; ++count;
	}else {
		cout << "value sub-B-2 does NOT exist in -key-B\n";
	}
	 
	if (sys.has_key_value("-key-B", "sub-B-3") || sys.key_value("-key-B", "sub-B-3") || sys("-key-B", "sub-B-3")) {
		cout << "value sub-B-3 exist in -key-B\n";
	}else {
		cout << "value sub-B-3 does NOT exist in -key-B\n"; ++count;
	}


	if (sys.key_value("-key-A", 2) == "sub-A-2" && sys["-key-A"][2] == "sub-A-2") {
		cout << "-key-A's 2nd element is sub-A-2\n";
		// note: the [0] would be the key but returns nothing sense you already have the key
		// [1] is the first value, [2] is the second value and so on.
		++count;
	}
	else {
		cout << "-key-A's 2nd element is NOT sub-A-2\n";
	}


	cout << endl;
	cout << ">>>>>>>>>> " << count << endl;
	cout << ">>>>>>>>>> " << Target << endl;
	
}