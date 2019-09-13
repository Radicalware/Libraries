#include<iostream>
#include<vector>
#include<string>
#include<ctime>


// ----- Radicalware Libs -------------------
// ----- eXtended STL Functionality ---------
#include "xstring.h"
#include "xvector.h"
#include "xmap.h"
// ----- eXtended STL Functionality ---------
// ----- Radicalware Libs -------------------

#include "SYS.h"   // Found on "https://github.com/Radicalware"

SYS sys;

using std::cout;
using std::endl;
using std::string;
using std::vector;



int main(int argc, char** argv) {


	sys.alias('a', "--key-A");
	sys.alias('b', "--key-B");
	sys.alias('f', "--key-F");
	sys.alias('x', "--key-X");

	sys.alias('p', "--port");

	sys.set_args(argc, argv);

	// based on the folowing arguments you pass
	//  --key-A sub-A-1 sub-A-2 sub-A-3 --key-X --key-B sub-B-1 sub-B-2 --key-C sub-C-1 sub-C-2 --key-Y --key-Z -n -wxyp 8080  9090 -ze -f 

	//  --key-A 
	//		sub-A-1 
	//		sub-A-2 
	//		sub-A-3 
	// --key-X
	// --key-B 
	//		sub-B-1 
	//		sub-B-2 
	// --key-C 
	//		sub-C-1 
	//		sub-C-2 
	// --key-Y
	// --key-Z
	// -a
	// -n
	// -wxyp 8080 9090
	// -ze
	// -f

	// notice, --key-A and 'a' are an alias of each other
	// they were both used which in most cases is not intended
	// this is to show you that even if the user does that, the program still works fine.
	
	cout << "\n\n all args = " << sys.argv()(1).join(' ');
	// we skipped the first arg (which is the program) using a slice

	//// and see the count via
	cout << "\n\n argv count = " << sys.argc();
	cout << "\n true argc  = " << argc;
	cout << "\n program start the count at 1, then add 1 for every arg.";

	// if we are using a small script use the [] operator
	cout << "\n\n sys[5]      = " << sys[5];

	cout << "\n sys values for key '--key-B' = " << sys["--key-B"].join(' ');

	cout << "\n full path   = " << sys.full_path();
	cout << "\n path        = " << sys.path();
	cout << "\n file        = " << sys.file() << "\n\n\n";

	// ----------------------------------------------------------------
	if ( sys.has("--key-B") && sys("--key-B")) {  // both methods are the same
		cout << "--key-B exists\n"; 
	}
	else {
		cout << "error: --key-B does not exists\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys.has("--key-D") || sys("--key-D")) { // both methods are the same
		cout << "error: --key-D exists\n"; exit(1);
	}
	else {
		cout << "--key-D does not exist\n";  
	}
	// ----------------------------------------------------------------
	if (sys('n') && sys('x')) {
		cout << "-n and -x options are both used\n";
	}
	else {
		cout << "error: -n and -x are not used together or not at all \n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys('x') && sys("--key-X")) { 
		// note: both were given (this is ok because thare are no associated values with either x or --key-X)
		cout << "-x and the alias --key-X are both used\n";
	}
	else {
		cout << "error: -x and the alias --key-X are not found together\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys('f') && sys("--key-F")) {
		cout << "-f and the alias --key-F are both usable but only one was given\n";
	}
	else {
		cout << "error: -f and the alias --key-F are not found together\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys('b') && sys("--key-B")) { 
		cout << "-b and the alias --key-B are both usable but only --key-B was given\n";
	}
	else {
		cout << "error: -b and the alias --key-B are not found together\n"; exit(1);
	}
	// ----------------------------------------------------------------
	bool pass = false;
	if (sys('p') && sys("--port")){
		cout << "we have the port key\n";
		if (*sys['p'][0] == "8080" && *sys["--port"][0] == "8080") {
			cout << "port designated by -p is on " << sys['p'].join(' ') << endl;
			pass = true;
		}
	}
	if(!pass){
		cout << "error: -p is not set" << endl; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys.key("--key-A").has("sub-B-2") || sys("--key-A", "sub-B-2")) {
		cout << "error: value sub-B-2 exist in --key-A\n"; exit(1);
	}
	else {
		cout << "value sub-B-2 does not exist in --key-A\n";
	}
	// ----------------------------------------------------------------
	if (sys.key("--key-A").has("sub-A-2") && sys("--key-A", "sub-A-3") ) {
		cout << "value sub-A-2 and sub-A-3 exist under --key-A\n";
	}
	else {
		cout << "error: value sub-A-2 and sub-A-3 not found under --key-A\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (*sys["--key-A"][1] == "sub-A-2" && *sys['p'][1] == "9090") {
		cout << "the 2nd elements for targed str and char keys passed\n";
	}
	else {
		cout << "error: str or char keys' value not found\n"; exit(1);
	}
	// ----------------------------------------------------------------


	// here is the shorthand for
	// 1. returning  key-values
	// 2. validating key-values
	cout << "\n\n";
	cout << R"(sys["--key-A"].join(' ')  ==  )" << sys["--key-A"].join(' ') << endl;
	cout << R"(sys("--key-A", "sub-A-1") ==  )" << sys("--key-A", "sub-A-1") << endl;
	cout << R"(sys("--key-A", "sub-A-5") ==  )" << sys("--key-A", "sub-A-5") << "\n\n";


	cout << R"(*sys["--key-A"][0] == )" << *sys["--key-A"][0] << endl;
	cout << R"(*sys["--key-A"][1] == )" << *sys["--key-A"][1] << endl;
	cout << R"(*sys["--key-A"][2] == )" << *sys["--key-A"][2] << "\n\n";

	cout << R"(*sys["--key-B"][0] == )" << *sys["--key-B"][0] << endl;
	cout << R"(*sys["--key-B"][1] == )" << *sys["--key-B"][1] << endl;

	return 0;
}
