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

	sys.set_args(argc, argv, true);
	sys.alias('p', "--port");
	// 3rd arg "true", means C-style (default is basic style)
	// link should be used when C-style is specified

	// C-Style example1:      netstat -antp 8080
	// C-Style example2:      aircrack-ng --bssid AA:AA:AA:AA:AA:AA
	// basic-style example3:  program -key1 value -key2 -key3 value1 value2

	// C-Style
	// strings args start with --
	// char or char array starts with -
	// the last char of a char array may be KVP (key value paired) -p 8080 or -antp 8080

	// basic style
	// strings start with either '-' or '--'
	// there are no char arrays so -antp would need to be -a -n -t -p


	// based on the folowing arguments you pass
	//  --key-A sub-A-1 sub-A-2 sub-A-3 --key-B sub-B-1 sub-B-2 -a -bcdp 8080  9090 -ef -g 

	//  --key-A 
	//		sub-A-1 
	//		sub-A-2 
	//		sub-A-3 
	// --key-B 
	//		sub-B-1 
	//		sub-B-2 
	// -a
	// -bcdp 8080 9090
	// -ef
	// -g

	
	cout << "\n\n all args = " << sys.argv()(1).join(' ');
	// we skipped the first arg (which is the program) using a slice

	//// and see the count via
	cout << "\n\n argv count = " << sys.argc();
	cout << "\n true argc  = " << argc;
	cout << "\n program start the count at 1, then add 1 for every arg.";

	// if we are using a small script use the [] operator
	cout << "\n\n sys[5]      = " << sys[5];

	cout << "\n keys        = " << sys.keys().join(' ');
	cout << "\n key count   = " << sys.keys().size();

	cout << "\n os values for key '--key-B' = " << sys.key_values("--key-B").join(' ');

	cout << "\n full path   = " << sys.full_path();
	cout << "\n path        = " << sys.path();
	cout << "\n file        = " << sys.file() << "\n\n\n";

	// ----------------------------------------------------------------
	if (   // all 5 give the same result and mean the same thing
		   sys("--key-B")
		&& sys.kvps().has("--key-B")
		&& sys.has_key("--key-B")
		&& sys.has("--key-B")
		&& sys.keys().has("--key-B")
	) {
		cout << "--key-B exists\n"; 
	}
	else {
		cout << "error: --key-B does not exists\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys.has("--key-B") && sys.kvps().has("--key-B")
		&& sys.has_key("--key-c") || sys("--key-c")) 
	{
		cout << "error: --key-c exists\n"; exit(1);
	}
	else {
		cout << "--key-c does not exist\n";  
	}
	//// ----------------------------------------------------------------
	if (sys('a') && sys("-c")) { 
		cout << "-a and -c options are both used\n"; 
	} else {
		cout << "error: -a and -c are not used together or not at all \n"; exit(1);
	}
	//// ----------------------------------------------------------------
	bool pass = false;
	if (sys('p') && sys("--port")){
		if (*sys['p'][0] == "8080" && *sys["--port"][0] == "8080") {
			cout << "port designated by -p is on " << sys['p'].join(' ') << endl;
			pass = true;
		}
	}
	if(!pass){
		cout << sys.keys().join(' ') << endl;
		cout << "error: -p is not set" << endl; exit(1);
	}
	//// ----------------------------------------------------------------
	if (sys.bool_arg('a') && !sys.kvp_arg('a') ) {  
		cout << "-a is used as a boolean arg so it does not have values\n"; 
	} else {exit(1);
		cout << "error: -a is kvp type and so -h has values\n"; exit(1);
	}
	//// ----------------------------------------------------------------
	if (sys.kvp_arg("--key-A") && !sys.bool_arg("--key-A")) {
		cout << "--key-A is kvp type and so --key-A has values\n"; 
	} else {
		cout << "error: --key-A does not use kvp and so --key-A does not have values\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (
		sys.has_key_value("--key-A", "sub-B-2") 
		|| sys("--key-A", "sub-B-2")) {
		cout << "error: value sub-B-2 exist in --key-A\n"; exit(1);
	}
	else {
		cout << "value sub-B-2 does not exist in --key-A\n"; 
	}
	// ----------------------------------------------------------------
	if (
		sys.has_key_value("--key-B", "sub-B-2") ){//&& sys("--key-B", "sub-B-2")) {
		cout << "value sub-B-2 exist in --key-B\n"; 
	}
	else {
		cout << "error: value sub-B-2 does not exist in --key-B\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys.has_key_value("--key-B", "sub-B-3") || sys("--key-B", "sub-B-3")) {
		cout << "error: value sub-B-3 exist in --key-B\n"; exit(1);
	}
	else {
		cout << "value sub-B-3 does not exist in --key-B\n"; 
	}
	// ----------------------------------------------------------------
	if (sys.key_value("--key-A", 1) == "sub-A-2" && *sys["--key-A"][1] == "sub-A-2" && sys.second("--key-A") == "sub-A-2") {
		cout << "--key-A's 2nd element is sub-A-2\n";
	}
	else {
		cout << "error: --key-A's 2nd element is not sub-A-2\n"; exit(1);
	}
	// ----------------------------------------------------------------


	//// here is the shorthand for
	//// 1. returning  key-values
	//// 2. validating key-values
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
