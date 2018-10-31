#include<iostream>
#include<vector>
#include<string>
#include<ctime>

#include "SYS.h"   // Found on "https://github.com/Radicalware"
#include "ord.h"   // Found on "https://github.com/Radicalware"

using std::cout;
using std::endl;
using std::string;
using std::vector;



int main(int argc, char** argv) {

	sys.set_args(argc, argv, true); // true means C-style (default is basic style)

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

	// return all the args via
	cout << ord::join(sys.argv(), " ") << endl;

	// based on the folowing arguments you pass
	//  --key-A sub-A-1 sub-A-2 sub-A-3 --key-B sub-B-1 sub-B-2 -a -bcd -ef -g

	//  --key-A 
	//		sub-A-1 
	//		sub-A-2 
	//		sub-A-3 
	// --key-B 
	//		sub-B-1 
	//		sub-B-2 
	// -a
	// -bcdp 8080
	// -ef
	// -g

	cout << ord::join(sys.argv(), " ") << endl;

	// and see the count via
	cout << "argv count = " << sys.argc() << endl;
	cout << "true argc  = " << argc << endl;
	cout << "program start the count at 1, then add 1 for every arg.\n";

	// if we are using a small script use the [] operator
	cout << "sys[5] = " << sys[5] << endl;

	

	cout << "keys   = " << ord::join(sys.keys(), " ") << endl;

	cout << "os values for key '--key-B' = " << ord::join(sys.key_values("--key-B")) << endl;

	cout << "full path  = " << sys.full_path() << endl;
	cout << "path       = " << sys.path() << endl;
	cout << "file       = " << sys.file() << endl;

	// ----------------------------------------------------------------
	if (sys.has_key("--key-B") && sys.has("--key-B") && sys("--key-B")) {
		cout << "--key-B exists\n"; 
	}
	else {
		cout << "error: --key-B does NOT exists\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys.has_key("--key-C") || sys("--key-C")) {
		cout << "error: --key-C exists\n"; exit(1);
	}
	else {
		cout << "--key-C does NOT exist\n";  
	}
	// ----------------------------------------------------------------
	if (sys('a') && sys("-c")) { // always use operators if using chars or wanting regex()
		cout << "-a and -c options are both used\n"; 
	} else {
		cout << "error: -a and -c are not used together or not at all \n"; exit(1);
	}
	// ----------------------------------------------------------------
	bool pass = false;
	if (sys("-p")) {
		if (sys['p'][0] == "8080") { 
			cout << "port designated by -p is on" << ord::join(sys['p']) << endl;
			pass = true;
		}
	}
	if(!pass){
		cout << "error: -p is not set" << endl; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys.bool_arg("-h") && !sys.kvp("-h")) { // bool_arg and kvp are opposites 
		cout << "-h is used as a boolean arg so it does not have values\n"; 
	} else {exit(1);
		cout << "error: -h is KVP type and so -h has values\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys.kvp("--key-A") && !sys.bool_arg("--key-A")) {
		cout << "--key-A is KVP type and so --key-A has values\n"; 
	} else {
		cout << "error: --key-A does NOT use KVP and so --key-A does NOT have values\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys.has_key_value("--key-A", "sub-B-2") || sys("--key-A", "sub-B-2")) {
		cout << "error: value sub-B-2 exist in --key-A\n"; exit(1);
	}
	else {
		cout << "value sub-B-2 does NOT exist in --key-A\n"; 
	}
	// ----------------------------------------------------------------
	if (sys.has_key_value("--key-B", "sub-B-2") && sys("--key-B", "sub-B-2")) {
		cout << "value sub-B-2 exist in --key-B\n"; 
	}
	else {
		cout << "error: value sub-B-2 does NOT exist in --key-B\n"; exit(1);
	}
	// ----------------------------------------------------------------
	if (sys.has_key_value("--key-B", "sub-B-3") || sys("--key-B", "sub-B-3")) {
		cout << "error: value sub-B-3 exist in --key-B\n"; exit(1);
	}
	else {
		cout << "value sub-B-3 does NOT exist in --key-B\n"; 
	}
	// ----------------------------------------------------------------
	if (sys.key_value("--key-A", 1) == "sub-A-2" && sys["--key-A"][1] == "sub-A-2" && sys.second("--key-A") == "sub-A-2") {
		cout << "--key-A's 2nd element is sub-A-2\n";
	}
	else {
		cout << "error: --key-A's 2nd element is NOT sub-A-2\n"; exit(1);
	}
	// ----------------------------------------------------------------


	// here is the shorthand for
	// 1. returning  key-values
	// 2. validating key-values
	cout << "Shorthand for returning and validating KVPs\n";
	cout << R"(sys["--key-A"] = )" << ord::join(sys["--key-A"], " ") << endl;
	cout << R"(sys("--key-A", "sub-A-1") =  )" << sys("--key-A", "sub-A-1") << endl;
	cout << R"(sys("--key-A", "sub-A-5") =  )" << sys("--key-A", "sub-A-5") << endl;

	cout << "\n\n";

	cout << R"(sys["--key-A"][0] == )" << sys["--key-A"][0] << endl;
	cout << R"(sys["--key-A"][1] == )" << sys["--key-A"][1] << endl;
	cout << R"(sys["--key-A"][2] == )" << sys["--key-A"][2] << "\n\n";

	cout << R"(sys["--key-B"][0] == )" << sys["--key-B"][0] << endl;
	cout << R"(sys["--key-B"][1] == )" << sys["--key-B"][1] << endl;

	return 0;
}
