#include <iostream>
#include <string>
#include <string.h>
#include <cstdint>

#include "Hexr.h" // I don't use hpp because it is doesn't work well with visual studio.

// Programed by Scourge
// Licenced  by Apache v2

// g++ ./fuz-hexr.cpp Hexr.h Hexr.cpp -Wfatal-errors -o ex && ./ex
// valgrind --leak-check=full ./ex

using std::cout;
using std::endl;
using std::string;

void print_array(Hexr &instance_hexr, string type){
	// used string type to keep it consistent with Hexr usage
	for(intmax_t i = 0; i < instance_hexr.byte_count(); i++){
		if      (type == "ca"){
			cout << instance_hexr.hex_string()[i] << " | ";
		}else if(type == "ia"){
			cout << instance_hexr.int_array()[i] << " | ";
		}else if(type == "ha"){
			cout << instance_hexr.hex_array()[i] << " | ";
		}
	}
	cout << '\n';
}

void print_vector(Hexr &instance_hexr, string type){
	// used string type to keep it consistent with Hexr usage
	for(intmax_t i = 0; i < instance_hexr.byte_count(); i++){
		if      (type == "ca"){
			cout << instance_hexr.char_vector()[i] << " | ";
		}else if(type == "ia"){
			cout << instance_hexr.int_vector()[i] << " | ";
		}else if(type == "ha"){
			cout << instance_hexr.hex_vector()[i] << " | ";
		}
	}
	cout << '\n';
}

int main(){
	cout << "-------------------------------------------------------------------------------\n";

	//  ascii_string > hex_string > int_array > hex_array > hex_string

	string ascii_string = "Hello World!";
	char   char_array  [] {'H','e','l','l','o',' ','W','o','r','l','d','!'};
	int    int_array   [] {72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33};
	string hex_array   [] {"48","65","6c","6c","6f","20","57","6f","72","6c","64","21"};
	string hex_string   = "0x48656c6c6f20576f726c6421"; // The '0x' is optional

	vector<char>   char_vector {'H','e','l','l','o',' ','W','o','r','l','d','!'};
	vector<int>    int_vector  {72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33};
	vector<string> hex_vector  {"48","65","6c","6c","6f","20","57","6f","72","6c","64","21"};


	// Thanks for taking the time to view this help, I am sure Hexr will save a lot of time
	// The purpos of Hexr is to convert the above C++ types to any other C++ type with ease
	// or just simply print it out.

	// note: input and output for vector's work the same way. A vector example is below.

	// ascii_string > char_array > int_array > hex_array > hex_string
	//     as            ca          ia           ha          hs

	// order of operation for Path1
	// as = ascii_string -
	// ca =  char_array <
	// ia =   int_array <
	// ha =   hex_array <
	// hs =   hex_string -

	// Here we make an anonymous class (class with no long-term instance) to convert
	// from an ascii string to a hex string

	cout << "Good 1.)\n";
	string stored_hex_string = Hexr(ascii_string,"hex_string").hex_string();
	cout << stored_hex_string << endl;

	// That is the long and clear way of doing the following

	cout << Hexr(ascii_string,"hs").s() << '\n';

	// which uses the "hs" or "hex_string" alias and uses the ret type's return value
	// sense we asked for a "hs" it is stored in the .s() getter so we don't need to
	// type out the long .hex_string();


	// We make a new class instance using a hex string, we will ask to get the int array.
	// This can often be useful. For example, we want to see if we are looking at a text
	// document which should only have ascii chars (chars not above 127), we may do 
	// something like the following.

	cout << "\nGood 2.)\n";
	Hexr hex_string_to_int_array(hex_string,"ia");
	print_array(hex_string_to_int_array,"ia");

	// Now we use print_array (function viewable above) to iterate through the array
	// and print out it's elements. Remember to pass your classes as a reference to 
	// improve speed. Sense we use "ia" that means we are only guaranteed to get
	// an "int_array" so if you want an instance with all getters drop the "ia"
	// and do the following. 

	cout << "\nGood 3.)\n";
	Hexr hex_string_to_all(hex_string); // create class instance
	cout << hex_string_to_all.ascii_string() << endl;
	print_array(hex_string_to_all, "ca");
	print_array(hex_string_to_all, "ia");
	print_array(hex_string_to_all, "ha");
	cout << hex_string_to_all.hex_string() << endl;

	// You can also use vectors as input and output, here I will do both at the same time
	// Anytime you ask for an array as output "(i/c/h) array" You will also have access
	// to the vector automatically. 
	// Input for a vector works the same way as an array or any of the other C++ types.

	cout << "\nGood 4.)\n";
	Hexr hexr_int_vector(hex_vector,"ia");
	print_vector(hexr_int_vector,"ia");

	// for added speed you can manually add the size of the array
	// Also, this method is 100% accurate. The dynamic way may have junk after your array if type char*
	cout << "\nGood 5.)\n";
	cout << '"' << "\\x" << Hexr(char_array, sizeof(char_array)/sizeof(char_array)[0], "ha", "\\x").s() << '"';

	// an int can be inserted into arg 2 which will move the ret type to arg 3 and the injection
	// spacer for arrays into arg 4.

	// Now there are 2 things that you will NOT want to do
	// first, if you ask for an array, you can't print it by itself without asking for it's string type
	// in the last one, we asked for the string conversion by using the .s(), but if we used .hex_array()
	// we would just get the address

	cout << "\n\nFail 1.)\n";
	cout << Hexr(char_array, "ha").hex_array() << endl;

	// Next, for spead reasons, if you are using Hexr as an anonymous class, be sure to add the ret type,
	// otherwise you will go through every conversion for no reason which is not as fast.

	cout << "\nFail 2.)\n";
	cout << Hexr(char_array).ascii_string() << endl;

	cout << "\n\n-------------------------------------------------------------------------------\n";

	return 0;
} 
