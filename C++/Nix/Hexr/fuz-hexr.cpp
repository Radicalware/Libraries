// Hexr.cpp : main project file.


//#include "stdafx.h" // <<<<< FOR VISUAL STUDIO (not needed in HEXR)
//#include "stdint.h" // <<<<< FOR VISUAL STUDIO (not needed in HEXR)


#include <iostream>
#include <string>
#include <string.h>
#include <stdio.h>  // uint64_t 

#include "Hexr.h"

// Programed by Scourge
// Licenced  by Apache

using std::cout;
using std::endl;
using std::string;


// Finished: Char vector
// Make: Int vector & Hex Vector

// Make a new input param that will allow for an in input that will specify the size of the array
// allowing for a manual input will in some cases speed up the ouptut and will increase
// the chance of success (I am relying on very high odds but it is still chance.)


void print_array(Hexr &instance_hexr, string type) {
	// used string type to keep it consistent with Hexr usage
	for (uint64_t  i = 0; i < instance_hexr.byte_count(); i++) {
		if (type == "ca") {
			cout << instance_hexr.char_array()[i] << " * ";
		}
		else if (type == "ia") {
			cout << instance_hexr.int_array()[i] << " * ";
		}
		else if (type == "ha") {
			cout << instance_hexr.hex_array()[i] << " * ";
		}
	}
	cout << '\n';
}

void print_vector(Hexr &instance_hexr, string type) {
	// used string type to keep it consistent with Hexr usage
	for (uint64_t  i = 0; i < instance_hexr.byte_count(); i++) {
		if (type == "ca") {
			cout << instance_hexr.char_vector()[i] << " * ";
		}
		else if (type == "ia") {
			cout << instance_hexr.int_vector()[i] << " * ";
		}
		else if (type == "ha") {
			cout << instance_hexr.hex_vector()[i] << " * ";
		}
	}
	cout << '\n';
}

int main() {

	//  ascii_string > char_array > int_array > hex_array > hex_string
	//      as            ca          ia           ha          hs

	// order of operation for Path1
	// as = ascii_string -
	// ca =  char_array <
	// ia =   int_array <
	// ha =   hex_array <
	// hs =   hex_string -

	string ascii_string = "Hello World!";
	char   char_array[]{ 'H','e','l','l','o',' ','W','o','r','l','d','!' };
	int    int_array[]{ 72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33 };
	string hex_array[]{ "48","65","6c","6c","6f","20","57","6f","72","6c","64","21" };
	string hex_string = "0x48656c6c6f20576f726c6421"; // The '0x' is optional

	vector<char>   char_vector{ 'H','e','l','l','o',' ','W','o','r','l','d','!' };
	vector<int>    int_vector{ 72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33 };
	vector<string> hex_vector{ "48","65","6c","6c","6f","20","57","6f","72","6c","64","21" };

	cout << "-------------------------------------------------------------------------------\n";
	cout << "Convert ASCII_STRING\n";
	cout << "-------------------------------------------------------------------------------\n";
	cout << "\nPART 1: Using Seperators\n";
	cout << "$$$>	" << Hexr(ascii_string,     "as", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(ascii_string, 12, "as", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(ascii_string,     "ca", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(ascii_string, 12, "ca", " - ").s() << '\n'; 

 	cout << "$$$>	" << Hexr(ascii_string,     "ia", " - ").s() << '\n'; 
 	cout << "$$$>	" << Hexr(ascii_string, 12, "ia", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(ascii_string,     "ha", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(ascii_string, 12, "ha", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(ascii_string,     "hs", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(ascii_string, 12, "hs", " - ").s() << '\n'; 

	cout << "\nPART 2: Without seperators\n";
	cout << "$$$>	" << Hexr(ascii_string,     "as").s() << '\n'; 
	cout << "$$$>	" << Hexr(ascii_string, 12, "as").s() << '\n'; 

	cout << "$$$>	" << Hexr(ascii_string,     "ca").s() << '\n'; 
	cout << "$$$>	" << Hexr(ascii_string, 12, "ca").s() << '\n'; 

	cout << "$$$>	" << Hexr(ascii_string,     "ia").s() << '\n'; 
	cout << "$$$>	" << Hexr(ascii_string, 12, "ia").s() << '\n'; 

	cout << "$$$>	" << Hexr(ascii_string,     "ha").s() << '\n'; 
	cout << "$$$>	" << Hexr(ascii_string, 12, "ha").s() << '\n'; 

	cout << "$$$>	" << Hexr(ascii_string,     "hs").s() << '\n'; 
	cout << "$$$>	" << Hexr(ascii_string, 12, "hs").s() << '\n'; 


	cout << "\nPart3: Getter Based\n";
	Hexr as_hexr(ascii_string);
	cout << "$$$>	" << as_hexr.ascii_string() << '\n'; 
	print_array(as_hexr, "ca");
	print_array(as_hexr, "ia");
	print_array(as_hexr, "ha");
	cout << "$$$>	" << as_hexr.hex_string() << "\n\n";

	Hexr man_as_hexr(ascii_string, 12); 
	cout << "$$$>	" << man_as_hexr.ascii_string() << '\n'; 
	print_array(man_as_hexr, "ca");
	print_array(man_as_hexr, "ia");
	print_array(man_as_hexr, "ha");
	cout << "$$$>	" << man_as_hexr.hex_string() << '\n'; 


	cout << "\n-------------------------------------------------------------------------------\n";
	cout << "CONVERT CHAR ARRAY\n";
	cout << "-------------------------------------------------------------------------------\n";
	cout << "\nPART 1: Using Seperators\n";
	cout << "$$$>	" << Hexr(char_array,     "as", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_array, 12, "as", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_array,     "ca", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_array, 12, "ca", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_array,     "ia", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_array, 12, "ia", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_array,     "ha", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_array, 12, "ha", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_array,     "hs", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_array, 12, "hs", " - ").s() << '\n'; 

	cout << "\nPART 2: Without seperators\n";
	cout << "$$$>	" << Hexr(char_array,     "as").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_array, 12, "as").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_array,     "ca").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_array, 12, "ca").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_array,     "ia").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_array, 12, "ia").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_array,     "ha").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_array, 12, "ha").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_array,     "hs").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_array, 12, "hs").s() << '\n'; 


	cout << "\nPart3: Getter Based\n";
	Hexr ca_hexr(char_array);
	cout << "$$$>	" << ca_hexr.ascii_string() << '\n'; 
	print_array(ca_hexr, "ca");
	print_array(ca_hexr, "ia");
	print_array(ca_hexr, "ha");
	cout << "$$$>	" << ca_hexr.hex_string() << '\n'; 

	cout << "\n\n";
	Hexr man_ca_hexr(char_array, 12);
	cout << "$$$>	" << man_ca_hexr.ascii_string() << '\n'; 
	print_array(man_ca_hexr, "ca");
	print_array(man_ca_hexr, "ia");
	print_array(man_ca_hexr, "ha");
	cout << "$$$>	" << man_ca_hexr.hex_string() << '\n'; 




	cout << "\n-------------------------------------------------------------------------------\n";
	cout << "CONVERT CHAR VECTOR\n";
	cout << "-------------------------------------------------------------------------------\n";
	cout << "\nPART 1: Using Seperators\n";
	cout << "$$$>	" << Hexr(char_vector,     "as", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_vector, 12, "as", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_vector,     "ca", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_vector, 12, "ca", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_vector,     "ia", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_vector, 12, "ia", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_vector,     "ha", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_vector, 12, "ha", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_vector,     "hs", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_vector, 12, "hs", " - ").s() << '\n'; 

	cout << "\nPART 2: Without seperators\n";
	cout << "$$$>	" << Hexr(char_vector,     "as").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_vector, 12, "as").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_vector,     "ca").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_vector, 12, "ca").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_vector,     "ia").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_vector, 12, "ia").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_vector,     "ha").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_vector, 12, "ha").s() << '\n'; 

	cout << "$$$>	" << Hexr(char_vector,     "hs").s() << '\n'; 
	cout << "$$$>	" << Hexr(char_vector, 12, "hs").s() << '\n'; 


	cout << "\nPart3: Getter Based\n";
	Hexr cv_hexr(char_vector);
	cout << "$$$>	" << cv_hexr.ascii_string() << '\n'; 
	print_vector(cv_hexr, "ca");
	print_vector(cv_hexr, "ia");
	print_vector(cv_hexr, "ha");
	cout << "$$$>	" << cv_hexr.hex_string() << '\n'; 

	cout << "\n\n";

	Hexr man_cv_hexr(char_vector, 12);
	cout << "$$$>	" << man_cv_hexr.ascii_string() << '\n'; 
	print_vector(man_cv_hexr, "ca");
	print_vector(man_cv_hexr, "ia");
	print_vector(man_cv_hexr, "ha");
	cout << "$$$>	" << man_cv_hexr.hex_string() << '\n'; 


	cout << "\n-------------------------------------------------------------------------------\n";
	cout << "CONVERT INT ARRAY\n";
	cout << "-------------------------------------------------------------------------------\n";
	cout << "\nPART 1: Using Seperators\n";
	cout << "$$$>	" << Hexr(int_array,     "as", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_array, 12, "as", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_array,     "ca", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_array, 12, "ca", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_array,     "ia", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_array, 12, "ia", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_array,     "ha", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_array, 12, "ha", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_array,     "hs", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_array, 12, "hs", " - ").s() << '\n'; 

	cout << "\nPART 2: Without seperators\n";
	cout << "$$$>	" << Hexr(int_array,     "as").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_array, 12, "as").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_array,     "ca").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_array, 12, "ca").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_array,     "ia").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_array, 12, "ia").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_array,     "ha").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_array, 12, "ha").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_array,     "hs").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_array, 12, "hs").s() << '\n'; 


	cout << "\nPart3: Getter Based\n";
	Hexr ia_hexr(int_array);
	cout << "$$$>	" << ia_hexr.ascii_string() << '\n'; 
	print_array(ia_hexr, "ca");
	print_array(ia_hexr, "ia");
	print_array(ia_hexr, "ha");
	cout << "$$$>	" << ia_hexr.hex_string() << '\n'; 

	cout << "\n\n";

	Hexr man_ia_hexr(int_array, 12);
	cout << "$$$>	" << man_ia_hexr.ascii_string() << '\n'; 
	print_array(man_ia_hexr, "ca");
	print_array(man_ia_hexr, "ia");
	print_array(man_ia_hexr, "ha");
	cout << "$$$>	" << man_ia_hexr.hex_string() << '\n'; 


	cout << "\n-------------------------------------------------------------------------------\n";
	cout << "CONVERT INT VECTOR\n";
	cout << "-------------------------------------------------------------------------------\n";
	cout << "\nPART 1: Using Seperators\n";
	cout << "$$$>	" << Hexr(int_vector,     "as", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_vector, 12, "as", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_vector,     "ca", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_vector, 12, "ca", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_vector,     "ia", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_vector, 12, "ia", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_vector,     "ha", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_vector, 12, "ha", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_vector,     "hs", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_vector, 12, "hs", " - ").s() << '\n'; 

	cout << "\nPART 2: Without seperators\n";
	cout << "$$$>	" << Hexr(int_vector,     "as").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_vector, 12, "as").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_vector,     "ca").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_vector, 12, "ca").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_vector,     "ia").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_vector, 12, "ia").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_vector,     "ha").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_vector, 12, "ha").s() << '\n'; 

	cout << "$$$>	" << Hexr(int_vector,     "hs").s() << '\n'; 
	cout << "$$$>	" << Hexr(int_vector, 12, "hs").s() << '\n'; 


	cout << "\nPart3: Getter Based\n";
	Hexr iv_hexr(int_vector);
	cout << "$$$>	" << iv_hexr.ascii_string() << '\n'; 
	print_vector(iv_hexr, "ca");
	print_vector(iv_hexr, "ia");
	print_vector(iv_hexr, "ha");
	cout << "$$$>	" << iv_hexr.hex_string() << '\n'; 

	cout << "\n\n";

	Hexr man_iv_hexr(int_vector, 12);
	cout << "$$$>	" << man_iv_hexr.ascii_string() << '\n'; 
	print_vector(man_iv_hexr, "ca");
	print_vector(man_iv_hexr, "ia");
	print_vector(man_iv_hexr, "ha");
	cout << "$$$>	" << man_iv_hexr.hex_string() << '\n'; 

	cout << "\n-------------------------------------------------------------------------------\n";
	cout << "CONVERT HEX ARRAY\n";
	cout << "-------------------------------------------------------------------------------\n";
	cout << "\nPART 1: Using Seperators\n";
	cout << "$$$>	" << Hexr(hex_array,     "as", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_array, 12, "as", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_array,     "ca", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_array, 12, "ca", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_array,     "ia", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_array, 12, "ia", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_array,     "ha", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_array, 12, "ha", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_array,     "hs", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_array, 12, "hs", " - ").s() << '\n'; 

	cout << "\nPART 2: Without seperators\n";
	cout << "$$$>	" << Hexr(hex_array,     "as").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_array, 12, "as").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_array,     "ca").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_array, 12, "ca").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_array,     "ia").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_array, 12, "ia").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_array,     "ha").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_array, 12, "ha").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_array,     "hs").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_array, 12, "hs").s() << '\n'; 

	cout << "\nPart3: Getter Based\n";
	Hexr ha_hexr(hex_array);
	cout << "$$$>	" << ha_hexr.ascii_string() << '\n'; 
	print_array(ha_hexr, "ca");
	print_array(ha_hexr, "ia");
	print_array(ha_hexr, "ha");
	cout << "$$$>	" << ha_hexr.hex_string() << '\n'; 

	cout << "\n\n";

	Hexr man_ha_hexr(hex_array, 12);
	cout << "$$$>	" << man_ha_hexr.ascii_string() << '\n'; 
	print_array(man_ha_hexr, "ca");
	print_array(man_ha_hexr, "ia");
	print_array(man_ha_hexr, "ha");
	cout << "$$$>	" << man_ha_hexr.hex_string() << '\n'; 


	cout << "\n-------------------------------------------------------------------------------\n";
	cout << "CONVERT HEX VECTOR\n";
	cout << "-------------------------------------------------------------------------------\n";
	cout << "\nPART 1: Using Seperators\n";
	cout << "$$$>	" << Hexr(hex_vector,     "as", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_vector, 12, "as", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_vector,     "ca", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_vector, 12, "ca", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_vector,     "ia", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_vector, 12, "ia", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_vector,     "ha", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_vector, 12, "ha", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_vector,     "hs", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_vector, 12, "hs", " - ").s() << '\n'; 

	cout << "\nPART 2: Without seperators\n";
	cout << "$$$>	" << Hexr(hex_vector,     "as").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_vector, 12, "as").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_vector,     "ca").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_vector, 12, "ca").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_vector,     "ia").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_vector, 12, "ia").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_vector,     "ha").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_vector, 12, "ha").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_vector,     "hs").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_vector, 12, "hs").s() << '\n'; 


	cout << "\nPart3: Getter Based\n";
	Hexr hv_hexr(hex_vector); 
	cout << "$$$>	" << hv_hexr.ascii_string() << '\n'; 
	print_vector(hv_hexr, "ca");
	print_vector(hv_hexr, "ia");
	print_vector(hv_hexr, "ha");
	cout << "$$$>	" << hv_hexr.hex_string() << '\n'; 

	cout << "\n\n";

	Hexr man_hv_hexr(hex_vector, 12); 
	cout << "$$$>	" << man_hv_hexr.ascii_string() << '\n'; 
	print_vector(man_hv_hexr, "ca");
	print_vector(man_hv_hexr, "ia");
	print_vector(man_hv_hexr, "ha");
	cout << "$$$>	" << man_hv_hexr.hex_string() << '\n'; 


	cout << "\n-------------------------------------------------------------------------------\n";
	cout << "CONVERT HEX STRING\n";
	cout << "-------------------------------------------------------------------------------\n";
	cout << "\nPART 1: Using Seperators\n";
	cout << "$$$>	" << Hexr(hex_string,     "as", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_string, 12, "as", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_string,     "ca", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_string, 12, "ca", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_string,     "ia", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_string, 12, "ia", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_string,     "ha", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_string, 12, "ha", " - ").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_string,     "hs", " - ").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_string, 12, "hs", " - ").s() << '\n'; 

	cout << "\nPART 2: Without seperators\n";
	cout << "$$$>	" << Hexr(hex_string,     "as").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_string, 12, "as").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_string,     "ca").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_string, 12, "ca").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_string,     "ia").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_string, 12, "ia").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_string,     "ha").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_string, 12, "ha").s() << '\n'; 

	cout << "$$$>	" << Hexr(hex_string,     "hs").s() << '\n'; 
	cout << "$$$>	" << Hexr(hex_string, 12, "hs").s() << '\n'; 

	cout << "\nPart3: Getter Based\n";
	Hexr hs_hexr(hex_string); 
	cout << "$$$>	" << hs_hexr.ascii_string() << '\n'; 
	print_array(hs_hexr, "ca");
	print_array(hs_hexr, "ia");
	print_array(hs_hexr, "ha");
	cout << "$$$>	" << hs_hexr.hex_string() << '\n'; 

	cout << "\n\n";

	Hexr man_hs_hexr(hex_string, 12); 
	cout << "$$$>	" << man_hs_hexr.ascii_string() << '\n'; 
	print_array(man_hs_hexr, "ca");
	print_array(man_hs_hexr, "ia");
	print_array(man_hs_hexr, "ha");
	cout << "$$$>	" << man_hs_hexr.hex_string() << '\n'; 	

	cout << '\n';
    return 0;
}
 
 
