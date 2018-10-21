#pragma once


#include<iostream>
#include<vector>
#include<string>

#include "ord.h"
#include "re.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

void line() {
	cout << "--------------------------------------------------------\n";
}
void dline() {
	cout << "========================================================\n";
}
void print(std::string data) {
	cout << data << endl;
}

template<typename T>
void pos_xy_pos_z(T list) {
	dline(); // wish C++ had python decorators
	print("Slice positive x/y & pos z\nForward Direction Z\n");
	cout << "(0, 0)\n"    << ord::join(ord::slice(list, 0, 0), " ") << endl;
	cout << "0 1 2 3 4 5 6 7 8" << endl;
	cout << "\n(0, 0, 1)\n" << ord::join(ord::slice(list, 0, 0, 1), " ") << endl;
	cout << "0 1 2 3 4 5 6 7 8" << endl;
	cout << "\n(0, 0, 2)\n" << ord::join(ord::slice(list, 0, 0, 2), " ") << endl;
	cout << "0 2 4 6 8" << endl;
	cout << "\n(0, 0, 3)\n" << ord::join(ord::slice(list, 0, 0, 3), " ") << endl;
	cout << "0 3 6" << endl;
	line();
	cout << "(0, 8)\n" << ord::join(ord::slice(list, 0, 8), " ") << endl;
	cout << "0 1 2 3 4 5 6 7 8" << endl;
	cout << "\n(1, 7)\n" << ord::join(ord::slice(list, 1, 7), " ") << endl;
	cout << "1 2 3 4 5 6 7" << endl;
	cout << "\n(2, 6)\n" << ord::join(ord::slice(list, 2, 6), " ") << endl;
	cout << "2 3 4 5 6" << endl;

}

template<typename T>
void pos_xy_neg_z(T list) {
	dline(); // wish C++ had python decorators
	print("Slice pos x/y neg z\nReverse Direction Z");
	cout << "\n(8, 0, -1)\n" << ord::join(ord::slice(list, 8, 0, -1), " ") << endl;
	cout << "8 7 6 5 4 3 2 1 0" << endl;
	cout << "\n(8, 0, -2)\n" << ord::join(ord::slice(list, 8, 0, -2), " ") << endl;
	cout << "8 6 4 2 0" << endl;
	cout << "\n(8, 0, -3)\n" << ord::join(ord::slice(list, 8, 0, -3), " ") << endl;
	cout << "8 5 2" << endl;
	line();
	cout << "(8, 0, -1)\n" << ord::join(ord::slice(list, 8, 0, -1), " ") << endl;
	cout << "8 7 6 5 4 3 2 1 0" << endl;
	cout << "\n(7, 1, -1)\n" << ord::join(ord::slice(list, 7, 1, -1), " ") << endl;
	cout << "7 6 5 4 3 2 1" << endl;
	cout << "\n(6, 2, -1)\n" << ord::join(ord::slice(list, 6, 2, -1), " ") << endl;
	cout << "6 5 4 3 2" << endl;

}

template<typename T>
void neg_xy_pos_z(T list) {
	dline(); // wish C++ had python decorators
	print("Slice neg x/y pos z\nForward Direction Z");
	cout << "\n(-8, 0, 1)\n" << ord::join(ord::slice(list, -8, 0, 1), " ") << endl;
	cout << "0 1 2 3 4 5 6 7 8" << endl;
	cout << "\n(-8, 0, 2)\n" << ord::join(ord::slice(list, -8, 0, 2), " ") << endl;
	cout << "0 2 4 6 8" << endl;
	cout << "\n(-8, 0, 3)\n" << ord::join(ord::slice(list, -8, 0, 3), " ") << endl;
	cout << "0 3 6" << endl;
	line();
	cout << "(-8,  0)\n" << ord::join(ord::slice(list, -8,  0), " ") << endl;
	cout << "0 1 2 3 4 5 6 7 8" << endl;
	cout << "\n(-7, -1)\n" << ord::join(ord::slice(list, -7, -1), " ") << endl;
	cout << "1 2 3 4 5 6 7" << endl;
	cout << "\n(-6, -2)\n" << ord::join(ord::slice(list, -6, -2), " ") << endl;
	cout << "2 3 4 5 6" << endl;

}

template<typename T>
void neg_xy_neg_z(T list) {
	dline(); // wish C++ had python decorators
	print("Slice neg x/y neg z\nReverse Direction Z");
	cout << "\n(0, -8, -1)\n" << ord::join(ord::slice(list, -0, -8, -1), " ") << endl;
	cout << "8 7 6 5 4 3 2 1 0" << endl;
	cout << "\n(0, -8, -2)\n" << ord::join(ord::slice(list, -0, -8, -2), " ") << endl;
	cout << "8 6 4 2 0" << endl;
	cout << "\n(0, -8, -3)\n" << ord::join(ord::slice(list, -0, -8, -3), " ") << endl;
	cout << "8 5 2" << endl;
	line();
	cout << "(0, -8, -1)\n" << ord::join(ord::slice(list, -0, -8, -1), " ") << endl;
	cout << "8 7 6 5 4 3 2 1 0" << endl;
	cout << "\n(-1, -7, -1)\n" << ord::join(ord::slice(list, -1, -7, -1), " ") << endl;
	cout << "7 6 5 4 3 2 1" << endl;
	cout << "\n(-2, -6, -1)\n" << ord::join(ord::slice(list, -2, -6, -1), " ") << endl;
	cout << "6 5 4 3 2" << endl;
}

void fuzz_slice() {
	line();
	cout << "Slice = ('start point (inclusive)', 'end point (inclusive)', 'direction & skipping')";
	cout << "end-point defaults >> if (Z > 0){ end-point = m_size } else { m_size = 0 }\n";
	cout << "both start/end points must be inclusive sense a null = 0; and such thing as ord::join(ord::slice(list,,2)\n";
	cout << "if (Z >= 0) { forward_direction; } else { Reverse_Direction; }\n";

	vector<int>	list{ 0,1,2,3,4,5,6,7,8 }; // 9 ints
	//std::string	list = "012345678" ; // 9 ints

	pos_xy_pos_z(list);
	pos_xy_neg_z(list);

	neg_xy_pos_z(list);
	neg_xy_neg_z(list);
}