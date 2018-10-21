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

template<typename T>
void _pos_xy_pos_z(T list) {
	dline(); // wish C++ had python decorators
	print("DICE positive x/y & pos z\nForward Direction Z\n");
	cout << "(0, 0)\n" << ord::join(ord::dice(list, 0, 0), " ") << endl;
	cout << "0 1 2 3 4 5 6 7 8" << endl;
	cout << "\n(0, 0, 1)\n" << ord::join(ord::dice(list, 0, 0, 1), " ") << endl;
	cout << "blank - everything removed" << endl;
	cout << "\n(0, 0, 2)\n" << ord::join(ord::dice(list, 0, 0, 2), " ") << endl;
	cout << "1 3 5 7" << endl;
	cout << "\n(0, 0, 3)\n" << ord::join(ord::dice(list, 0, 0, 3), " ") << endl;
	cout << "1 2 4 5 7 8" << endl;
}

template<typename T>
void _pos_xy_neg_z(T list) {
	dline(); // wish C++ had python decorators
	print("DICE pos x/y neg z\nReverse Direction Z");
	cout << "\n(8, 0, -1)\n" << ord::join(ord::dice(list, 8, 0, -1), " ") << endl;
	cout << "8 7 6 5 4 3 2 1 0" << endl;
	cout << "\n(8, 0, -2)\n" << ord::join(ord::dice(list, 8, 0, -2), " ") << endl;
	cout << "7 5 3 1" << endl;
	cout << "\n(8, 0, -3)\n" << ord::join(ord::dice(list, 8, 0, -3), " ") << endl;
	cout << "7 6 4 3 1 0" << endl;
}

template<typename T>
void _neg_xy_pos_z(T list) {
	dline(); // wish C++ had python decorators
	print("DICE neg x/y pos z\nForward Direction Z");
	cout << "\n(-8, 0, 1)\n" << ord::join(ord::dice(list, -8, 0, 1), " ") << endl;
	cout << "blank, all erased" << endl;
	cout << "\n(-8, 0, 2)\n" << ord::join(ord::dice(list, -8, 0, 2), " ") << endl;
	cout << "1 3 5 7" << endl;
	cout << "\n(-8, 0, 3)\n" << ord::join(ord::dice(list, -8, 0, 3), " ") << endl;
	cout << "1 2 4 5 7 8" << endl;
}

template<typename T>
void _neg_xy_neg_z(T list) {
	dline(); // wish C++ had python decorators
	print("DICE neg x/y neg z\nReverse Direction Z");
	cout << "\n(0, -8, -1)\n" << ord::join(ord::dice(list, -0, -8, -1), " ") << endl;
	cout << "8 7 6 5 4 3 2 1 0" << endl;
	cout << "\n(0, -8, -2)\n" << ord::join(ord::dice(list, -0, -8, -2), " ") << endl;
	cout << "7 5 3 1" << endl;
	cout << "\n(0, -8, -3)\n" << ord::join(ord::dice(list, -0, -8, -3), " ") << endl;
	cout << "7 6 4 3 1 0" << endl;
}

void fuzz_dice() {
	line();
	cout << "dice = ('start point (inclusive)', 'end point (inclusive)', 'direction & deleting')\n";
	cout << "end-point defaults >> if (Z > 0){ end-point = m_size } else { m_size = 0 }\n";
	cout << "both start/end points must be inclusive sense a null = 0; and\n";
	cout << "there is no such thing as ord::join(ord::dice(list,,2)\n";
	cout << "if (Z >= 0) { forward_direction; } else { Reverse_Direction; }\n\n";

	cout << "Note: Dice should only be used when (z > 1 || z < -1) and you want to delete (not skip)\n";

	vector<int>	list{ 0,1,2,3,4,5,6,7,8 }; // 9 ints
	//std::string	list = "012345678" ; // 9 ints

	_pos_xy_pos_z(list);
	_pos_xy_neg_z(list);

	_neg_xy_pos_z(list);
	_neg_xy_neg_z(list);
}