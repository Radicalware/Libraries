#include<iostream>
#include<string>
#include<vector>


using std::cout;
using std::endl;
using std::string;
using std::vector;

#include "./fuzz_slice.h"
#include "./fuzz_dice.h"

int main(int argc, char** argv){

	fuzz_slice();
	fuzz_dice();

    return 0;
}
