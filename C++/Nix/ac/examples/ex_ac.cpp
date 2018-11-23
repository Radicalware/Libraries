#include "./include/ex_basics.h"
#include "./include/fuzz_slice.h"
#include "./include/fuzz_dice.h"

int main(int argc, char** argv){


	ex_basics();
	fuzz_slice();
	fuzz_dice();	

    return 0;
}
