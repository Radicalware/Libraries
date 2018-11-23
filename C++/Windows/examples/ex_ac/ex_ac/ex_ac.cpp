#include ".\include\ex_ac.h"
#include ".\include\fuzz_slice.h"
#include ".\include\fuzz_dice.h"

int main()
{
	ex_ac();
	fuzz_slice();
	fuzz_dice();

	return 0;
}
