#include "stdafx.h"

using namespace System;

#include "ex_order.h"
#include "fuzz_slice.h"

int main(array<System::String ^> ^args)
{

	ex_order();
	fuzz_slice();

    return 0;
}
