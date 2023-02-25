#pragma once

#include "Macros.h"
#include "re2/re2.h"



xstring GetStr(xint FInt)
{
	return RA::BindStr("Num: ", FInt);
}


void ObjectReturnHandling()
{
	xstring Str = "Num: XX";
	Nexus<xstring> Tasks;

	xstring Var1 = "XX";
	xstring Var2 = "XX";

	// in construction
}