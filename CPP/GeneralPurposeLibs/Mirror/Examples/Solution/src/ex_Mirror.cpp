#include "Mirror.h"

#include "ExString.h"
#include "ExStringInt.h"
#include "ExInterval.h"

#include <iostream>


using std::cout;
using std::endl;
using std::string;

void PrintLine() {
    cout << "-----------------------------------------" << endl;
}

int main(int argc, char** argv)
{
    Example::String();
    PrintLine();
    Example::StringInt();
    PrintLine();
    Example::Interval();
    return 0;
}
