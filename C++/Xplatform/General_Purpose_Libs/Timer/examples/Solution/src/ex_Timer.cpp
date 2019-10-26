#include <iostream>

#include "Timer.h"

using std::cout;
using std::endl;

int main(int argc, char** argv)
{
    cout << "Wait 1/10 second\n";
    Timer timer;
    timer.sleep(100);
    cout << timer << "\n\n";

    // -----------------------------------------------------------
    cout << "Reset and wait 2/10 sec, lap and wait 1/10 second\n";
    timer.reset();
    
    timer.sleep(200);
    timer.lap();

    timer.sleep(100);
    timer.lap("3/10");

    cout << timer.get(0) << "\n";
    cout << timer.get("3/10") << "\n";
    cout << timer.get(1) << "\n\n";
    // -----------------------------------------------------------

    cout << "Pause until we lap at 4/10 sec\n";
    timer.wait_seconds(0.4);
    cout << timer << endl;

    cout << "Pause until we lap at 6/10 sec\n";
    timer.wait(600);
    cout << timer << endl;

	return 0;
}
