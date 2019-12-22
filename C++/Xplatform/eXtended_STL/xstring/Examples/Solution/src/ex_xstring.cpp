
#include "Full.h"
#include "Option.h"

int main()
{
    Nexus<>::Start();

    // NOTE: All test functions are inline to make example reading easier.
    Full full;
    full.Basics();
    full.add_n_join();

    // NOTE: All test functions are inline to make example reading easier.
    Option option;
    option.split();
    option.findall();
    option.search(); 
    option.match();
    option.sub();
    option.char_count();
    option.str_count();


    Nexus<>::Stop();
    return 0;
}
