
#include "Full.h"
#include "Option.h"


int main()
{
    // NOTE: All test functions are inline to make example reading easier.
    Full full;
    full.Basics();
    full.add_n_join();

    // NOTE: All test functions are inline to make example reading easier.
    Option option;
    option.split();
    option.findall();
    option.match();
    option.sub();
    option.char_count();
    option.str_count();

    return 0;
}
