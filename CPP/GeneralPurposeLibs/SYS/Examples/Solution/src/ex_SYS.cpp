
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
#include<vector>
#include<string>
#include<ctime>


// ----- Radicalware Libs -------------------
#include "xstring.h"
#include "xvector.h"
#include "xmap.h"
#include "SYS.h"
#include "Timer.h"
// ----- Radicalware Libs -------------------


RA::SYS Args;

using std::cout;
using std::endl;

void CleanExit()
{
    Nexus<>::Stop();
    xstring().ToWhite().Print();
}

int main(int argc, char** argv) 
{
    Begin();
    Nexus<>::Start();
    RA::Timer Time;
    Args.AddAlias('a', "--key-A");
    Args.AddAlias('b', "--key-B");
    Args.AddAlias('f', "--key-F");
    Args.AddAlias('x', "--key-X");

    Args.AddAlias('p', "--port");

    Args.SetArgs(argc, argv);
    cout << "Arg Parse Time: " << Time.GetElapsedTime() << endl;

    // based on the folowing arguments you pass
    //  --key-A sub-A-1 sub-A-2 sub-A-3 --key-X --key-B sub-B-1 sub-B-2 --key-C sub-C-1 sub-C-2 --key-Y --key-Z -n -wxyp 8080  9090 -ze -f 

    //  --key-A 
    //        sub-A-1 
    //        sub-A-2 
    //        sub-A-3 
    // --key-X
    // --key-B 
    //        sub-B-1 
    //        sub-B-2 
    // --key-C 
    //        sub-C-1 
    //        sub-C-2 
    // --key-Y
    // --key-Z
    // -a
    // -n
    // -wxyp 8080 9090
    // -ze
    // -f

    // notice, --key-A and 'a' are an alias of each other
    // they were both used which in most cases is not intended
    // this is to show you that even if the user does that, the program still works fine.
    
    cout << "\n\n all args = " << Args.ArgV()(1).Join(' ');
    // we skipped the first arg (which is the program) using a slice

    //// and see the count via
    cout << "\n\n argv count = " << Args.ArgC();
    cout << "\n true argc  = " << argc;
    cout << "\n program start the count at 1, then add 1 for every arg.";

    // if we are using a small script use the [] operator
    cout << "\n\n Args[5]      = " << Args[5];

    cout << "\n Args values for key '--key-B' = " << Args["--key-B"].Join(' ');

    cout << "\n full path   = " << Args.FullPath();
    cout << "\n path        = " << Args.Path();
    cout << "\n file        = " << Args.File() << "\n\n\n";

    // ----------------------------------------------------------------
    if ( Args.Has("--key-B") && Args("--key-B")) {  // both methods are the same
        cout << "--key-B exists\n"; 
    }
    else {
        cout << "error: --key-B does not exists\n"; exit(1);
    }
    // ----------------------------------------------------------------
    if (Args.Has("--key-D") || Args("--key-D")) { // both methods are the same
        cout << "error: --key-D exists\n"; exit(1);
    }
    else {
        cout << "--key-D does not exist\n";  
    }
    // ----------------------------------------------------------------
    if (Args('n') && Args('x')) {
        cout << "-n and -x options are both used\n";
    }
    else {
        cout << "error: -n and -x are not used together or not at all \n"; exit(1);
    }
    // ----------------------------------------------------------------
    if (Args('x') && Args("--key-X")) { 
        // note: both were given (this is ok because thare are no associated values with either x or --key-X)
        cout << "-x and the alias --key-X are both used\n";
    }
    else {
        cout << "error: -x and the alias --key-X are not found together\n"; exit(1);
    }
    // ----------------------------------------------------------------
    if (Args('f') && Args("--key-F")) {
        cout << "-f and the alias --key-F are both usable but only one was given\n";
    }
    else {
        cout << "error: -f and the alias --key-F are not found together\n"; exit(1);
    }
    // ----------------------------------------------------------------
    if (Args('b') && Args("--key-B")) { 
        cout << "-b and the alias --key-B are both usable but only --key-B was given\n";
    }
    else {
        cout << "error: -b and the alias --key-B are not found together\n"; exit(1);
    }
    // ----------------------------------------------------------------
    bool pass = false;
    if (Args('p') && Args("--port")){
        cout << "we have the port key\n";
        if (Args ['p'][0] == "8080" && Args["--port"][0] == "8080") {
            cout << "port designated by -p is on " << Args['p'].Join(' ') << endl;
            pass = true;
        }
    }
    if(!pass){
        cout << "error: -p is not set" << endl; exit(1);
    }
    // ----------------------------------------------------------------
    if (Args.Key("--key-A").Has("sub-B-2") || Args("--key-A", "sub-B-2")) {
        cout << "error: value sub-B-2 exist in --key-A\n"; exit(1);
    }
    else {
        cout << "value sub-B-2 does not exist in --key-A\n";
    }
    // ----------------------------------------------------------------
    if (Args.Key("--key-A").Has("sub-A-2") && Args("--key-A", "sub-A-3") ) {
        cout << "value sub-A-2 and sub-A-3 exist under --key-A\n";
    }
    else {
        cout << "error: value sub-A-2 and sub-A-3 not found under --key-A\n"; exit(1);
    }
    // ----------------------------------------------------------------
    if (Args["--key-A"][1] == "sub-A-2" && Args['p'][1] == "9090") {
        cout << "the 2nd elements for targed str and char keys passed\n";
    }
    else {
        cout << "error: str or char keys' value not found\n"; exit(1);
    }
    // ----------------------------------------------------------------


    // here is the shorthand for
    // 1. returning  key-values
    // 2. validating key-values
    cout << "\n\n";
    cout << R"(Args["--key-A"].Join(' ')  ==  )" << Args["--key-A"].Join(' ') << endl;
    cout << R"(Args("--key-A", "sub-A-1") ==  )" << Args("--key-A", "sub-A-1") << endl;
    cout << R"(Args("--key-A", "sub-A-5") ==  )" << Args("--key-A", "sub-A-5") << "\n\n";


    cout << R"(*Args["--key-A"][0] == )" << Args["--key-A"][0] << endl;
    cout << R"(*Args["--key-A"][1] == )" << Args["--key-A"][1] << endl;
    cout << R"(*Args["--key-A"][2] == )" << Args["--key-A"][2] << "\n\n";

    cout << R"(*Args["--key-B"][0] == )" << Args["--key-B"][0] << endl;
    cout << R"(*Args["--key-B"][1] == )" << Args["--key-B"][1] << endl;

    cout << xstring().ToGreen() << "Success" << xstring().ToWhite() << endl;

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
