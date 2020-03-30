
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>
#include<exception>

#include<functional>  // for testing only
#include<thread>      // for testing only
#include<type_traits> // for testing only

#include "Nexus.h"
#include "xvector.h"
#include "xstring.h"
#include "xmap.h"
#include "Timer.h"

using std::cout;
using std::endl;
using std::bind;

using ull = unsigned long long;

ull prime_number(size_t placement) {
    if (placement == 1)
        return 1;
    else if (placement < 1)
        throw "A value less than 1 is not allowed"; // char throw
    else if (placement == 99)
        throw xstring().at(1); // std throw

    ull prime = 1;
    int current = 1;
    int counter = 0;
    do {
        current++;
        bool zeroed = false;

        for (ull i = 2; i <= current / 2; i++) {
            if (current % i == 0) {
                zeroed = true;
                break;
            }
        }
        if (!zeroed) {
            prime = current;
            counter++;
        }
    } while (counter < placement || current > 9223372036854775806);

    return prime;
}

xstring str_prime_number(size_t placement) {
    return to_xstring(prime_number(placement));
}

template<typename T>
void test_exception(xstring input, Nexus<T>& nex_ull) {
    try {
        cout << nex_ull.get(input) << endl;
    }
    catch (const std::exception & exc) {
        cout << "Exception by: \"" << input << "\"\n" << exc.what() << "\n\n";
    }
    catch (const char* err_str) {
        cout << "Exception by: \"" << input << "\"\n" << err_str << "\n\n";
    }
}


template<typename T>
void test_exception(size_t input, Nexus<T>& nex_ull) {
    try {
        cout << nex_ull.get(input) << endl;
    }
    catch (const std::exception & exc) {
        cout << "Exception by: \"" << input << "\"\n" << exc.what() << "\n\n";
    }
    catch (const char* err_str) {
        cout << "Exception by: \"" << input << "\"\n" << err_str << "\n\n";
    }
}

struct Bank
{
    struct Account
    {
        xstring name;
        int balance = 0;
        NX_Mutex nx_mutex; 
        // If NX_Mutex goes out of scope and then your job is queued, then your program will crash
        // NX_Mutex makes it so we can update multiple accounts in the same object at the same time!
    };
    Account account1;
    Account account2;
    size_t looper;

    Bank() {}
    void inc_account1(int value = 1) {
        Timer::Sleep(10);
        account1.balance += value;
    }
    void inc_account2(int value = 1) {
        Timer::Sleep(10);
        account2.balance += value;
    }
    void dec_account1(int value = 1) {
        Timer::Sleep(10);
        account1.balance -= value;
    }
    void dec_account2(int value = 1) {
        Timer::Sleep(10);
        account2.balance -= value;
    }

    void reset() {
        account1.balance = 0;
        account2.balance = 0;
    }
};

int main() 
{

    Nexus<>::Start(); // note: you could just make an instance of type void
    // and it would do the same thing, then when it would go out of scope (the main function)
    // it would automatically get deleted. I did what is above because I like keeping
    // static classes static and instnace classes instance based to not confuse anyone. 
    // -------------------------------------------------------------------------------------

    Bank bank;
    bank.looper = 55;
    bank.account1.name = "account1";
    bank.account2.name = "account2";

    // -------------------------------------------------------------------------------------

    //Nexus<>::Mutex_On(/* object_index */); this is on by default;
    Timer t;
    for (int i = 0; i < bank.looper; i++) {
        if (i % 3 == 0) { // occurs 1/3 of the time
            Nexus<>::Add_Job(bank.account1.nx_mutex, bank, &Bank::dec_account1, 2);
            Nexus<>::Add_Job(bank.account2.nx_mutex, bank, &Bank::inc_account2, 2);
        }
        else {            // occurs 2/3 of the time
            Nexus<>::Add_Job(bank.account1.nx_mutex, bank, &Bank::inc_account1, 2);
            Nexus<>::Add_Job(bank.account2.nx_mutex, bank, &Bank::dec_account2, 2);
        }
    }
    Nexus<>::Wait_All();
    cout << "Timer (Seperate Mutex (Fast)) : " << t.elapsed() << endl;
    cout << bank.account1.name << ": " << bank.account1.balance << endl;
    cout << bank.account2.name << ": " << bank.account2.balance << "\n\n";

    // -------------------------------------------------------------------------------------

    bank.reset();
    t.reset();
    NX_Mutex nx_mutex;
    for (int i = 0; i < bank.looper; i++) {
        if (i % 3 == 0) { // occurs 1/3 of the time
            Nexus<>::Add_Job(nx_mutex, bank, &Bank::dec_account1, 2);
            Nexus<>::Add_Job(nx_mutex, bank, &Bank::inc_account2, 2);
        }
        else {            // occurs 2/3 of the time
            Nexus<>::Add_Job(nx_mutex, bank, &Bank::inc_account1, 2);
            Nexus<>::Add_Job(nx_mutex, bank, &Bank::dec_account2, 2);
        }
    }
    Nexus<>::Wait_All();
    cout << "Timer (Shared Mutex   (Slow)) : " << t.elapsed() << endl;
    cout << "Account1: " << bank.account1.balance << endl;
    cout << "Account2: " << bank.account2.balance << "\n\n";

    // -------------------------------------------------------------------------------------

    // =============================================================================================

    Nexus<ull> nex_ull;
    nex_ull.add_job("ten", prime_number, 10);
    nex_ull.add_job("eleven", prime_number, 11);
    nex_ull.add_job("twelve", prime_number, 12);
    nex_ull.add_job(prime_number, 13);             // no key for 13, you must reference by int
    nex_ull.add_job("fourteen", prime_number, 14);
    nex_ull.add_job("fifteen", prime_number, 15);

    cout << "thread zero  : " << nex_ull(0) << endl;
    cout << "thread one   : " << nex_ull.get(1) << endl;
    cout << "thread two   : " << nex_ull.get(2) << endl;
    cout << "thread three : " << nex_ull.get(3) << endl;
    cout << "thread four  : " << nex_ull.get(4) << endl;
    cout << "thread five  : " << nex_ull.get(5) << endl;
    test_exception(20, nex_ull); // 20 tasks were not created.

    cout << "thread zero  : " << nex_ull.get("ten") << endl;
    cout << "thread one   : " << nex_ull.get("eleven") << endl;
    cout << "thread two   : " << nex_ull.get("twelve") << endl;
    test_exception("thirteen", nex_ull); // key thirteen was not given.
    cout << "thread four  : " << nex_ull.get("fourteen") << endl;
    cout << "thread five  : " << nex_ull.get("fifteen") << "\n\n";

    // =============================================================================================

    nex_ull.add_job("Failure 0", prime_number, 0);
    nex_ull.add_job("Failure 99", prime_number, 99);

    test_exception("Failure 0", nex_ull);
    test_exception("Failure 99", nex_ull);
    cout << "\n";

    // =============================================================================================

    Nexus<xstring> nex_str;
    auto prime_34_arg0 = []() -> xstring {
        return to_xstring(prime_number(34));
    };
    auto prime_35_arg1 = [](size_t val) -> xstring {
        return to_xstring(int(val));
    };
    auto prime_46_arg2 = [](int val1, const xstring& val2) -> xstring {
        return to_xstring(prime_number(static_cast<size_t>(val1) + val2.to_64()));
    };
    // works with no args, one arg or multiple args
    nex_str.add_job(prime_34_arg0);
    nex_str.add_job(prime_35_arg1, 35);
    nex_str.add_job("two args", prime_46_arg2, 36, "10");
    cout << nex_str.get((size_t)0) << endl;
    cout << nex_str.get(1) << endl;
    cout << nex_str.get("two args") << endl;

    test_exception(3, nex_str);

    // =============================================================================================

    nex_ull.wait_all();
    nex_ull.clear();

    nex_str.wait_all();
    nex_str.clear();

    cout << "Jobs in progress should not exceed: " << nex_ull.Thread_Count() << endl;
    cout << "Jobs In Progress: " << nex_ull.Threads_Used() << endl;
    cout << "Starting Loop\n";

    cout << "waiting for jobs to finish\n";

    nex_ull.wait_all();
    nex_str -= 1; // decrease allowed threads by 1
    cout << "Usable Threads: " << nex_ull.Thread_Count() << endl;

    for (int i = 0; i < nex_ull.Thread_Count() * 2; i++) {
        nex_ull.add_job(prime_number, 10000);
        nex_ull.sleep(5); // a thread isn't listed as being "used" until the actual process starts
        // not when the "add_job" function is executed because that process may be just sitting in a queue
        cout << "Jobs Running: " << nex_ull.Threads_Used() << endl;
    }
    nex_ull.Reset_Thread_Count();
    cout << "Usable Threads: " << nex_ull.Thread_Count() << endl;

    nex_ull.wait_all(); // wait all isn't required because the getter will cause the wait
    // but the print will be smoother if you wait for all the values to be populated first
    for (int i = 0; i < nex_ull.size(); i++) {
        cout << nex_ull(i) << endl;
    }

    Nexus<>::Add_Job([]()->void {
        cout << "Void Job" << endl;
        });
    Nexus<>::Wait_All();

    Nexus<>::Stop();
    return 0;
}
