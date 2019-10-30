#include<iostream>
#include<exception>

#include<functional>  // for testing only
#include<thread>      // for testing only
#include<type_traits> // for testing only

#include "Nexus.h"
#include "xstring.h"
#include "xvector.h"

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

struct NUM // number represented by a str
{
    xstring val = "0";
    NUM() {}

    xstring inc(int input)
    {
        val = to_xstring(val.to_int() + input);
        return val;
    }
    xstring dec(int input)
    {
        val = to_xstring(val.to_int() - input);
        return val;
    }
};

int main() 
{
    NUM num;
    Nexus<void> nxv;
    nxv.mutex_on(); // very important when modifying one object in multiple threads!!

    for (int i = 0; i < 200; i++) {
        if (i % 3 == 0)
            nxv.add_job(&NUM::dec, num, 1);
        else
            nxv.add_job(&NUM::inc, num, 1);
    }
    nxv.wait_all();
    num.val.print();

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

    cout << "Jobs in progress should not exceed: " << nex_ull.thread_count() << endl;
    cout << "Jobs In Progress: " << nex_ull.threads_used() << endl;
    cout << "Starting Loop\n";

    cout << "waiting for jobs to finish\n";

    nex_ull.wait_all();
    nex_str -= 1; // decrease allowed threads by 1
    cout << "Usable Threads: " << nex_ull.thread_count() << endl;

    for (int i = 0; i < nex_ull.thread_count() * 2; i++) {
        nex_ull.add_job(prime_number, 10000);
        nex_ull.sleep(5); // a thread isn't listed as being "used" until the actual process starts
        // not when the "add_job" function is executed because that process may be just sitting in a queue
        cout << "Jobs Running: " << nex_ull.threads_used() << endl;
    }
    nex_ull.reset_thread_count();
    cout << "Usable Threads: " << nex_ull.thread_count() << endl;

    nex_ull.wait_all(); // wait all isn't required because the getter will cause the wait
    // but the print will be smoother if you wait for all the values to be populated first
    for (int i = 0; i < nex_ull.size(); i++) {
        cout << nex_ull(i) << endl;
    }

    Nexus<void> nex_void;
    nex_void.add_job([]()->void {
        cout << "Void Job" << endl;
        });
    nex_void.wait_all();

    return 0;
}
