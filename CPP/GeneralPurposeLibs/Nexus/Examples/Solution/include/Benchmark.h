
#include "Macros.h"
#include "Functions.h"

using ull = unsigned long long;

template<typename T>
void test_exception(xstring FnInput, Nexus<T>& FoNexUll) {
    try {
        cout << FoNexUll.Get(FnInput) << endl;
    }
    catch (const std::exception& FoExc) {
        cout << "Exception by: \"" << FnInput << "\"\n" << FoExc.what() << "\n\n";
    }
    catch (const char* FcErr) {
        cout << "Exception by: \"" << FnInput << "\"\n" << FcErr << "\n\n";
    }
}


template<typename T>
void test_exception(size_t FnInput, Nexus<T>& FoNexUll) {
    try {
        cout << FoNexUll.Get(FnInput) << endl;
    }
    catch (const std::exception& FoExc) {
        cout << "Exception by: \"" << FnInput << "\"\n" << FoExc.what() << "\n\n";
    }
    catch (const char* FcErr) {
        cout << "Exception by: \"" << FnInput << "\"\n" << FcErr << "\n\n";
    }
}

ull PrimeNumber(size_t FnPlacement) {
    if (FnPlacement == 1)
        return 1;
    else if (FnPlacement < 1)
        throw "A value less than 1 is not allowed"; // char throw
    else if (FnPlacement == 99)
        throw xstring().At(1); // std throw

    ull Prime = 1;
    int Current = 1;
    int Counter = 0;
    do {
        Current++;
        bool zeroed = false;

        for (ull i = 2; i <= Current / 2; i++) {
            if (Current % i == 0) {
                zeroed = true;
                break;
            }
        }
        if (!zeroed) {
            Prime = Current;
            Counter++;
        }
    } while (Counter < FnPlacement || Current > 9223372036854775806);

    return Prime;
}

xstring str_prime_number(size_t FnPlacement) {
    return RA::ToXString(PrimeNumber(FnPlacement));
}

void Benchmark()
{
// =============================================================================================
    Begin();
    Nexus<ull> NexUll;
    NexUll.AddJob("ten", PrimeNumber, 10);
    NexUll.AddJob("eleven", PrimeNumber, 11);
    NexUll.AddJob("twelve", PrimeNumber, 12);
    NexUll.AddJob(PrimeNumber, 13);             // no key for 13, you must reference by int
    NexUll.AddJob("fourteen", PrimeNumber, 14);
    NexUll.AddJob("fifteen", PrimeNumber, 15);
    NexUll.WaitAll();

    cout << "thread zero  : " << NexUll(0) << endl;
    cout << "thread one   : " << NexUll.Get(1) << endl;
    cout << "thread two   : " << NexUll.Get(2) << endl;
    cout << "thread three : " << NexUll.Get(3) << endl;
    cout << "thread four  : " << NexUll.Get(4) << endl;
    cout << "thread five  : " << NexUll.Get(5) << endl;
    test_exception(20, NexUll); // 20 tasks were not created.

    cout << "thread zero  : " << NexUll.Get("ten") << endl;
    cout << "thread one   : " << NexUll.Get("eleven") << endl;
    cout << "thread two   : " << NexUll.Get("twelve") << endl;
    test_exception("thirteen", NexUll); // key thirteen was not given.
    cout << "thread four  : " << NexUll.Get("fourteen") << endl;
    cout << "thread five  : " << NexUll.Get("fifteen") << "\n\n";

    // =============================================================================================

    NexUll.AddJob("Failure 0", PrimeNumber, 0);
    NexUll.AddJob("Failure 99", PrimeNumber, 99);

    test_exception("Failure 0", NexUll);
    test_exception("Failure 99", NexUll);
    cout << "\n";

    // =============================================================================================

    Nexus<xstring> NexStr;
    auto prime_34_arg0 = []() -> xstring {
        return RA::ToXString(PrimeNumber(34));
    };
    auto prime_35_arg1 = [](size_t val) -> xstring {
        return RA::ToXString(int(val));
    };
    auto prime_46_arg2 = [](int val1, const xstring& val2) -> xstring {
        return RA::ToXString(PrimeNumber(static_cast<size_t>(val1) + val2.To64()));
    };
    // works with no args, one arg or multiple args
    NexStr.AddJob(prime_34_arg0);
    NexStr.AddJob(prime_35_arg1, 35);
    NexStr.AddJob("two args", prime_46_arg2, 36, "10");
    NexStr.WaitAll();

    cout << NexStr.Get((size_t)0) << endl;
    cout << NexStr.Get(1) << endl;
    cout << NexStr.Get("two args") << endl;

    test_exception(3, NexStr);

    // =============================================================================================

    NexUll.WaitAll();
    NexUll.Clear();

    NexStr.WaitAll();
    NexStr.Clear();



    FunctionalExamples(NexUll, NexStr, PrimeNumber);
    Rescue();
}