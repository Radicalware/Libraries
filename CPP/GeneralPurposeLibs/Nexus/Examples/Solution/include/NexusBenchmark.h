
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
void test_exception(xint FnInput, Nexus<T>& FoNexUll) {
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

ull PrimeNumber(xint FnPlacement) 
{
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

    FnPlacement = Prime;
    return Prime;
}


ull PrimeNumberRef(xint& FnPlacement)
{
    FnPlacement = PrimeNumber(FnPlacement);
    return FnPlacement;
}

void PrintPrimeNumber(xint FnPrimeNum)
{
    printf("%llu\n", PrimeNumber(FnPrimeNum));
}

xstring str_prime_number(xint FnPlacement) {
    return RA::ToXString(PrimeNumber(FnPlacement));
}

void BenchmarkNexus()
{
// =============================================================================================
    Begin();
    Nexus<ull> NexUll;

    xint i = 0;
    NexUll.Disable(); // for fast loading
    for (i = 10; i < 16; i++)
        NexUll.AddTask(PrimeNumber, i);
    NexUll.Enable();
    NexUll.WaitAll();
    for (int i = 1; i <= NexUll.Size(); i++)
        cout << i << " " << NexUll.Get(i) << endl;

    cout << "I: " << i << endl;
    NexUll.AddTaskRef(PrimeNumberRef, i); // ref versions take ref params
    NexUll.WaitAll();
    cout << "I: " << i << endl;

    NexUll.Clear();

    NexUll.AddTask("ten", PrimeNumber, 10);
    NexUll.AddTask("eleven", PrimeNumber, 11);
    NexUll.AddTask("twelve", PrimeNumber, 12);
    NexUll.AddTask(PrimeNumber, 13);             // no key for 13, you must reference by int
    NexUll.AddTask("fourteen", PrimeNumber, 14);
    NexUll.AddTask("fifteen", PrimeNumber, 15);

    NexUll.WaitAll();

    cout << "thread zero  : " << NexUll(1) << endl; // task ID 0 is reserved for void tasks
    cout << "thread one   : " << NexUll.Get(2) << endl;
    cout << "thread two   : " << NexUll.Get(3) << endl;
    cout << "thread three : " << NexUll.Get(4) << endl;
    cout << "thread four  : " << NexUll.Get(5) << endl;
    cout << "thread five  : " << NexUll.Get(6) << endl;
    test_exception(20, NexUll); // 20 tasks were not created.


    cout << "thread zero  : " << NexUll.Get("ten") << endl;
    cout << "thread one   : " << NexUll.Get("eleven") << endl;
    cout << "thread two   : " << NexUll.Get("twelve") << endl;
    test_exception("thirteen", NexUll); // key thirteen was not given.
    cout << "thread four  : " << NexUll.Get("fourteen") << endl;
    cout << "thread five  : " << NexUll.Get("fifteen") << "\n\n";

    // =============================================================================================

    NexUll.AddTask("Failure 0", PrimeNumber, 0);
    NexUll.AddTask("Failure 99", PrimeNumber, 99);

    test_exception("Failure 0", NexUll);
    test_exception("Failure 99", NexUll);
    cout << "\n";

    // =============================================================================================

    Nexus<xstring> NexStr;
    auto prime_34_arg0 = []() -> xstring {
        return RA::ToXString(PrimeNumber(34));
    };
    auto prime_35_arg1 = [](xint val) -> xstring {
        return RA::ToXString(int(val));
    };
    auto prime_46_arg2 = [](int val1, const xstring& val2) -> xstring {
        return RA::ToXString(PrimeNumber(static_cast<xint>(val1) + val2.To64()));
    };
    // works with no args, one arg or multiple args

    RA::TestAttribute::Function1(prime_34_arg0);
    RA::TestAttribute::Aggrigate1(prime_34_arg0);
    RA::TestAttribute::Invocable1(prime_34_arg0);
    RA::TestAttribute::InvocReq1(prime_34_arg0);
    RA::TestAttribute::Trivial1(prime_34_arg0);
    RA::TestAttribute::CopyAssignable1(prime_34_arg0);
    RA::TestAttribute::Constructible1(prime_34_arg0);

    NexStr.AddTask(prime_34_arg0);
    NexStr.AddTask(prime_35_arg1, 35);
    NexStr.AddTask("two args", prime_46_arg2, 36, "10");
    NexStr.WaitAll();

    cout << NexStr.Get(1) << endl; // remember task id 0 is reserved for void tasks
    cout << NexStr.Get(2) << endl;
    cout << NexStr.Get("two args") << endl;

    test_exception(3, NexStr);

    // =============================================================================================

    NexUll.WaitAll();
    NexUll.Clear();

    FunctionalExamples(NexUll, NexStr, &PrimeNumber);

    NexStr.WaitAll();
    NexStr.Clear();

    Rescue();
}