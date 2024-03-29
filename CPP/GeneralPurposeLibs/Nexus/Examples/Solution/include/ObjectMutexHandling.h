
#include "Macros.h"
#include "Mutex.h"

struct Bank
{
    struct Account : public RA::MutexHandler
    {
        xstring MsName;
        int MnBalance = 0;
        RA::SharedPtr<RA::Mutex> MutexPtr;
        // If LoMutex goes out of scope and then your job is queued, then your program will crash
        // LoMutex makes it so we can update multiple accounts in the same object at the same time!
    };
    Account MoAccount1;
    Account MoAccount2;
    inline static const xint MnLooper = 1550;
    inline static const xint MnSleep = 10;

    Bank() 
    {
    }

    void InstantModAccount1(int value = 1) {
        MoAccount1.MnBalance += value;
    }

    void UnsafeModAccount1(int value = 1)
    {
        RA::Timer::Sleep(MnSleep); // FOOBAR example (lock before timer)
        MoAccount1.MnBalance += value;
    }
    void UnsafeModAccount2(int value = 1)
    {
        RA::Timer::Sleep(MnSleep); // FOOBAR example (lock before timer)
        MoAccount2.MnBalance += value;
    }

    void SafeModAccount1(int value = 1)
    {
        RA::Timer::Sleep(MnSleep);
        auto Lock = MoAccount1.CreateLock();
        MoAccount1.MnBalance += value;
    }
    void SafeModAccount2(int value = 1)
    {
        RA::Timer::Sleep(MnSleep);
        auto Lock = MoAccount2.CreateLock();
        MoAccount2.MnBalance += value;
    }

    void Reset() {
        MoAccount1.MnBalance = 0;
        MoAccount2.MnBalance = 0;
    }
};


void ObjectMutexHandling()
{
    Bank LoBank;
    LoBank.MoAccount1.MsName = "Account1";
    LoBank.MoAccount2.MsName = "Account2";

    // -----------------------------------------------------------------------------------------------
    // FAST METHOD 1 (EACH ACCOUNT GETS THEIR OWN MUTEX)
    //Nexus<>::SetMutexOn(/* object_index */); this is on by default;
    RA::Timer t;
    for (int i = 0; i < LoBank.MnLooper; i++) {
        if (i % 3 == 0) { // occurs 1/3 of the time
            Nexus<>::AddTask(LoBank.MoAccount1.MutexPtr, LoBank, &Bank::UnsafeModAccount1, -2);
            Nexus<>::AddTask(LoBank.MoAccount2.MutexPtr, LoBank, &Bank::UnsafeModAccount2,  2);
        }
        else {            // occurs 2/3 of the time
            Nexus<>::AddTask(LoBank.MoAccount1.MutexPtr, LoBank, &Bank::UnsafeModAccount1,  2);
            Nexus<>::AddTask(LoBank.MoAccount2.MutexPtr, LoBank, &Bank::UnsafeModAccount2, -2);
        }
    }
    Nexus<>::WaitAll();
    cout << "Timer (Double Full Lock Function) : " << t.GetElapsedTime() << endl;
    cout << LoBank.MoAccount1.MsName << ": " << LoBank.MoAccount1.MnBalance << endl;
    cout << LoBank.MoAccount2.MsName << ": " << LoBank.MoAccount2.MnBalance << "\n\n";

    // -----------------------------------------------------------------------------------------------
    // FAST METHOD 2 (EACH ACCOUNT GETS THEIR OWN MUTEX)
    //Nexus<>::SetMutexOn(/* object_index */); this is on by default;
    LoBank.Reset();
    t.Reset();
    for (int i = 0; i < LoBank.MnLooper; i++) {
        if (i % 3 == 0) { // occurs 1/3 of the time
            Nexus<>::AddTask(LoBank, &Bank::SafeModAccount1, -2);
            Nexus<>::AddTask(LoBank, &Bank::SafeModAccount2,  2);
        }
        else {            // occurs 2/3 of the time
            Nexus<>::AddTask(LoBank, &Bank::SafeModAccount1,  2);
            Nexus<>::AddTask(LoBank, &Bank::SafeModAccount2, -2);
        }
    }
    Nexus<>::WaitAll();
    cout << "Timer (Safe Function (Fast)) : " << t.GetElapsedTime() << endl;
    cout << LoBank.MoAccount1.MsName << ": " << LoBank.MoAccount1.MnBalance << endl;
    cout << LoBank.MoAccount2.MsName << ": " << LoBank.MoAccount2.MnBalance << "\n\n";

    // -----------------------------------------------------------------------------------------------
    // SLOW METHOD (SAME MUTEX FOR BOTH ACCOUNTS)
    LoBank.Reset();
    t.Reset();
    xp<RA::Mutex> LoMutexPtr;
    for (int i = 0; i < LoBank.MnLooper; i++) {
        if (i % 3 == 0) { // occurs 1/3 of the time
            Nexus<>::AddTask(LoMutexPtr, LoBank, &Bank::UnsafeModAccount1, -2);
            Nexus<>::AddTask(LoMutexPtr, LoBank, &Bank::UnsafeModAccount2,  2);
        }
        else {            // occurs 2/3 of the time
            Nexus<>::AddTask(LoMutexPtr, LoBank, &Bank::UnsafeModAccount1,  2);
            Nexus<>::AddTask(LoMutexPtr, LoBank, &Bank::UnsafeModAccount2, -2);
        }
    }
    Nexus<>::WaitAll();
    cout << "Timer (Single Full Lock Function (Slow)) : " << t.GetElapsedTime() << endl;
    cout << "Account1: " << LoBank.MoAccount1.MnBalance << endl;
    cout << "Account2: " << LoBank.MoAccount2.MnBalance << "\n\n";

    // -----------------------------------------------------------------------------------------------
    // USE DATA OUTSIDE A NEXUS FUNCTION
    cout << "\n\n";
    LoBank.Reset();
    auto& Mutex = *LoBank.MoAccount1.MutexPtr;
    xint Modcount = 1;
    for (int i = 0; i < 12; i++) {
        if (i % 3 == 0) { // occurs 1/3 of the time
            Nexus<>::AddTask(LoBank.MoAccount1.MutexPtr, LoBank, &Bank::InstantModAccount1, 10);
            auto Lock = Mutex.CreateLock();
            cout << "Account 1 (" << Modcount++ << ") :" << LoBank.MoAccount1.MnBalance << endl;
        }
        else {            // occurs 2/3 of the time
            Nexus<>::AddTask(LoBank.MoAccount1.MutexPtr, LoBank, &Bank::InstantModAccount1, 5);
            auto Lock = Mutex.CreateLock(); // wait for nexus to finish, lock mutex so Nexus can't roll-over memory space
            cout << "Account 1 (" << Modcount++ << ") :" << LoBank.MoAccount1.MnBalance << endl; // use memory
        }
    }
    cout << "Account 1 Syncing\n";
    Nexus<>::WaitAll();
    cout << "Account 1 Final (" << Modcount++ << ") :" << LoBank.MoAccount1.MnBalance << "\n\n";    
}

