#pragma once


#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include<Windows.h>
#else 
#include<unistd.h>
#endif

#include<iostream>
#include<initializer_list>
#include<utility>

#include<thread>
#include<mutex>
#include<condition_variable>

#include<queue>
#include<unordered_map>
#include<functional>
#include<type_traits>

#include "RawMapping.h"
#include "Atomic.h"
#include "Threads.h"
#include "Mutex.h"
#include "Task.h"
#include "Job.h"

#define SyncNexus() Nexus<>::WaitAll()
#define EXIT() exit(Nexus<>::Stop())

template<typename M>
using IsSharedPtr = typename std::enable_if<IsSame(M, RA::SharedPtr<RA::Mutex>) || IsSame(M, xp<RA::Mutex>), void>::type;

// =========================================================================================

template<typename T> class Nexus;
template<> class Nexus<void>;

// WARNING: MAKE SURE YOU ONLY USE FUNCTIONS THAT TAKE L-VALUE REFERENCE
template<>
class __single_inheritance Nexus<void> : public RA::Threads
{
private:
    istatic RA::Mutex SoMutex;

    istatic bool SbInitialized = false;
    istatic std::atomic<bool> SbFinishTasks = false;
    istatic RA::Atomic<long long int> SnInstTaskCount = 0;
    
    // thread not jthread for NVCC
    istatic std::vector<std::thread> MvThreads; // these threads start in the constructor and don't stop until Nexus is over
    istatic std::queue<RA::SharedPtr<Task<void>>> ScTaskQueue; // This is where tasks are held before being run
    // No Getter Mutex/Sig for Nexus<void> because you pass in by ref, you don't pull any stored values
    istatic std::unordered_map<size_t, RA::SharedPtr<RA::Mutex>> SmMutex; // for objects in threads
    istatic RA::SharedPtr<RA::Mutex> SoBlankMutexPtr = RA::MakeShared<RA::Mutex>(); // This blank Mutex used when no mutex is given as a param
    // Nexus<void> can't store output, hance no storage containers here

    istatic void TaskLooper(int thread_idx);

public:
    INL Nexus(){ Nexus<void>::Start(); }
    INL virtual ~Nexus() { Nexus<void>::Stop(); }
    istatic void Start();
    istatic int  Stop();
    istatic void ForceStop();

    istatic void SetMutexOn(size_t FoMutex)  { SmMutex.at(FoMutex).Get().SetMutexOn(); }
    istatic void SetMutexOff(size_t FoMutex) { SmMutex.at(FoMutex).Get().SetMutexOff(); }
    // ------------------------------------------------------------------------------------------------------------------------------------------------
    // Job: Function + Args
    template <typename F, typename... A>
    istatic UsingFunction(void) AddJob(F&& Function, A&& ... Args);

    // Job Object.Function + Args
    template <typename O, typename F, typename... A>
    istatic UsingFunction(void) AddJob(O& object, F&& Function, A&& ... Args);

    //  Job: Mutex + Object.Function + Args
    template <typename M, typename O, typename F, typename... A>
    istatic UsingFunction(void) AddJob(M& FoMutex, O& object, F&& Function, A&& ... Args);
    // ------------------------------------------------------------------------------------------------------------------------------------------------
    // Job: Function + Ref-Arg + Args
    template <typename F, typename V, typename... A>
    istatic void AddJobVal(F&& Function, V& element, A&&... Args);

    // Job: Function + Ref-(Key/Value) + Args
    template <typename K, typename V, typename F, typename... A>
    istatic void AddJobPair(F&& Function, K key, V& value, A&& ... Args);

    // Required due to Linux not Windows (returns void Job<T>)
    // static Job<short int> GetWithoutProtection(size_t dummy) noexcept;

    istatic size_t Size() { return SnInstTaskCount; }

    istatic void WaitAll();
    istatic bool TaskCompleted() { return ScTaskQueue.size();  }
    istatic void Clear();
    istatic void CheckClearMutexes();

    istatic void Sleep(unsigned int FnMilliseconds);
};

// =========================================================================================


INL void Nexus<void>::TaskLooper(int thread_idx)
{
    while (true)
    {
        size_t MutexIdx = 0;
        RA::SharedPtr<Task<void>> TaskPtr;
        {
            auto Lock = SoMutex.CreateLock([]() {
                return ((Nexus<void>::SbFinishTasks || Nexus<void>::ScTaskQueue.size()) && (RA::Threads::GetAllowedThreadCount() - SnInstTaskCount > 0));
                });

            if (ScTaskQueue.empty())
                return;

            TaskPtr = ScTaskQueue.front();
            MutexIdx = ScTaskQueue.front().Get().GetMutexID();
            ScTaskQueue.pop();
            SnInstTaskCount++;
        }

        if (TaskPtr == nullptr)
            continue;
        Task<void>& VoidTask = TaskPtr.Get();

        if (!MutexIdx) // no lock given
            VoidTask.RunTask();
        else if (SmMutex.at(MutexIdx).Get().IsMutexOn()) // lock was given with a mutex set to on
        {
            auto Lock = SmMutex.at(MutexIdx).Get().CreateLock();
            VoidTask.RunTask();
        }
        else // lock was given but the mutex was set to off
            VoidTask.RunTask();

        auto Lock = SoMutex.CreateLock();
        if (SbFinishTasks && !ScTaskQueue.size())
        {
            SnInstTaskCount = 0;
            return;
        }
        SnInstTaskCount--;
    }
};

INL void Nexus<void>::Start()
{
    SbFinishTasks = false; // Used to exit when threading need to join for the last time;
    if (!SbInitialized)
    {
        SoBlankMutexPtr.Get().SetMutexOff();
        SmMutex.clear();
        SmMutex.insert({ SoBlankMutexPtr.Get().GetID(), SoBlankMutexPtr }); // starts size at 1 and index at 0

        RA::Threads::InstanceCount++;
        MvThreads.reserve(RA::Threads::Allowed);
        for (int i = 0; i < RA::Threads::Allowed; ++i)
            MvThreads.emplace_back(std::bind((void(*)(int)) & Nexus<void>::TaskLooper, i)); // static member Function, don't use "this"

        RA::Threads::Used = 0;
    }
    SbInitialized = true;
}

INL int Nexus<void>::Stop()
{
    if (SbInitialized)
    {
        Nexus<void>::WaitAll();
        SbFinishTasks = true;
        SoMutex.UnlockAll();
        for (auto& thrd : MvThreads) thrd.join();
        SmMutex.clear();
    }
    SbInitialized = false;
    for (auto& LoThread : MvThreads)
        if(LoThread.joinable())
            LoThread.join();
    return 0;
}

INL void Nexus<void>::ForceStop()
{
    MvThreads.clear();
    while (ScTaskQueue.size())
        ScTaskQueue.pop();
    SmMutex.clear();

    SbInitialized = false;
    SbFinishTasks = false;
    SnInstTaskCount = 0;
    RA::Threads::InstanceCount--;
    RA::Threads::Used = 0;

}

// If you are meaning to get an object version, pass in "This" instead of "this"
template<typename F, typename ...A>
INL UsingFunction(void) Nexus<void>::AddJob(F&& Function, A&& ...Args)
{
    auto Lock = SoMutex.CreateLock();
    auto BindedFunction = std::bind(std::move(Function), std::forward<A>(Args)...);
    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), 0));
}

template<typename O, typename F, typename ...A>
INL UsingFunction(void) Nexus<void>::AddJob(O& object, F&& Function, A&& ...Args)
{
    auto Lock = SoMutex.CreateLock();
    auto BindedFunction = std::bind(std::move(Function), std::ref(object), std::forward<A>(Args)...);
    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), 0));
}

template <typename M, typename O, typename F, typename... A>
INL UsingFunction(void) Nexus<void>::AddJob(M& FoMutex, O& object, F&& Function, A&& ...Args)
{
    auto Lock = SoMutex.CreateLock();
    CheckClearMutexes();
    auto BindedFunction = std::bind(std::move(Function), std::ref(object), std::forward<A>(Args)...);

    if (FoMutex == nullptr)
        FoMutex = RA::MakeShared<RA::Mutex>();

    if (SmMutex.size() <= FoMutex.Get().GetID()) // id should never be 'gt' size
    {
        if (SmMutex.find(FoMutex.Get().GetID()) != SmMutex.end()) // find not contains for NVCC
            SmMutex[FoMutex.Get().GetID()] = FoMutex;
        else
            SmMutex.insert({ FoMutex.Get().GetID(), FoMutex });
    }

    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), FoMutex.Get().GetID()));
    // nxm.id references the location in SmMutex
}

template <typename F, typename V, typename... A>
INL void Nexus<void>::AddJobVal(F&& Function, V& element, A&&... Args)
{
    auto Lock = SoMutex.CreateLock();
    CheckClearMutexes();
    auto BindedFunction = std::bind(std::move(Function), std::ref(element), std::ref(Args)...);
    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), 0));
}

template<typename K, typename V, typename F, typename ...A>
INL void Nexus<void>::AddJobPair(F&& Function, K key, V& value, A&& ...Args)
{
    auto Lock = SoMutex.CreateLock();
    CheckClearMutexes();
    auto BindedFunction = std::bind(std::move(Function), std::ref(key), std::ref(value), std::ref(Args)...);
    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), 0));
}

// Class required due to Linux (not Windows)
// Job<short int> Nexus<void>::GetWithoutProtection(size_t dummy) noexcept { return Job<short int>(); }

INL void Nexus<void>::WaitAll()
{
    while (ScTaskQueue.size() || SnInstTaskCount > 0)
    {
        Nexus<void>::Sleep(1);
    }
}

INL void Nexus<void>::Clear()
{
    Nexus<void>::WaitAll();

    SnInstTaskCount = 0;
    if (RA::Threads::Used == 0 && ScTaskQueue.size() == 0)
    {
        SmMutex.clear();
        SmMutex.insert({ SoBlankMutexPtr.Get().GetID(), SoBlankMutexPtr });
    }
    RA::Mutex::ResetTotalMutexCount();
}

INL void Nexus<void>::CheckClearMutexes()
{
    if (RA::Threads::Used == 0 && ScTaskQueue.size() == 0)
    {
        SmMutex.clear();
        SmMutex.insert({ SoBlankMutexPtr.Get().GetID(), SoBlankMutexPtr });
    }
}

INL void Nexus<void>::Sleep(unsigned int FnMilliseconds)
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    ::Sleep(FnMilliseconds);
#else
    ::usleep(FnMilliseconds);
#endif
}

