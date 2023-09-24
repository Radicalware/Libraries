#pragma once

#include "RawMapping.h"
#ifdef BxWindows
#include<Windows.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#else 
#define BxNix
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

#include "Memory.h"
#include "Atomic.h"
#include "Threads.h"
#include "Mutex.h"

#include "Task/TaskVoid.h"
#include "Task/TaskValue.h"
#include "Task/TaskXP.h"
#include "Task/TaskFundamental.h"


#define SyncNexus() Nexus<>::WaitAll()
#define EXIT() exit(Nexus<>::Stop())

template<typename M>
using IsSharedPtr = typename std::enable_if<IsSame(M, RA::SharedPtr<RA::Mutex>) || IsSame(M, xp<RA::Mutex>), void>::type;

// =========================================================================================

template<typename T> class Nexus;
template<> class Nexus<void>;

// WARNING: MAKE SURE YOU ONLY USE FUNCTIONS THAT TAKE L-VALUE REFERENCE
template<>
class Nexus<void> : public RA::Threads
{
private:
    istatic RA::Threads SoThreads;
    istatic RA::Mutex SoMutex;
    istatic RA::Atomic<bool> SbDisabled = false;

    istatic bool SbInitialized = false;
    istatic std::atomic<bool> SbFinishTasks = false;
    istatic RA::Atomic<xint> SnInstTaskCount = 0;
    istatic RA::Atomic<bool> SbGroupingOn = false;
    
    // thread not jthread for NVCC
    istatic std::vector<std::thread> MvThreads; // these threads start in the constructor and don't stop until Nexus is over
    istatic std::queue<xp<Task<void>>> SvTaskQueue; // The is where Tasks are held before being run
    // No Getter Mutex/Sig for Nexus<void> because you pass in by ref, you don't pull any stored values
    istatic std::unordered_map<xint, xp<RA::Mutex>> SmMutex; // for objects in threads
    istatic xp<RA::Mutex> SoBlankMutexPtr = RA::MakeShared<RA::Mutex>(); // The blank Mutex used when no mutex is given as a param
    // Nexus<void> can't store output, hance no storage containers here

    istatic void TaskLooper(int thread_idx);

public:
    RIN Nexus(){ Nexus<void>::Start(); }
    RIN virtual ~Nexus() { Nexus<void>::Stop(); }
    istatic void Start();
    istatic int  Stop();
    istatic void ForceStop();
    istatic void Disable();
    istatic void Enable();
    istatic void SetGroupingOn(const bool FTruth = true) { SbGroupingOn = FTruth; }

    // ----------------------------------------------------------------------------------------------------------------------------------------------------
    // Task: Function + Args
    template <typename F, typename... A> istatic UsingFunction(void) AddTask(   F&& Function, A&& ...Args);
    template <typename F, typename... A> istatic UsingFunction(void) AddTaskRef(F&& Function, A&& ...Args);

    // Task Object.Function + Args
    template <typename O, typename F, typename... A> istatic UsingFunction(void) AddTask(   O& object, F&& Function, A&& ...Args);
    template <typename O, typename F, typename... A> istatic UsingFunction(void) AddTaskRef(O& object, F&& Function, A&& ...Args);

    //  Task: Mutex + Object.Function + Args
    template <typename M, typename O, typename F, typename... A> istatic UsingFunction(void) AddTask(   M& FoMutex, O& object, F&& Function, A&& ...Args);
    template <typename M, typename O, typename F, typename... A> istatic UsingFunction(void) AddTaskRef(M& FoMutex, O& object, F&& Function, A&& ...Args);
private:
    template <typename M, typename F> istatic void MutexTask(M& FoMutex, F& FoBindedFunction);
public:
    // ----------------------------------------------------------------------------------------------------------------------------------------------------
    template <typename F, typename V, typename... A>             istatic void AddTaskVal( F&& Function, V& element,      A&& ...Args); // Used in xvector
    template <typename K, typename V, typename F, typename... A> istatic void AddTaskPair(F&& Function, K key, V& value, A&& ...Args); // used in xmap
    // ----------------------------------------------------------------------------------------------------------------------------------------------------

    // Required due to Linux not Windows (returns void Task<T>)
    // static Task<short int> GetWithoutProtection(xint dummy) noexcept;

    istatic xint Size() { return SnInstTaskCount; }

    istatic void WaitAll();
    istatic bool TaskCompleted() { return SvTaskQueue.size();  }
    istatic void Clear();
    istatic void CheckClearMutexes();

    istatic void Sleep(unsigned int FnMilliseconds);
};

// =========================================================================================

RIN void Nexus<void>::TaskLooper(int thread_idx)
{
    try
    {
        std::list<xp<Task<void>>> LvThreadTasks;
        xp<Task<void>> LoSingleTaskPtr;
        while (true)
        {
            {
                constexpr auto BxNexusVoidThreadReady = []() {
                    return (
                        !SbDisabled
                        && (Nexus<void>::SbFinishTasks || Nexus<void>::SvTaskQueue.size())
                        && (RA::Threads::BxThreadsAreAvailable()));
                };
                auto Lock = SoMutex.CreateLock(BxNexusVoidThreadReady);

                if (SbDisabled == true)
                    continue;
                if (SvTaskQueue.empty())
                    return;

                ++SoThreads;
                ++SnInstTaskCount;

                if (!!SbGroupingOn && SvTaskQueue.size() >= (xint)RA::Threads::CPUThreads * 2)
                {
                    xint LvTaskCount = SvTaskQueue.size() / RA::Threads::CPUThreads;
                    LvTaskCount += SvTaskQueue.size() % RA::Threads::CPUThreads;
                    for (xint i = 0; i < LvTaskCount; i++)
                    {
                        LvThreadTasks.push_back(SvTaskQueue.front());
                        SvTaskQueue.pop();
                    }
                }
                else
                {
                    LoSingleTaskPtr = SvTaskQueue.front();
                    SvTaskQueue.pop();
                }
            }

            if (!!LoSingleTaskPtr)
            {
                (*LoSingleTaskPtr).RunTask();
                LoSingleTaskPtr = nullptr;
            }
            else
            {
                for (auto& LoTaskPtr : LvThreadTasks)
                {
                    Task<void>& VoidTask = LoTaskPtr.Get();
                    if (VoidTask.GetID() != 0
                        && SmMutex.contains(VoidTask.GetID())
                        && SmMutex.at(VoidTask.GetID()).Get().BxUseMutex())
                    {
                        auto Lock = SmMutex.at(VoidTask.GetID()).Get().CreateLock();
                        VoidTask.RunTask();
                    }
                    else // lock was given but the mutex was set to off
                        VoidTask.RunTask();
                }
                LvThreadTasks.clear();
            }

            --SoThreads;
            --SnInstTaskCount;
        }
    }
    catch (...)
    {
        throw "Error @ Nexus<void>::TaskLooper";
    }
};

RIN void Nexus<void>::Start()
{
    SbFinishTasks = false; // Used to exit when threading need to join for the last time;
    if (!SbInitialized)
    {
        SoBlankMutexPtr.Get().BxUnlocked();
        SmMutex.clear();
        SmMutex.insert({ SoBlankMutexPtr.Get().GetID(), SoBlankMutexPtr }); // starts size at 1 and index at 0

        SoThreads.IncInstanceCount();
        MvThreads.reserve(SoThreads.GetAllowedThreadCount());
        for (int i = 0; i < SoThreads.GetAllowedThreadCount(); ++i)
            MvThreads.emplace_back(std::bind((void(*)(int)) & Nexus<void>::TaskLooper, i)); // static member Function, don't use "this"

    }
    SbInitialized = true;
}

RIN int Nexus<void>::Stop()
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

RIN void Nexus<void>::ForceStop()
{
    MvThreads.clear();
    while (SvTaskQueue.size())
        SvTaskQueue.pop();
    SmMutex.clear();

    SbInitialized   = false;
    SbFinishTasks   = false;
    SnInstTaskCount = 0;
}

RIN void Nexus<void>::Disable()
{
    WaitAll();
    SbDisabled = true;
}

RIN void Nexus<void>::Enable()
{
    SbDisabled = false;
    SoMutex.SignalOne();
}

// ------------------------------------------------------------------------------------------------------------
// If you are meaning to get an object version, pass in "The" instead of "this"
template<typename F, typename ...A>
RIN UsingFunction(void) Nexus<void>::AddTask(F&& Function, A&& ...Args)
{
    auto BindedFunction = std::bind(std::move(Function), std::forward<A>(Args)...);
    if (SbDisabled)
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
    else 
    {
        auto Lock = SoMutex.CreateLock();
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
        SoMutex.SignalOne();
    }
}
template<typename F, typename ...A>
RIN UsingFunction(void) Nexus<void>::AddTaskRef(F&& Function, A&& ...Args)
{
    auto BindedFunction = std::bind(std::move(Function), std::ref(Args)...);
    if(SbDisabled)
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
    else
    {
        auto Lock = SoMutex.CreateLock();
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
        SoMutex.SignalOne();
    }
}
// ------------------------------------------------------------------------------------------------------------
template<typename O, typename F, typename ...A>
RIN UsingFunction(void) Nexus<void>::AddTask(O& object, F&& Function, A&& ...Args)
{
    auto BindedFunction = std::bind(std::move(Function), std::ref(object), std::forward<A>(Args)...);
    if (SbDisabled)
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
    else
    {
        auto Lock = SoMutex.CreateLock();
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
        SoMutex.SignalOne();
    }
}
template<typename O, typename F, typename ...A>
RIN UsingFunction(void) Nexus<void>::AddTaskRef(O& object, F&& Function, A&& ...Args)
{
    auto BindedFunction = std::bind(std::move(Function), std::ref(object), std::ref(Args)...);
    if (SbDisabled)
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
    else
    {
        auto Lock = SoMutex.CreateLock();
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
        SoMutex.SignalOne();
    }
}
// ------------------------------------------------------------------------------------------------------------
template <typename M, typename O, typename F, typename... A>
RIN UsingFunction(void) Nexus<void>::AddTask(M& FoMutex, O& object, F&& Function, A&& ...Args)
{
    auto BindedFunction = std::bind(std::move(Function), std::ref(object), std::forward<A>(Args)...);
    if (SbDisabled)
    {
        CheckClearMutexes();
        MutexTask(FoMutex, BindedFunction);
    }
    else
    {
        auto Lock = SoMutex.CreateLock();
        CheckClearMutexes();
        MutexTask(FoMutex, BindedFunction);
        SoMutex.SignalOne();
    }
    // nxm.id references the location in SmMutex
}
template <typename M, typename O, typename F, typename... A>
RIN UsingFunction(void) Nexus<void>::AddTaskRef(M& FoMutex, O& object, F&& Function, A&& ...Args)
{
    auto BindedFunction = std::bind(std::move(Function), std::ref(object), std::ref<A>(Args)...);
    if (SbDisabled)
    {
        CheckClearMutexes();
        MutexTask(FoMutex, BindedFunction);
    }
    else
    {
        auto Lock = SoMutex.CreateLock();
        CheckClearMutexes();
        MutexTask(FoMutex, BindedFunction);
        SoMutex.SignalOne();
    }
    // nxm.id references the location in SmMutex
}
template <typename M, typename F>
void Nexus<void>::MutexTask(M& FoMutex, F& FoBindedFunction)
{
    if (FoMutex == nullptr)
    {
        FoMutex = RA::MakeShared<RA::Mutex>();
        FoMutex->SetUseMutexOn();
    }

    if (SmMutex.size() <= FoMutex.Get().GetID()) // id should never be 'gt' size
    {
        if (SmMutex.find(FoMutex.Get().GetID()) != SmMutex.end()) // find not contains for NVCC
            SmMutex[FoMutex.Get().GetID()] = FoMutex;
        else
            SmMutex.insert({ FoMutex.Get().GetID(), FoMutex });
    }

    SvTaskQueue.emplace(RA::MakeShared<Task<void>>(FoMutex.Get().GetID(), std::move(FoBindedFunction)));
}
// ------------------------------------------------------------------------------------------------------------
template <typename F, typename V, typename... A>
RIN void Nexus<void>::AddTaskVal(F&& Function, V& element, A&& ...Args)
{
    auto BindedFunction = std::bind(std::move(Function), std::ref(element), std::ref(Args)...);
    if(SbDisabled)
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
    else
    {
        auto Lock = SoMutex.CreateLock();
        CheckClearMutexes();
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
        SoMutex.SignalOne();
    }
}

template<typename K, typename V, typename F, typename ...A>
RIN void Nexus<void>::AddTaskPair(F&& Function, K key, V& value, A&& ...Args)
{
    auto BindedFunction = std::bind(std::move(Function), std::ref(key), std::ref(value), std::ref(Args)...);
    if (SbDisabled)
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
    else
    {
        auto Lock = SoMutex.CreateLock();
        CheckClearMutexes();
        SvTaskQueue.emplace(RA::MakeShared<Task<void>>(0, std::move(BindedFunction)));
        SoMutex.SignalOne();
    }
}
// ------------------------------------------------------------------------------------------------------------
// Class required due to Linux (not Windows)
// Task<short int> Nexus<void>::GetWithoutProtection(xint dummy) noexcept { return Task<short int>(); }

RIN void Nexus<void>::WaitAll()
{
    while (SvTaskQueue.size() || SnInstTaskCount > 0)
    {
        if (SnInstTaskCount == 0 && SvTaskQueue.size())
            SoMutex.SignalOne();
        Nexus<void>::Sleep(1);
    }
}

RIN void Nexus<void>::Clear()
{
    Nexus<void>::WaitAll();

    SnInstTaskCount = 0;
    if (SoThreads.GetUsedThreadCount() == 0 && SvTaskQueue.size() == 0)
    {
        SmMutex.clear();
        SmMutex.insert({ SoBlankMutexPtr.Get().GetID(), SoBlankMutexPtr });
    }
    RA::Mutex::ResetTotalMutexCount();
}

RIN void Nexus<void>::CheckClearMutexes()
{
    if (SoThreads.GetUsedThreadCount() == 0 && SvTaskQueue.size() == 0)
    {
        SmMutex.clear();
        SmMutex.insert({ SoBlankMutexPtr.Get().GetID(), SoBlankMutexPtr });
    }
}

RIN void Nexus<void>::Sleep(unsigned int FnMilliseconds)
{
#ifdef BxWindows
    ::Sleep(FnMilliseconds);
#else
    ::usleep(FnMilliseconds);
#endif
}

