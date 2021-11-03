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

#include "Atomic.h"
#include "Threads.h"
#include "Mutex.h"
#include "Task.h"
#include "Job.h"

#define Sync() Nexus<>::WaitAll()
#define EXIT() exit(Nexus<>::Stop())

template<typename M>
using IsSharedPtr = std::enable_if<std::is_same<M, RA::SharedPtr<RA::Mutex>>::value || std::is_same<M, xptr<RA::Mutex>>::value, void>::type;

// =========================================================================================

template<typename T> class Nexus;

// WARNING: MAKE SURE YOU ONLY USE FUNCTIONS THAT TAKE L-VALUE REFERENCE
template<>
class __single_inheritance Nexus<void> : public RA::Threads
{
private:
    inline static RA::Mutex SoMutex;

    inline static bool SbInitialized = false;
    inline static std::atomic<bool> SbFinishTasks = false;
    inline static RA::Atomic<long long int> SnInstTaskCount = 0;
    
    inline static std::vector<std::thread> MvThreads; // these threads start in the constructor and don't stop until Nexus is over
    inline static std::queue<RA::SharedPtr<Task<void>>> ScTaskQueue; // This is where tasks are held before being run
    // No Getter Mutex/Sig for Nexus<void> because you pass in by ref, you don't pull any stored values
    inline static std::unordered_map<size_t, RA::SharedPtr<RA::Mutex>> SmMutex; // for objects in threads
    inline static RA::SharedPtr<RA::Mutex> SoBlankMutexPtr = RA::MakeShared<RA::Mutex>(); // This blank Mutex used when no mutex is given as a param
    // Nexus<void> can't store output, hance no storage containers here

    static void TaskLooper(int thread_idx);

public:
    Nexus();
    virtual ~Nexus();
    static void Start();
    static int  Stop();

    static void SetMutexOn(size_t FoMutex); 
    static void SetMutexOff(size_t FoMutex);
    // ------------------------------------------------------------------------------------------------------------------------------------------------

#define IsFunction(__TempObj__) std::is_function<typename std::remove_pointer<__TempObj__>::type>::value
#define IsClass(__TempObj__)    std::is_class<__TempObj__>::value
#define IsMutexPtr(__TempObj__) std::is_same<__TempObj__, RA::SharedPtr<RA::Mutex>>::value

    // Job: Function + Args

    template <typename F>
    static inline void
        AddJob(F&& Function);
    template <typename F, typename... A>
    static inline typename std::enable_if<IsFunction(F) && !IsClass(F) && !IsMutexPtr(F), void>::type
        AddJob(F&& Function, A&& ... Args);

    // Job Object.Function + Args
    template <typename O, typename F, typename... A>
    static inline typename std::enable_if<IsClass(O) && !IsFunction(O) && !IsMutexPtr(O), void>::type
        AddJob(O& object, F&& Function, A&& ... Args);
    template <typename O, typename F>
    static inline typename std::enable_if<IsClass(O) && !IsFunction(O) && !IsMutexPtr(O), void>::type
        AddJob(O& object, F&& Function);

    //  Job: Mutex + Object.Function + Args
    template <typename M, typename O, typename F, typename... A>
    static inline typename std::enable_if<IsMutexPtr(M) && !IsFunction(M), void>::type
        AddJob(M& FoMutex, O& object, F&& Function, A&& ... Args);
    // ------------------------------------------------------------------------------------------------------------------------------------------------
    // Job: Function + Ref-Arg + Args
    template <typename F, typename V, typename... A>
    static inline void
        AddJobVal(F&& Function, V& element, A&&... Args);

    // Job: Function + Ref-(Key/Value) + Args
    template <typename K, typename V, typename F, typename... A>
    static void AddJobPair(F&& Function, K key, V& value, A&& ... Args);

    // Required due to Linux not Windows (returns void Job<T>)
    // static Job<short int> GetWithoutProtection(size_t dummy) noexcept;

    static size_t Size();

    static void WaitAll();
    static bool TaskCompleted();
    static void Clear();
    static void CheckClearMutexes();

    static void Sleep(unsigned int FnMilliseconds);
};

// =========================================================================================

inline void Nexus<void>::TaskLooper(int thread_idx)
{
    std::atomic<size_t> LnLastCount = 0;
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

            TaskPtr  = ScTaskQueue.front();
            MutexIdx = ScTaskQueue.front().Get().GetMutexID();
            ScTaskQueue.pop();
            SnInstTaskCount++;
        }

        if (!MutexIdx) // no lock given
            TaskPtr.Get()();
        else if (SmMutex.at(MutexIdx).Get().IsMutexOn()) // lock was given with a mutex set to on
        {
            auto Lock = SmMutex.at(MutexIdx).Get().CreateLock();
            TaskPtr.Get()();
        }
        else // lock was given but the mutex was set to off
            TaskPtr.Get()();

        auto Lock = SoMutex.CreateLock();
        if (SbFinishTasks && !ScTaskQueue.size())
        {
            SnInstTaskCount = 0;
            return;
        }
        SnInstTaskCount--;
    }
};

inline Nexus<void>::Nexus(){
    Nexus<void>::Start();
}

inline Nexus<void>::~Nexus(){
}

inline void Nexus<void>::Start()
{
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

inline int Nexus<void>::Stop()
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
    return 0;
}

inline void Nexus<void>::SetMutexOn(size_t FoMutex)
{
    SmMutex.at(FoMutex).Get().SetMutexOn();
}

inline void Nexus<void>::SetMutexOff(size_t FoMutex)
{
    SmMutex.at(FoMutex).Get().SetMutexOff();
}

//template <typename F, typename... A>
//inline typename std::enable_if<
//    std::is_function<F>::value
//    && !std::is_class<F>::value
//    && !std::is_same<F, RA::SharedPtr<RA::Mutex>>::value
//    && !std::is_pointer<F>::value, void>::type
//Nexus<void>::AddJob(F&& Function, A&&... Args)
//{
//    auto Lock = SoMutex.CreateLock();
//    auto BindedFunction = std::bind(std::move(Function), std::forward<A>(Args)...);
//    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), 0));
//}

template<typename F>
inline typename void
Nexus<void>::AddJob(F&& Function)
{
    auto Lock = SoMutex.CreateLock();
    auto BindedFunction = std::bind(std::move(Function));
    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), 0));
}

template<typename F, typename ...A>
inline typename typename std::enable_if<IsFunction(F) && !IsClass(F) && !IsMutexPtr(F), void>::type
Nexus<void>::AddJob(F&& Function, A&& ...Args)
{
    auto Lock = SoMutex.CreateLock();
    auto BindedFunction = std::bind(std::move(Function), std::forward<A>(Args)...);
    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), 0));
}

template<typename O, typename F, typename ...A>
inline typename std::enable_if<IsClass(O) && !IsFunction(O) && !IsMutexPtr(O), void>::type
Nexus<void>::AddJob(O& object, F&& Function, A&& ...Args)
{
    auto Lock = SoMutex.CreateLock();
    auto BindedFunction = std::bind(std::move(Function), std::ref(object), std::forward<A>(Args)...);
    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), SoBlankMutexPtr.Get().GetID()));
}

template<typename O, typename F>
inline typename std::enable_if<IsClass(O) && !IsFunction(O) && !IsMutexPtr(O), void>::type
Nexus<void>::AddJob(O& object, F&& Function)
{
    auto Lock = SoMutex.CreateLock();
    auto BindedFunction = std::bind(std::move(Function), std::ref(object));
    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), SoBlankMutexPtr.Get().GetID()));
}

template <typename M, typename O, typename F, typename... A>
static inline typename std::enable_if<IsMutexPtr(M) && !IsFunction(M), void>::type
Nexus<void>::AddJob(M& FoMutex, O& object, F&& Function, A&& ...Args)
{
    auto Lock = SoMutex.CreateLock();
    CheckClearMutexes();
    auto BindedFunction = std::bind(std::move(Function), std::ref(object), std::forward<A>(Args)...);

    if (!FoMutex)
        FoMutex = RA::MakeShared<RA::Mutex>();

    if (SmMutex.size() <= FoMutex.Get().GetID()) // id should never be 'gt' size
    {
        if (SmMutex.contains(FoMutex.Get().GetID()))
            SmMutex[FoMutex.Get().GetID()] = FoMutex;
        else
            SmMutex.insert({ FoMutex.Get().GetID(), FoMutex });
    }

    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), FoMutex.Get().GetID()));
    // nxm.id references the location in SmMutex
}

template <typename F, typename V, typename... A>
static inline void
    Nexus<void>::AddJobVal(F&& Function, V& element, A&&... Args)
{
    auto Lock = SoMutex.CreateLock();
    CheckClearMutexes();
    auto BindedFunction = std::bind(std::move(Function), std::ref(element), std::ref(Args)...);
    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), 0));
}

template<typename K, typename V, typename F, typename ...A>
inline void Nexus<void>::AddJobPair(F&& Function, K key, V& value, A&& ...Args)
{
    auto Lock = SoMutex.CreateLock();
    CheckClearMutexes();
    auto BindedFunction = std::bind(std::move(Function), std::ref(key), std::ref(value), std::ref(Args)...);
    ScTaskQueue.emplace(RA::MakeShared<Task<void>>(std::move(BindedFunction), 0));
}

// Class required due to Linux (not Windows)
// inline Job<short int> Nexus<void>::GetWithoutProtection(size_t dummy) noexcept { return Job<short int>(); }

inline size_t Nexus<void>::Size(){
    return SnInstTaskCount;
}

inline void Nexus<void>::WaitAll()
{
    while (ScTaskQueue.size() || SnInstTaskCount > 0)
    {
        Nexus<void>::Sleep(1);
    }
}

inline bool Nexus<void>::TaskCompleted()
{
    return !ScTaskQueue.size();
}

inline void Nexus<void>::Clear() 
{
    Nexus<void>::WaitAll();

    SnInstTaskCount = 0;
    if (RA::Threads::Used == 0 && ScTaskQueue.size() == 0)
    {
        SmMutex.clear();
        SmMutex.insert({ SoBlankMutexPtr.Get().GetID(), SoBlankMutexPtr});
    }
    RA::Mutex::ResetTotalMutexCount();
}

inline void Nexus<void>::CheckClearMutexes()
{
    if (RA::Threads::Used == 0 && ScTaskQueue.size() == 0)
    {
        SmMutex.clear();
        SmMutex.insert({ SoBlankMutexPtr.Get().GetID(), SoBlankMutexPtr });
    }
}

inline void Nexus<void>::Sleep(unsigned int FnMilliseconds)
{
    #if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
        ::Sleep(FnMilliseconds);
    #else
        ::usleep(FnMilliseconds);
    #endif
}

