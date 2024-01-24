#pragma once

#include "RawMapping.h"
#ifdef BxWindows
#include<Windows.h>
#else 
#include <unistd.h>
#endif

#include<iostream>
#include<vector>
#include<unordered_map>
#include<initializer_list>
#include<utility>

#include<thread>
#include<mutex>
#include<condition_variable>

#include<deque>
#include<stack>
#include<functional>
#include<type_traits>

#include "Atomic.h"
#include "SharedPtr.h"
#include "Threads.h"

#include "Task/TaskVoid.h"
#include "Task/TaskValue.h"
#include "Task/TaskXP.h"
#include "Task/TaskFundamental.h"

// =========================================================================================

template<typename T>
class Nexus : public RA::Threads
{
private:
    RA::Mutex MoThreadMtx;

    istatic RA::Threads SoThreads;

    bool MbDisabled = false;
    RA::Atomic<bool>          MbFinishTasks = false;
    RA::Atomic<long long int> MnInstTaskCount = 0;
    xint                      MnNextTaskID = 0;
    bool                      MbSetGroupingOn = false;

    std::vector<std::thread>  MvThreads;     // these threads start in the constructor and don't stop until Nexus is over
    std::deque<xp<Task<T>>>   MvTaskDeque;   // The is where tasks are held before being run
                                            // deque chosen over queue because the deque has an iterator

    // These values are used for pulling outpu tdata from your jobs.
    std::unordered_map<std::string, xp<Task<T>>> MmStrTask;
    std::unordered_map<xint,        xp<Task<T>>> MmIdxTask;

    RIN void TaskLooper(int thread_idx);

    template<typename F, typename ...A> RIN void Add(   F& function, A&& ...Args);
    template<typename F, typename ...A> RIN void AddRef(F& function, A&& ...Args);

    template<typename F, typename ...A> RIN void AddKVP(   const std::string& key, F& function, A&& ...Args);
    template<typename F, typename ...A> RIN void AddKVPRef(const std::string& key, F& function, A&& ...Args);

public:
    typedef T value_type;

    RIN Nexus();
    RIN ~Nexus();

    RIN void Disable();
    RIN void Enable();

    template <typename F, typename ...A> RIN UsingFunction(void) AddTask(   const std::string& key, F&& function, A&& ...Args);
    template <typename F, typename ...A> RIN UsingFunction(void) AddTaskRef(const std::string& key, F&& function, A&& ...Args);

    template <typename F, typename ...A> RIN UsingFunction(void) AddTask(   const char* key, F&& function, A&& ...Args);
    template <typename F, typename ...A> RIN UsingFunction(void) AddTaskRef(const char* key, F&& function, A&& ...Args);

    template <typename F, typename ...A> RIN UsingFunction(void) AddTask(   F&& function, A&& ...Args);
    template <typename F, typename ...A> RIN UsingFunction(void) AddTaskRef(F&& function, A&& ...Args);

    // These are to be used by xvector/xmap
    template <typename F, typename ONE, typename ...A>           RIN void AddTaskVal( F&& function, ONE& element,     A&& ...Args); // commonly used by xvector
    template <typename K, typename V, typename F, typename ...A> RIN void AddTaskPair(F&& function, K& key, V& value, A&& ...Args); // commonly used by xmap

    // Getters can't be const due to the mutex
    RIN Task<T>& operator()(const std::string& val);
    RIN Task<T>& operator()(const xint val);

    // Getters can't be const due to the mutex
private:
    template<typename O, typename I> RIN auto ContGetAllPtrs(); // auto because of Nexus<T> and Nexus<xp<T>>
public:
    RIN std::vector<T> GetAll();
    RIN auto           GetAllPtrs(); // auto because of Nexus<T> and Nexus<xp<T>>
    RIN std::vector<T> GetMoveAllIndices();
    RIN Task<T>& Get(const std::string& val);
    RIN Task<T>& Get(const xint val);
    RIN Task<T>& Get(const char* Input);
    RIN Task<T>& GetWithoutProtection(const xint val) noexcept;

    RIN xint Size() const;
    RIN void WaitAll();
    RIN bool TaskCompleted() const;
    RIN void Clear();
    RIN void Sleep(unsigned int extent) const;
};

// =========================================================================================

template<typename T>
RIN void Nexus<T>::TaskLooper(int thread_idx)
{
    try
    {
        std::unordered_map<xint, xp<Task<T>>> LmLocalMap;
        xp<Task<T>> LoSingleTaskPtr;
        while (true)
        {
            {
                const auto LoThreadLock = MoThreadMtx.CreateLock([this]() {
                    return (
                        !MbDisabled
                        && (MbFinishTasks || MvTaskDeque.size())
                        && RA::Threads::BxThreadsAreAvailable()); });

                if (MbDisabled)
                    continue;
                if (MvTaskDeque.empty())
                    return;

                ++SoThreads;
                ++MnInstTaskCount;

                xint LvTaskCount = 0;
                
                if (MbSetGroupingOn && MvTaskDeque.size() >= RA::Threads::CPUThreads * 2)
                {
                    LvTaskCount = MvTaskDeque.size() / RA::Threads::CPUThreads;
                    LvTaskCount += MvTaskDeque.size() % RA::Threads::CPUThreads;
                    for (xint i = 0; i < LvTaskCount; i++)
                    {
                        auto  LoTaskPtr = The.MvTaskDeque.front();
                        auto& LoTask = *LoTaskPtr;
                        MvTaskDeque.pop_front();
                        MmIdxTask.insert({ LoTask.GetID(), LoTaskPtr });
                        LmLocalMap.insert({ LoTask.GetID(), LoTaskPtr }); // for safe mem managment
                        if (!!LoTask.GetNamePtr())
                            MmStrTask.insert({ LoTask.GetName(), LoTaskPtr });
                    }
                }
                else
                {
                    LoSingleTaskPtr = The.MvTaskDeque.front();
                    auto& LoTask = *LoSingleTaskPtr;
                    MvTaskDeque.pop_front();
                    MmIdxTask.insert({ LoTask.GetID(), LoSingleTaskPtr });
                    if (!!LoTask.GetNamePtr())
                        MmStrTask.insert({ LoTask.GetName(), LoSingleTaskPtr });
                }
            }

            if (!!LoSingleTaskPtr)
            {
                (*LoSingleTaskPtr).RunTask();
                LoSingleTaskPtr = nullptr;
            }
            else
            {
                for (auto& Pair : LmLocalMap)
                    Pair.second.Get().RunTask();
                LmLocalMap.clear();
            }

            --SoThreads;
            --MnInstTaskCount;
        }
    }
    catch (...)
    {
        throw "Error @ Nexus<T>::TaskLooper";
    }
}

template<typename T>
template<typename F, typename ...A>
RIN void Nexus<T>::Add(F& function, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::forward<A>(Args)...);
    if (MbDisabled)
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction)));
    else
    {
        const auto Lock = MoThreadMtx.CreateLock();
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction)));
        MoThreadMtx.SignalOne();
    }
}

template<typename T>
template<typename F, typename ...A>
RIN void Nexus<T>::AddRef(F& function, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::ref(Args)...);
    if (MbDisabled)
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction)));
    else
    {
        const auto Lock = MoThreadMtx.CreateLock();
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction)));
        MoThreadMtx.SignalOne();
    }
}

template<typename T>
template<typename F, typename ...A>
RIN void Nexus<T>::AddKVP(const std::string& key, F& function, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::forward<A>(Args)...);
    if (MbDisabled)
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, key, std::move(BindedFunction)));
    else
    {
        const auto Lock = MoThreadMtx.CreateLock();
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, key, std::move(BindedFunction)));
        MoThreadMtx.SignalOne();
    }
}

template<typename T>
template<typename F, typename ...A>
RIN void Nexus<T>::AddKVPRef(const std::string& key, F& function, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::ref(Args)...);
    if (MbDisabled)
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, key, std::move(BindedFunction)));
    else
    {
        const auto Lock = MoThreadMtx.CreateLock();
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, key, std::move(BindedFunction)));
        MoThreadMtx.SignalOne();
    }
}

// ------------------------------------------------------------------------------------------
template<typename T>
RIN Nexus<T>::Nexus()
{
    RA::Threads::InstanceCount++;
    MvThreads.reserve(RA::Threads::Allowed);
    for (int i = 0; i < RA::Threads::Allowed; ++i)
        MvThreads.emplace_back(std::bind(&Nexus<T>::TaskLooper, std::ref(*this), i));
    RA::Threads::Used = 0;
}

template<typename T>
RIN Nexus<T>::~Nexus()
{
    MbDisabled = false;
    The.WaitAll();
    MbFinishTasks = true;
    MoThreadMtx.UnlockAll();
    for (auto& t : MvThreads) 
        t.join();
}

template<typename T>
RIN void Nexus<T>::Disable()
{
    WaitAll();
    MbDisabled = true;
}

template<typename T>
RIN void Nexus<T>::Enable()
{
    MbDisabled = false;
    MoThreadMtx.SignalOne();
}

// ------------------------------------------------------------------------------------------
template<typename T>
template<typename F, typename ...A>
RIN UsingFunction(void) Nexus<T>::AddTask(const std::string& key, F&& function, A&& ...Args)
{
    The.AddKVP(key, function, std::forward<A>(Args)...);
}
template<typename T>
template<typename F, typename ...A>
RIN UsingFunction(void) Nexus<T>::AddTaskRef(const std::string& key, F&& function, A&& ...Args)
{
    The.AddKVPRef(key, function, std::ref(Args)...);
}
// ------------------------------------------------------------------------------------------
template<typename T>
template<typename F, typename ...A>
RIN UsingFunction(void) Nexus<T>::AddTask(const char* key, F&& function, A&& ...Args)
{
    The.AddKVP(std::string(key), function, std::forward<A>(Args)...);
}
template<typename T>
template<typename F, typename ...A>
RIN UsingFunction(void) Nexus<T>::AddTaskRef(const char* key, F&& function, A&& ...Args)
{
    The.AddKVPRef(std::string(key), function, std::ref(Args)...);
}
// ------------------------------------------------------------------------------------------
template<typename T>
template <typename F, typename ...A>
RIN UsingFunction(void) Nexus<T>::AddTask(F&& function, A&& ...Args)
{
    The.Add(function, std::forward<A>(Args)...);
}
template<typename T>
template <typename F, typename ...A>
RIN UsingFunction(void) Nexus<T>::AddTaskRef(F&& function, A&& ...Args)
{
    The.AddRef(function, std::ref(Args)...);
}
// ------------------------------------------------------------------------------------------
//template<typename T>
//template <typename O, typename F, typename... A>
//RIN UsingFunction(void) Nexus<T>::AddTask(O& object, F&& Function, A&& ... Args)
//{
//
//}

template<typename T>
template <typename F, typename ONE, typename ...A>
RIN void Nexus<T>::AddTaskVal(F&& function, ONE& element, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::ref(element), std::ref(Args)...);
    if (MbDisabled)
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction)));
    else
    {
        auto Lock = MoThreadMtx.CreateLock();
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction)));
        MoThreadMtx.SignalOne();
    }
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
RIN void Nexus<T>::AddTaskPair(F&& function, K& key, V& value, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::ref(key), std::ref(value), std::ref(Args)...);
    if (MbDisabled)
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction)));
    else
    {
        auto Lock = MoThreadMtx.CreateLock();
        MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID,std::move(BindedFunction)));
        MoThreadMtx.SignalOne();
    }
}
// ------------------------------------------------------------------------------------------

template<typename T>
RIN Task<T>& Nexus<T>::operator()(const std::string& Input)
{
    // If it is not already cached, is it queued?
    if (!MmStrTask.count(Input))
        throw "Nexus Key(str) Not Found!";

    Task<T>& LoTask = MmStrTask[Input].Get();
    if (!LoTask.IsDone())
    {
        while (LoTask.IsDone() == false)
            Sleep(1);
    }

    LoTask.TestException();
    return LoTask;
}


template<typename T>
RIN Task<T>& Nexus<T>::operator()(const xint Input)
{
    if (!MmIdxTask.count(Input))
        throw std::runtime_error("Nexus Key(int) Not Found!");

    Task<T>& LoTask = MmIdxTask[Input].Get();
    if (!LoTask.IsDone())
    {
        while (LoTask.IsDone() == false)
            Sleep(1);
    }

    LoTask.TestException();
    return LoTask;
}

template<typename T>
template<typename O, typename I>
RIN auto Nexus<T>::ContGetAllPtrs() // Output / Input
{
    auto Captures = std::vector<O>(); // T or xp<T>
    for (std::pair<const xint, xp<Task<I>>>& TaskPair : MmIdxTask)
    {
        auto& LoTask = *TaskPair.second;
        LoTask.TestException();
        Captures.push_back(LoTask.GetValuePtr());
    }
    if (MmStrTask.size())
    {
        for (std::pair<const std::string, xp<Task<I>>>& TaskPair : MmStrTask)
        {
            auto& LoTask = TaskPair.second.Get();
            if (!LoTask.BxRemoved())
            {
                try {
                    LoTask.TestException();
                }
                catch (...) {
                    throw "Nexus Task Exception Thrown\n";
                }
                Captures.push_back(LoTask.GetValuePtr());
            }
        }
    }

    Clear();
    return Captures;
}

template<typename T>
RIN std::vector<T> Nexus<T>::GetAll()
{
    The.WaitAll();
    auto Lock = MoThreadMtx.CreateLock();
    std::vector<T> Captures;
    for (std::pair<const xint, xp<Task<T>>>& TaskPair : MmIdxTask)
    {
        auto& LoTask = *TaskPair.second;
        LoTask.TestException();
        Captures.push_back(LoTask.GetValue());
    }
    if (MmStrTask.size())
    {
        for (std::pair<const std::string, xp<Task<T>>>& TaskPair : MmStrTask)
        {
            auto& LoTask = TaskPair.second.Get();
            if (!LoTask.BxRemoved())
            {
                LoTask.TestException();
                Captures.push_back(LoTask.GetValue());
            }
        }
    }
    Clear();
    return Captures;
}

template<typename T>
RIN auto Nexus<T>::GetAllPtrs()
{
    The.WaitAll();
    auto Lock = MoThreadMtx.CreateLock();

    if constexpr (IsSharedPtr(T)){
        return ContGetAllPtrs<T, T>();
    }
    else{
        return ContGetAllPtrs<xp<T>, T>();
    }
}

template<typename T>
RIN std::vector<T> Nexus<T>::GetMoveAllIndices()
{
    The.WaitAll();
    auto Lock = MoThreadMtx.CreateLock();
    std::vector<T> Captures;
    Captures.reserve(Size());
    for (auto& Target : MmIdxTask)
    {
        auto& LoTask = Target.second.Get();
        try{
            LoTask.TestException();
        }
        catch (...){
            throw "Nexus Task Exception Thrown\n";
        }
        if constexpr (IsFundamental(T))
            Captures.push_back(LoTask.GetValue());
        else
            Captures.push_back(std::move(LoTask.GetValue()));
    }
    Clear();
    return Captures;
}

template<typename T>
RIN Task<T>& Nexus<T>::Get(const std::string& Input) {
    return The.operator()(Input);
}

template<typename T>
RIN Task<T>& Nexus<T>::Get(const char* Input) {
    return The.operator()(std::string(Input));
}

template<typename T>
RIN Task<T>& Nexus<T>::GetWithoutProtection(const xint val) noexcept{
    return *MmIdxTask[val];
}

template<typename T>
RIN Task<T>& Nexus<T>::Get(const xint Input) {
    return The.operator()(Input);
}


// ------------------------------------------------------------------------------------------

template<typename T>
RIN xint Nexus<T>::Size() const
{
    return MmIdxTask.size();
}

template<typename T>
RIN void Nexus<T>::WaitAll()
{
    while (MvTaskDeque.size() || MnInstTaskCount > 0)
    {
        if (MvTaskDeque.size() && MnInstTaskCount == 0)
            MoThreadMtx.SignalOne();
        Nexus<T>::Sleep(1);
    }
}

template<typename T>
RIN bool Nexus<T>::TaskCompleted() const
{
    return !MvTaskDeque.size();
}

template<typename T>
RIN void Nexus<T>::Clear()
{

    if (MnInstTaskCount == 0 && MvTaskDeque.size() == 0)
    {
        MmIdxTask.clear();
        MmStrTask.clear();
        MnInstTaskCount = 0;
        MnNextTaskID = 0;
    }
}

template<typename T>
RIN void Nexus<T>::Sleep(unsigned int extent) const
{
#ifdef BxWindows
    ::Sleep(extent);
#else
    ::usleep(extent);
#endif
}
