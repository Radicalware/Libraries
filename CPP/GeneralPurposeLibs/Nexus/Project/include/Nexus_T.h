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

#include "RawMapping.h"
#include "Atomic.h"
#include "SharedPtr.h"
#include "Threads.h"
#include "Task.h"
#include "Job.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include<Windows.h>
#else 
#include<unistd.h>
#endif

// =========================================================================================

template<typename T>
class __single_inheritance Nexus : public RA::Threads
{
private:
    RA::Mutex MoMutex;

    std::atomic<bool>         MbFinishTasks = false;
    RA::Atomic<long long int> MnInstTaskCount = 0;
    xint                      MnNextTaskID = 0;

    std::vector<std::thread>           MvThreads;     // these threads start in the constructor and don't stop until Nexus is over
    std::deque<RA::SharedPtr<Task<T>>> MvTaskDeque;   // The is where tasks are held before being run
                                            // deque chosen over queue because the deque has an iterator

    // These values are used for pulling outpu tdata from your jobs.
    std::unordered_map<std::string, RA::SharedPtr<Job<T>>> MmStrJob;
    std::unordered_map<xint, RA::SharedPtr<Job<T>>>        MmIdxJob;

    INL void TaskLooper(int thread_idx);

    template <typename F, typename ...A>
    INL void Add(F& function, A&& ...Args);

    template<typename F, typename ...A>
    INL void AddKVP(const std::string& key, F& function, A&& ...Args);

public:
    typedef T value_type;

    Nexus();
    ~Nexus();

    template <typename F, typename ...A>
    INL UsingFunction(void) AddJob(const std::string& key, F&& function, A&& ...Args);

    template <typename F, typename ...A>
    INL UsingFunction(void) AddJob(const char* key, F&& function, A&& ...Args);

    template <typename F, typename ...A>
    INL UsingFunction(void) AddJob(F&& function, A&& ...Args);

    //template <typename O, typename F, typename... A>
    //INL UsingFunction(void) AddJob(O& object, F&& Function, A&& ... Args);

    // These are to be used by xvector/xmap
    template <typename F, typename ONE, typename ...A>
    INL void AddJobVal(F&& function, ONE& element, A&& ...Args); // commonly used by xvector

    template <typename K, typename V, typename F, typename ...A>
    INL void AddJobPair(F&& function, K& key, V& value, A&& ...Args); // commonly used by xmap

    // Getters can't be const due to the mutex
    INL Job<T>& operator()(const std::string& val);
    INL Job<T>& operator()(const xint val);

    // Getters can't be const due to the mutex
    INL std::vector<T> GetAll();
    INL std::vector<T> GetMoveAllIndices();
    INL Job<T>& Get(const std::string& val);
    INL Job<T>& Get(const xint val);
    INL Job<T>& Get(const char* Input);
    INL Job<T>& GetWithoutProtection(const xint val) noexcept;

    INL xint Size() const;
    INL void WaitAll();
    INL bool TaskCompleted() const;
    INL void Clear();
    INL void Sleep(unsigned int extent) const;
};

// =========================================================================================

template<typename T>
INL void Nexus<T>::TaskLooper(int thread_idx)
{
    int LnAtomicIdx = 0;
    std::list<xint> LvTasks;
    while (true) 
    {
        xint LnTaskIdx = 0;
        {
            auto Lock = MoMutex.CreateLock([this]() {
                return ((MbFinishTasks || MvTaskDeque.size()) && RA::Threads::GetAllowedThreadCount() > MnInstTaskCount); });

            if (MvTaskDeque.empty())
                return;

            xint LvTaskCount = MvTaskDeque.size() / RA::Threads::CPUThreads;
            LvTaskCount += MvTaskDeque.size() % RA::Threads::CPUThreads;

            for (xint i = 0; i < LvTaskCount; i++)
            {
                LnTaskIdx = The.MvTaskDeque.front().Get().GetID();
                auto JobPtr = RA::MakeShared<Job<T>>(MvTaskDeque.front(), RA::Threads::TotalTasksCounted);
                MvTaskDeque.pop_front();
                MmIdxJob.insert({ LnTaskIdx, JobPtr });

                if (JobPtr.Get().GetTask().HasName())
                    MmStrJob.insert({ JobPtr.Get().GetTask().GetName(), JobPtr });

                ++MnInstTaskCount;
                LvTasks.push_back(LnTaskIdx);
            }
        }

        for (auto& LnID : LvTasks)
            MmIdxJob.at(LnID).Get().Run();

        const auto LnTasksCompletionCount = LvTasks.size();
        LvTasks.clear();

        auto Lock = MoMutex.CreateLock();
        MnInstTaskCount -= LnTasksCompletionCount;
        if (MbFinishTasks && !MvTaskDeque.size() && !MnInstTaskCount)
        {
            MnInstTaskCount = 0;
            return;
        }
    }
}

template<typename T>
template<typename F, typename ...A>
INL void Nexus<T>::Add(F& function, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::ref(Args)...);
    auto Lock = MoMutex.CreateLock();
    MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction)));
}

template<typename T>
template<typename F, typename ...A>
INL void Nexus<T>::AddKVP(const std::string& key, F& function, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::ref(Args)...);
    auto Lock = MoMutex.CreateLock();
    MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction), key));
}
// ------------------------------------------------------------------------------------------
template<typename T>
INL Nexus<T>::Nexus()
{
    RA::Threads::InstanceCount++;
    MvThreads.reserve(RA::Threads::Allowed);
    for (int i = 0; i < RA::Threads::Allowed; ++i)
        MvThreads.emplace_back(std::bind(&Nexus<T>::TaskLooper, std::ref(*this), i));
    RA::Threads::Used = 0;
}

template<typename T>
INL Nexus<T>::~Nexus()
{
    The.WaitAll();
    MbFinishTasks = true;
    MoMutex.UnlockAll();
    for (auto& t : MvThreads) 
        t.join();
}


// ------------------------------------------------------------------------------------------
template<typename T>
template<typename F, typename ...A>
INL UsingFunction(void) Nexus<T>::AddJob(const std::string& key, F&& function, A&& ...Args)
{
    The.AddKVP(key, function, std::ref(Args)...);
}

template<typename T>
template<typename F, typename ...A>
INL UsingFunction(void) Nexus<T>::AddJob(const char* key, F&& function, A&& ...Args)
{
    The.AddKVP(std::string(key), function, std::ref(Args)...);
}

template<typename T>
template <typename F, typename ...A>
INL UsingFunction(void) Nexus<T>::AddJob(F&& function, A&& ...Args)
{
    The.Add(function, std::ref(Args)...);
}

//template<typename T>
//template <typename O, typename F, typename... A>
//INL UsingFunction(void) Nexus<T>::AddJob(O& object, F&& Function, A&& ... Args)
//{
//
//}

template<typename T>
template <typename F, typename ONE, typename ...A>
INL void Nexus<T>::AddJobVal(F&& function, ONE& element, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::ref(element), Args...);
    auto Lock = MoMutex.CreateLock();
    MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction)));
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
INL void Nexus<T>::AddJobPair(F&& function, K& key, V& value, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::ref(key), std::ref(value), Args...);
    auto Lock = MoMutex.CreateLock();
    MvTaskDeque.emplace_back(RA::MakeShared<Task<T>>(++MnNextTaskID, std::move(BindedFunction)));
}
// ------------------------------------------------------------------------------------------

template<typename T>
INL Job<T>& Nexus<T>::operator()(const std::string& Input)
{
    // If it is not already cached, is it queued?
    if (!MmStrJob.count(Input))
        throw "Nexus Key(str) Not Found!";

    auto Lock = MoMutex.CreateLock([this, &Input]()
        { return MmStrJob.count(Input) && MmStrJob[Input].Get().IsDone(); });
    auto& Target = MmStrJob[Input].Get();
    Target.TestException();
    return Target;
}


template<typename T>
INL Job<T>& Nexus<T>::operator()(const xint Input)
{
    if (!MmIdxJob.count(Input))
        throw std::runtime_error("Nexus Key(int) Not Found!");

    auto Lock = MoMutex.CreateLock([this, &Input]() 
        { return MmIdxJob.count(Input) && MmIdxJob[Input].Get().IsDone(); });
    auto& Target = MmIdxJob[Input].Get();
    Target.TestException();
    return Target;
}

template<typename T>
INL std::vector<T> Nexus<T>::GetAll()
{
    The.WaitAll();
    auto Lock = MoMutex.CreateLock();
    std::vector<T> Captures;
    for (std::pair<const xint, RA::SharedPtr<Job<T>>>& Target : MmIdxJob)
    {
        Target.second.Get().TestException();
        Captures.push_back(Target.second.Get().Move());
    }
    for (std::pair<const std::string, RA::SharedPtr<Job<T>>>& Target : MmStrJob)
    {
        auto& LoJob = Target.second.Get();
        if (!LoJob.BxRemoved())
        {
            LoJob.TestException();
            Captures.push_back(LoJob.Move());
        }
    }
    Clear();
    return Captures;
}

template<typename T>
INL std::vector<T> Nexus<T>::GetMoveAllIndices()
{
    The.WaitAll();
    auto Lock = MoMutex.CreateLock();
    std::vector<T> Captures;
    Captures.reserve(Size());
    for (std::pair<const xint, RA::SharedPtr<Job<T>>>& Target : MmIdxJob)
    {
        Target.second.Get().TestException();
        Captures.push_back(std::move(Target.second.Get().GetValue()));
    }
    Clear();
    return Captures;
}

template<typename T>
INL Job<T>& Nexus<T>::Get(const std::string& Input) {
    return The.operator()(Input);
}

template<typename T>
INL Job<T>& Nexus<T>::Get(const char* Input) {
    return The.operator()(std::string(Input));
}

template<typename T>
INL Job<T>& Nexus<T>::GetWithoutProtection(const xint val) noexcept{
    return *MmIdxJob[val];
}

template<typename T>
INL Job<T>& Nexus<T>::Get(const xint Input) {
    return The.operator()(Input);
}


// ------------------------------------------------------------------------------------------

template<typename T>
INL xint Nexus<T>::Size() const
{
    return MmIdxJob.size();
}

template<typename T>
INL void Nexus<T>::WaitAll()
{
    while (MvTaskDeque.size() || MnInstTaskCount > 0)
    {
        if (MvTaskDeque.size() && MnInstTaskCount == 0)
            MoMutex.SignalOne();
        Nexus<T>::Sleep(1);
    }
}

template<typename T>
INL bool Nexus<T>::TaskCompleted() const
{
    return !MvTaskDeque.size();
}

template<typename T>
INL void Nexus<T>::Clear()
{

    if (MnInstTaskCount == 0 && MvTaskDeque.size() == 0)
    {
        MmIdxJob.clear();
        MmStrJob.clear();
        MnInstTaskCount = 0;
        MnNextTaskID = 0;
    }
}

template<typename T>
INL void Nexus<T>::Sleep(unsigned int extent) const
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    ::Sleep(extent);
#else
    ::usleep(extent);
#endif
}
