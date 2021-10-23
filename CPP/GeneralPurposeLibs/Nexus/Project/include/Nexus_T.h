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
#include<functional>
#include<type_traits>

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

    std::vector<std::thread>           MvThreads;     // these threads start in the constructor and don't stop until Nexus is over
    std::deque<RA::SharedPtr<Task<T>>> McTaskDeque;   // This is where tasks are held before being run
                                            // deque chosen over queue because the deque has an iterator

    // These values are used for pulling outpu tdata from your jobs.
    std::unordered_map<std::string, RA::SharedPtr<Job<T>>> MmStrJob;
    std::unordered_map<size_t, RA::SharedPtr<Job<T>>>      MmIdxJob;

    void TaskLooper(int thread_idx);

    template <typename F, typename ...A>
    void Add(F& function, A&& ...Args);

    template<typename F, typename ...A>
    inline void AddKVP(const std::string& key, F& function, A&& ...Args);

public:
    typedef T value_type;

    Nexus();
    ~Nexus();

    // Job: String (for referencing unfinished job) + Function + args
    template <typename F, typename ...A>
    void AddJob(const std::string& key, F&& function, A&& ...Args);
    template <typename F, typename ...A>
    // Job: char* (for referencing unfinished job) + Function + args
    void AddJob(const char* key, F&& function, A&& ...Args);
    template <typename F, typename ...A>
    inline typename std::enable_if<!std::is_same<F, std::string>::value, void>::type AddJob(F&& function, A&& ...Args);

    // These are to be used by xvector/xmap
    template <typename F, typename ONE, typename ...A>
    void AddJobVal(F&& function, ONE& element, A&& ...Args); // commonly used by xvector

    template <typename K, typename V, typename F, typename ...A>
    void AddJobPair(F&& function, K& key, V& value, A&& ...Args); // commonly used by xmap

    // Getters can't be const due to the mutex
    Job<T>& operator()(const std::string& val);
    Job<T>& operator()(const size_t val);

    // Getters can't be const due to the mutex
    std::vector<T> GetAll();
    Job<T>& Get(const std::string& val);
    Job<T>& Get(const size_t val);
    Job<T>& Get(const char* Input);
    Job<T>& GetWithoutProtection(const size_t val) noexcept;

    size_t Size() const;
    void WaitAll();
    bool TaskCompleted() const;
    void CheckAndClear();
    void Clear();
    void Sleep(unsigned int extent) const;
};

// =========================================================================================

template<typename T>
inline void Nexus<T>::TaskLooper(int thread_idx)
{
    std::atomic<size_t> LnLastTaskIdx = 0;
    while (true) 
    {
        std::atomic<size_t>  LnTaskIdx = 0;
        {
            auto Lock = MoMutex.CreateLock([this]() {
                return ((MbFinishTasks || McTaskDeque.size()) && RA::Threads::GetAllowedThreadCount() - MnInstTaskCount > 0); });

            if (McTaskDeque.empty())
                return;

            LnTaskIdx = MmIdxJob.size();
            if (LnLastTaskIdx != 0 && LnLastTaskIdx == LnTaskIdx)
                continue;

            RA::SharedPtr<Job<T>> JobPtr = MakePtr<Job<T>>(McTaskDeque.front(), RA::Threads::TotalTasksCounted);
            McTaskDeque.pop_front();
            MmIdxJob.insert({ LnTaskIdx, JobPtr });

            if (JobPtr.Get().GetTask().HasName())
                MmStrJob.insert({ JobPtr.Get().GetTask().GetName(), JobPtr });

            MnInstTaskCount++;
        }

        MmIdxJob.at(LnTaskIdx).Get().Run();

        auto Lock = MoMutex.CreateLock();
        if (MbFinishTasks && !McTaskDeque.size())
        {
            MnInstTaskCount = 0;
            return;
        }
        LnLastTaskIdx = LnTaskIdx.load();
        MnInstTaskCount--;
    }
}

template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::Add(F& function, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::ref(Args)...);
    McTaskDeque.emplace_back(MakePtr<Task<T>>(std::move(BindedFunction)));
}

template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::AddKVP(const std::string& key, F& function, A&& ...Args)
{
    auto BindedFunction = std::bind(function, std::ref(Args)...);
    McTaskDeque.emplace_back(MakePtr<Task<T>>(std::move(BindedFunction), key));
}
// ------------------------------------------------------------------------------------------
template<typename T>
Nexus<T>::Nexus()
{
    RA::Threads::InstanceCount++;
    MvThreads.reserve(RA::Threads::Allowed);
    for (int i = 0; i < RA::Threads::Allowed; ++i)
        MvThreads.emplace_back(std::bind(&Nexus<T>::TaskLooper, std::ref(*this), i));
    RA::Threads::Used = 0;
}

template<typename T>
Nexus<T>::~Nexus()
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
inline void Nexus<T>::AddJob(const std::string& key, F&& function, A&& ...Args)
{
    auto Lock = MoMutex.CreateLock();
    CheckAndClear();
    The.AddKVP(key, function, std::ref(Args)...);
}

template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::AddJob(const char* key, F&& function, A&& ...Args)
{
    auto Lock = MoMutex.CreateLock();
    CheckAndClear();
    The.AddKVP(std::string(key), function, std::ref(Args)...);
}

template<typename T>
template <typename F, typename ...A>
inline typename std::enable_if<!std::is_same<F, std::string>::value, void>::type 
    Nexus<T>::AddJob(F&& function, A&& ...Args)
{
    auto Lock = MoMutex.CreateLock();
    CheckAndClear();
    The.Add(function, std::ref(Args)...);
}

template<typename T>
template <typename F, typename ONE, typename ...A>
inline void Nexus<T>::AddJobVal(F&& function, ONE& element, A&& ...Args)
{
    auto Lock = MoMutex.CreateLock();
    CheckAndClear();
    auto BindedFunction = std::bind(function, std::ref(element), Args...);
    McTaskDeque.emplace_back(MakePtr<Task<T>>(std::move(BindedFunction)));
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline void Nexus<T>::AddJobPair(F&& function, K& key, V& value, A&& ...Args)
{
    auto Lock = MoMutex.CreateLock();
    CheckAndClear();
    auto BindedFunction = std::bind(function, std::ref(key), std::ref(value), Args...);
    McTaskDeque.emplace_back(MakePtr<Task<T>>(std::move(BindedFunction)));
}
// ------------------------------------------------------------------------------------------

template<typename T>
inline Job<T>& Nexus<T>::operator()(const std::string& Input)
{
    auto Lock = MoMutex.CreateLock();

    // If it is not already cached, is it queued?
    if (!MmStrJob.count(Input))
        throw std::runtime_error("Nexus Key Not Found!");

    auto& Target = MmStrJob[Input].Get();
    while (!Target.IsDone()) The.Sleep(1);

    Target.TestException();
    return Target;
}


template<typename T>
inline Job<T>& Nexus<T>::operator()(const size_t Input)
{
    auto Lock = MoMutex.CreateLock();

    if (Input > MmIdxJob.size()) // Idx is baesd on Job-Hold Size
        throw std::runtime_error("Requested Job is Out of Range\n");

    while (!MmIdxJob.count(Input))          The.Sleep(1);

    auto& Target = MmIdxJob[Input].Get();
    while (!Target.IsDone()) The.Sleep(1);

    Target.TestException();
    return Target;
}

template<typename T>
inline std::vector<T> Nexus<T>::GetAll() 
{
    The.WaitAll();
    auto Lock = MoMutex.CreateLock();
    std::vector<T> Captures;
    for (std::pair<const size_t, RA::SharedPtr<Job<T>>> & Target : MmIdxJob)
    {
        Target.second.Get().TestException();
        Captures.push_back(Target.second.Get().Move());
    }
    for (std::pair<const std::string, RA::SharedPtr<Job<T>>>& Target : MmStrJob)
    {
        Target.second.Get().TestException();
        Captures.push_back(Target.second.Get().Move());
    }
    return Captures;
}

template<typename T>
inline Job<T>& Nexus<T>::Get(const std::string& Input) {
    return The.operator()(Input);
}

template<typename T>
inline Job<T>& Nexus<T>::Get(const char* Input) {
    return The.operator()(std::string(Input));
}

template<typename T>
inline Job<T>& Nexus<T>::GetWithoutProtection(const size_t val) noexcept{
    return *MmIdxJob[val];
}

template<typename T>
inline Job<T>& Nexus<T>::Get(const size_t Input) {
    return The.operator()(Input);
}


// ------------------------------------------------------------------------------------------

template<typename T>
inline size_t Nexus<T>::Size() const
{
    return MmIdxJob.size();
}

template<typename T>
inline void Nexus<T>::WaitAll()
{
    while (McTaskDeque.size() || MnInstTaskCount > 0)
    {
        The.Sleep(1);
    }
}

template<typename T>
inline bool Nexus<T>::TaskCompleted() const
{
    return !McTaskDeque.size();
}

template<typename T>
inline void Nexus<T>::CheckAndClear()
{
    if (McTaskDeque.size() || MnInstTaskCount > 0)
        The.Clear();
}

template<typename T>
inline void Nexus<T>::Clear()
{
    //MmIdxJob.clear();
    //MmStrJob.clear();
    MnInstTaskCount = 0;
}

template<typename T>
inline void Nexus<T>::Sleep(unsigned int extent) const
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    ::Sleep(extent);
#else
    ::usleep(extent);
#endif
}
