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

#include "Nexus_T.h"
#include "RA_Threads.h"
#include "RA_Mutex.h"
#include "Task.h"
#include "Job.h"


#define Sync() Nexus<>::WaitAll()
#define EXIT() exit(Nexus<>::Stop())

// =========================================================================================


//template<typename T> class Nexus;

// error C2292: 'Nexus<void>': best case inheritance representation: 'virtual_inheritance' declared but 'single_inheritance' required
template<>
class __single_inheritance Nexus<void> : public RA::Threads
{
public:
    struct tsk_st // used when poped from the queue to create a task
    {
        Task<void> tsk;
        size_t mutex_idx = 0; // 0 means never a mutex lock
        tsk_st() {}
        tsk_st(const Task<void>& itsk, size_t imutex) :
            tsk(itsk), mutex_idx(imutex) {}
    };

private:
    static bool s_initialized;  // false;
    static bool s_finish_tasks; // = false
    static size_t s_inst_task_count; // = 0

    static std::unordered_map<size_t, xptr<RA::Mutex>> s_lock_lst; // for objects in threads
    static std::mutex s_mutex;  // for Nexus
    static std::condition_variable s_sig_queue; // allow the next thread to go when work queued

    static std::vector<std::thread> s_threads; // these threads start in the constructor and don't stop until Nexus is over
    static std::queue<Nexus<void>::tsk_st> s_task_queue; // This is where tasks are held before being run

    inline static xptr<RA::Mutex> SoBlankMutexPtr;

    static void TaskLooper(int thread_idx);

public:
    Nexus();
    virtual ~Nexus();
    static void Start();
    static int Stop();

    static void SetMutexOn(size_t nx_mutex); 
    static void SetMutexOff(size_t nx_mutex);

    // Job: Function + Args
    template <typename F, typename... A>
    static inline void AddJob(F&& function, A&& ... Args);

    // Job: Object.Function + Args
    template <typename O, typename F, typename... A>
    static inline typename std::enable_if<!std::is_fundamental<O>::value && !std::is_same<O, RA::Mutex>::value, void>::type
        AddJob(O&& object, F&& function, A&& ... Args);

    // Job: Mutex + Object.Function + Args
    template <typename O, typename F, typename... A>
    static void AddJob(xptr<RA::Mutex>& nx_mutex, O&& object, F&& function, A&& ... Args);

    // Job: Function + Ref-Arg + Args
    template <typename F, typename V, typename... A>
    static inline typename std::enable_if<std::is_function<F>::value, void>::type
        AddJobVal(F&& function, V& element, A&&... Args);

    // Job: Function + Ref-(Key/Value) + Args
    template <typename K, typename V, typename F, typename... A>
    static void AddJobPair(F&& function, K key, V& value, A&& ... Args);

    // Required due to Linux not Windows (returns void Job<T>)
    static Job<short int> GetWithoutProtection(size_t dummy) noexcept;

    static size_t Size();

    static void WaitAll();
    static bool TaskCompleted();
    static void Clear();

    static void Sleep(unsigned int FnMilliseconds);
};

// =========================================================================================

inline void Nexus<void>::TaskLooper(int thread_idx)
{
    while (true) {
        size_t mutex_idx = 0;
        Task<void> tsk;
        {
            std::unique_lock<std::mutex> lock(s_mutex);
            s_sig_queue.wait(lock, []() {
                return ((Nexus<void>::s_finish_tasks || Nexus<void>::s_task_queue.size()) && RA::Threads::ThreadsAreAvailable());
            });

            if (s_task_queue.empty())
                return;

            RA::Threads::Used++;
            tsk = std::move(s_task_queue.front().tsk);
            mutex_idx = s_task_queue.front().mutex_idx;

            RA::Threads::TaskCount++;
            s_inst_task_count++;

            s_task_queue.pop();
        }

        if (!mutex_idx) // no lock given
            tsk();
        else if (s_lock_lst.at(mutex_idx)->MbUseMutex) // lock was given with a mutex set to on
        {
            RA::Mutex& locker = *s_lock_lst.at(mutex_idx);
            //std::unique_lock<std::mutex> UL(locker.key);
            //locker.sig.wait(UL, [&locker]() {return !locker.mutex_locked; });
            //locker.mutex_locked = true;
            locker.WaitAndLock();
            tsk();
            locker.SetLockOff();
            //locker.mutex_locked = false;
            //locker.sig.notify_one();

            if(RA::Threads::Used == 0 && s_task_queue.size() == 0)
                s_lock_lst.clear();
        }
        else // lock was given but the mutex was set to off
            tsk();
        
        RA::Threads::Used--; // protected as atomic
    }
}

inline Nexus<void>::Nexus(){
    Nexus<void>::Start();
}

inline Nexus<void>::~Nexus(){
}

inline void Nexus<void>::Start()
{
    if (!s_initialized) 
    {
        if (!SoBlankMutexPtr)
        {
            SoBlankMutexPtr = MakePtr<RA::Mutex>();
            SoBlankMutexPtr->SetMutexOff();
        }

        s_lock_lst.clear();
        s_lock_lst.insert({ 0, nullptr }); // starts size at 1 and index at 0
        s_lock_lst.insert({ 1, SoBlankMutexPtr }); // starts size at 1 and index at 0

        RA::Threads::InstanceCount++;
        s_threads.reserve(RA::Threads::Remaining);
        for (int i = 0; i < RA::Threads::Remaining; ++i)
            s_threads.emplace_back(std::bind((void(*)(int)) & Nexus<void>::TaskLooper, i)); // static member function, don't use "this"

        RA::Threads::Used = 0;
    }
    s_initialized = true;
}

inline int Nexus<void>::Stop()
{
    if (s_initialized)
    {
        Nexus<>::WaitAll();
        {
            std::unique_lock <std::mutex> lock(s_mutex);
            s_finish_tasks = true;
            s_sig_queue.notify_all();
        }
        for (auto& thrd : s_threads) thrd.join();
        s_lock_lst.clear();
    }
    s_initialized = false;
    return 0;
}

inline void Nexus<void>::SetMutexOn(size_t nx_mutex)
{
    s_lock_lst.at(nx_mutex)->MbUseMutex = true;
}

inline void Nexus<void>::SetMutexOff(size_t nx_mutex)
{
    s_lock_lst.at(nx_mutex)->MbUseMutex = false;
}

template <typename F, typename... A>
static inline void Nexus<void>::AddJob(F&& function, A&&... Args)
{
    auto binded_function = std::bind(function, std::ref(Args)...);
    std::lock_guard <std::mutex>lock(s_mutex);
    s_task_queue.emplace(Task<void>(std::move(binded_function)), 0);
    s_sig_queue.notify_one();
}

template<typename O, typename F, typename ...A>
inline typename std::enable_if<!std::is_fundamental<O>::value && !std::is_same<O, RA::Mutex>::value, void>::type
    Nexus<void>::AddJob(O&& object, F&& function, A&& ...Args)
{
    auto binded_function = std::bind(function, std::ref(object), std::ref(Args)...);
    std::lock_guard <std::mutex>lock(s_mutex);

    s_task_queue.emplace(Task<void>(std::move(binded_function)), SoBlankMutexPtr->id);
    s_sig_queue.notify_one();
}

template<typename O, typename F, typename ...A>
inline void Nexus<void>::AddJob(xptr<RA::Mutex>& nx_mutex, O&& object, F&& function, A&& ...Args)
{
    auto binded_function = std::bind(function, std::ref(object), std::ref(Args)...);
    std::lock_guard <std::mutex>lock(s_mutex);

    if (!nx_mutex)
        nx_mutex = MakePtr<RA::Mutex>();

    if (s_lock_lst.size() <= nx_mutex->id) // id should never be 'gt' size
        s_lock_lst.insert({ nx_mutex->id, nx_mutex });

    s_task_queue.emplace(Task<void>(std::move(binded_function)), nx_mutex->id);
    // nxm.id references the location in s_lock_lst
    s_sig_queue.notify_one();
}

template <typename F, typename V, typename... A>
static inline typename std::enable_if<std::is_function<F>::value, void>::type
    Nexus<void>::AddJobVal(F&& function, V& element, A&&... Args)
{
    auto binded_function = std::bind(function, std::ref(element), std::ref(Args)...);
    std::lock_guard <std::mutex> lock(s_mutex);
    s_task_queue.emplace(Task<void>(std::move(binded_function)), 0);
    s_sig_queue.notify_one();
}

template<typename K, typename V, typename F, typename ...A>
inline void Nexus<void>::AddJobPair(F&& function, K key, V& value, A&& ...Args)
{
    auto binded_function = std::bind(function, std::ref(key), std::ref(value), std::ref(Args)...);
    std::lock_guard <std::mutex> lock(s_mutex);
    s_task_queue.emplace(Task<void>(std::move(binded_function)), 0);
    s_sig_queue.notify_one();
}

// Class required due to Linux (not Windows)
inline Job<short int> Nexus<void>::GetWithoutProtection(size_t dummy) noexcept { return Job<short int>(); }

inline size_t Nexus<void>::Size(){
    return s_inst_task_count;
}

inline void Nexus<void>::WaitAll()
{
    while (RA::Threads::Used > 0) Nexus<void>::Sleep(1);
    while (s_task_queue.size())   Nexus<void>::Sleep(1);
}

inline bool Nexus<void>::TaskCompleted()
{
    return !s_task_queue.size();
}

inline void Nexus<void>::Clear() 
{
    s_inst_task_count = 0;

    Nexus<void>::WaitAll();
    s_lock_lst.clear();
    RA::Mutex::Mutex_Total = 0;
}

inline void Nexus<void>::Sleep(unsigned int FnMilliseconds)
{
    #if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
        ::Sleep(FnMilliseconds);
    #else
        ::usleep(FnMilliseconds);
    #endif
}

