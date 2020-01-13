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

#include<future>
#include<thread>
#include<mutex>
#include<condition_variable>

#include<queue>
#include<unordered_map>
#include<functional>
#include<type_traits>

#include "Nexus_T.h"
#include "NX_Threads.h"
#include "NX_Mutex.h"
#include "Task.h"
#include "Job.h"



// =========================================================================================

template<typename T> class Nexus;


template<>
class Nexus<void> : public NX_Threads // This class is for updating member values in objects
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

    static std::vector<NX_Mutex*> s_lock_lst; // for objects in threads
    static std::mutex s_mutex;  // for Nexus
    static std::condition_variable s_sig_queue; // allow the next thread to go when work queued

    static std::vector<std::thread> s_threads; // these threads start in the constructor and don't stop until Nexus is over
    static std::queue<Nexus<void>::tsk_st> s_task_queue; // This is where tasks are held before being run

    static void Task_Looper(int thread_idx);

public:
    Nexus();
    ~Nexus();
    static void Start();
    static int Stop();

    static void Set_Mutex_On(size_t nx_mutex); 
    static void Set_Mutex_Off(size_t nx_mutex);

    template <typename F, typename... A>
    static void Add_Job(F&& function, A&& ... Args);

    template <typename F, typename... A>
    static void Add_Job(NX_Mutex& nx_mutex, F&& function, A&& ... Args);

    template <typename F, typename V, typename... A>
    static void Add_Job_Val(F&& function, V& element, A&&... Args);

    template <typename K, typename V, typename F, typename... A>
    static void Add_Job_Pair(F&& function, K key, V& value, A&& ... Args);

    // Required due to Linux not Windows (returns void Job<T>)
    static Job<short int> Get_Fast(size_t dummy) noexcept;

    static size_t Size();

    static void Wait_All();
    static void Clear();

    static void Sleep(unsigned int extent);
};

// =========================================================================================

inline void Nexus<void>::Task_Looper(int thread_idx)
{
    while (true) {
        size_t mutex_idx = 0;
        Task<void> tsk;
        {
            std::unique_lock<std::mutex> lock(s_mutex);
            s_sig_queue.wait(lock, []() {
                return ((Nexus<void>::s_finish_tasks || Nexus<void>::s_task_queue.size()) && NX_Threads::Threads_Are_Available());
            });

            if (s_task_queue.empty())
                return;

            NX_Threads::s_Threads_Used++;
            tsk = std::move(s_task_queue.front().tsk);
            mutex_idx = s_task_queue.front().mutex_idx;

            NX_Threads::s_task_count++;
            s_inst_task_count++;

            s_task_queue.pop();

        }

        if (!mutex_idx) // no lock given
            tsk();
        else if (s_lock_lst.at(mutex_idx)->use_mutex) // lock was given with a mutex set to on
        {
            NX_Mutex& locker = *s_lock_lst.at(mutex_idx);
            std::unique_lock<std::mutex> UL(locker.key);
            locker.sig.wait(UL, [&locker]() {return locker.mutex_unlocked; });
            locker.mutex_unlocked = false;
            tsk();
            locker.mutex_unlocked = true;
            locker.sig.notify_one();
        }
        else // lock was given but the mutex was set to off
            tsk();
        
        NX_Threads::s_Threads_Used--; // protected as atomic
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
        s_lock_lst.clear();
        s_lock_lst.push_back(nullptr); // starts size at 1 and index at 0

        NX_Threads::s_Inst_Count++;
        s_threads.reserve(NX_Threads::s_Thread_Count);
        for (int i = 0; i < NX_Threads::s_Thread_Count; ++i)
            s_threads.emplace_back(std::bind((void(*)(int)) & Nexus<void>::Task_Looper, i)); // static member function, don't use "this"

        NX_Threads::s_Threads_Used = 0;
    }
    s_initialized = true;
}

inline int Nexus<void>::Stop()
{
    if (s_initialized)
    {
        Nexus<>::Wait_All();
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

inline void Nexus<void>::Set_Mutex_On(size_t nx_mutex)
{
    s_lock_lst.at(nx_mutex)->use_mutex = true;
}

inline void Nexus<void>::Set_Mutex_Off(size_t nx_mutex)
{
    s_lock_lst.at(nx_mutex)->use_mutex = false;
}

template <typename F, typename... A>
inline void Nexus<void>::Add_Job(F&& function, A&&... Args)
{
    auto binded_function = std::bind(function, std::ref(Args)...);
    std::lock_guard <std::mutex>lock(s_mutex);
    s_task_queue.emplace(Task<void>(std::move(binded_function)), 0);
    s_sig_queue.notify_one();
}


template<typename F, typename ...A>
inline void Nexus<void>::Add_Job(NX_Mutex& nx_mutex, F&& function, A&& ...Args)
{
    auto binded_function = std::bind(function, std::ref(Args)...);
    std::lock_guard <std::mutex>lock(s_mutex);

    if(s_lock_lst.size() <= nx_mutex.id) // id should never be 'gt' size
        s_lock_lst.push_back(&nx_mutex);

    s_task_queue.emplace(Task<void>(std::move(binded_function)), nx_mutex.id);
    // nxm.id references the location in s_lock_lst
    s_sig_queue.notify_one();
}

template <typename F, typename V, typename... A>
void Nexus<void>::Add_Job_Val(F&& function, V& element, A&&... Args)
{
    auto binded_function = std::bind(function, std::ref(element), std::ref(Args)...);
    std::lock_guard <std::mutex> lock(s_mutex);
    s_task_queue.emplace(Task<void>(std::move(binded_function)), 0);
    s_sig_queue.notify_one();
}

template<typename K, typename V, typename F, typename ...A>
inline void Nexus<void>::Add_Job_Pair(F&& function, K key, V& value, A&& ...Args)
{
    auto binded_function = std::bind(function, std::ref(key), std::ref(value), std::ref(Args)...);
    std::lock_guard <std::mutex> lock(s_mutex);
    s_task_queue.emplace(Task<void>(std::move(binded_function)), 0);
    s_sig_queue.notify_one();
}

// Class required due to Linux (not Windows)
inline Job<short int> Nexus<void>::Get_Fast(size_t dummy) noexcept { return Job<short int>(); }

inline size_t Nexus<void>::Size() {
    return s_inst_task_count;
}

inline void Nexus<void>::Wait_All()
{
    while (s_task_queue.size()) Nexus<void>::Sleep(1);
    while (NX_Threads::s_Threads_Used > 0) Nexus<void>::Sleep(1);
}

inline void Nexus<void>::Clear() {
    s_inst_task_count = 0;

    Nexus<void>::Wait_All();
    s_lock_lst.clear();
    NX_Mutex::Mutex_Total = 0;
}

inline void Nexus<void>::Sleep(unsigned int extent)
{
    #if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
        ::Sleep(extent);
    #else
        ::usleep(extent);
    #endif
}

