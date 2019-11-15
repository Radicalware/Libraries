#pragma once

#include<iostream>
#include<initializer_list>
#include<utility>

#include<future>
#include<thread>
#include<mutex>
#include<condition_variable>

#include<deque>
#include<functional>
#include<type_traits>

#include "NX_Threads.h"
#include "Task.h"
#include "Job.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#include<Windows.h>
#else 
#include<unistd.h>
#endif


#include<iostream>
#include<vector>
#include<unordered_map>
#include<initializer_list>
#include<utility>

#include<future>
#include<thread>
#include<mutex>
#include<condition_variable>

#include<deque>
#include<functional>
#include<type_traits>

#include "NX_Threads.h"
#include "Task.h"
#include "Job.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#include<Windows.h>
#else 
#include<unistd.h>
#endif


// =========================================================================================


template<typename T = void>
class Nexus : public NX_Threads
{
private:
    bool m_finish_tasks = false;
    size_t m_inst_task_count = 0;

    std::mutex m_mutex;
    std::condition_variable m_sig_deque;
    std::mutex m_get_mutex;
    std::condition_variable m_sig_get;

    std::vector<std::thread> m_threads; // these threads start in the constructor and don't stop until Nexus is over
    std::deque<Task<T>> m_task_deque; // This is where tasks are held before being run
    // deque chosen over queue because the deque has an iterator

    std::unordered_map<std::string,  const size_t> m_str_inst_mp; // KVP (std::string >> job inst)
    std::unordered_map<size_t, Job<T>>* m_inst_job_mp = nullptr;  //                    (job inst >> Job)

    void task_looper(int thread_idx);

    template <typename F, typename ...A>
    void add(F& function, A&& ...Args);

    template<typename F, typename ...A>
    inline void add_kvp(const std::string& key, F& function, A&& ...Args);

public:
    typedef T value_type;

    Nexus();
    ~Nexus();

    template <typename F, typename ...A>
    void add_job(const std::string& key, F&& function, A&& ...Args);
    template <typename F, typename ...A>
    void add_job(const char* key, F&& function, A&& ...Args);
    template <typename F, typename ...A>
    inline typename std::enable_if<!std::is_same<F, std::string>::value, void>::type add_job(F&& function, A&& ...Args);

    // These are to be used by xvector/xmap
    template <typename F, typename ONE, typename ...A>
    void add_job_val(F&& function, ONE& element, A&& ...Args); // commonly used by xvector
    template <typename K, typename V, typename F, typename ...A>
    void add_job_pair(F&& function, K& key, V& value, A&& ...Args); // commonly used by xmap

    // Getters can't be const due to the mutex
    Job<T> operator()(const std::string& val);
    Job<T> operator()(const size_t val);

    // Getters can't be const due to the mutex
    Job<T> get(const std::string& val);
    Job<T> get(const size_t val);
    Job<T> get(const char* input);
    Job<T> get_fast(const size_t val) noexcept; 

    size_t size() const;
    void wait_all() const;
    void clear();
    void sleep(unsigned int extent) const;
};

// =========================================================================================

template<typename T>
inline void Nexus<T>::task_looper(int thread_idx)
{
    while (true) {
        size_t tsk_idx;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_sig_deque.wait(lock, [this]() {
                return ((m_finish_tasks || m_task_deque.size()) && NX_Threads::Threads_Are_Available());
            });

            if (m_task_deque.empty())
                return;

            NX_Threads::s_Threads_Used++;
            tsk_idx = m_inst_task_count;
            if (m_task_deque.front().blank())
                continue;
            else
                (*m_inst_job_mp).insert({ tsk_idx, Job<T>(std::move(m_task_deque.front()), NX_Threads::s_task_count) });

            const Task<T>* latest_task = (*m_inst_job_mp)[tsk_idx].task_ptr();
            if (latest_task->has_name())
                m_str_inst_mp.insert({ latest_task->name(), tsk_idx });

            NX_Threads::s_task_count++;
            m_inst_task_count++;

            m_task_deque.pop_front();
            m_sig_get.notify_all();
        }
        (*m_inst_job_mp)[tsk_idx].init();
    }
}

template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::add(F& function, A&& ...Args)
{
    std::lock_guard<std::mutex> glock(m_get_mutex);
    auto binded_function = std::bind(function, std::ref(Args)...);
    std::lock_guard <std::mutex> lock(m_mutex);
    m_task_deque.emplace_back(std::move(binded_function));
    m_sig_deque.notify_one();
    m_sig_get.notify_all();
}

template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::add_kvp(const std::string& key, F& function, A&& ...Args)
{
    std::lock_guard<std::mutex> glock(m_get_mutex);
    auto binded_function = std::bind(function, std::ref(Args)...);
    std::lock_guard <std::mutex> lock(m_mutex);
    m_task_deque.emplace_back(std::move(binded_function), key);
    m_sig_deque.notify_one();
    m_sig_get.notify_all();
}
// ------------------------------------------------------------------------------------------
template<typename T>
Nexus<T>::Nexus()
{
    m_inst_job_mp = new std::unordered_map<size_t, Job<T>>; 
    NX_Threads::s_Inst_Count++;
    m_threads.reserve(NX_Threads::s_Thread_Count);
    for (int i = 0; i < NX_Threads::s_Thread_Count; ++i)
        m_threads.emplace_back(std::bind(&Nexus<T>::task_looper, std::ref(*this), i));

    NX_Threads::s_Threads_Used = 0;
}

template<typename T>
Nexus<T>::~Nexus()
{
    {
        std::unique_lock <std::mutex> lock(m_mutex);
        m_finish_tasks = true;
        m_sig_deque.notify_all();
    }
    for (auto& t : m_threads) t.join();
    if(m_inst_job_mp != nullptr)
        delete m_inst_job_mp;
}


// ------------------------------------------------------------------------------------------
template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::add_job(const std::string& key, F&& function, A&& ...Args)
{
    this->add_kvp(key, function, std::ref(Args)...);
}

template<typename T>
template<typename F, typename ...A>
inline void Nexus<T>::add_job(const char* key, F&& function, A&& ...Args)
{
    this->add_kvp(std::string(key), function, std::ref(Args)...);
}

template<typename T>
template <typename F, typename ...A>
inline typename std::enable_if<!std::is_same<F, std::string>::value, void>::type Nexus<T>::add_job(F&& function, A&& ...Args)
{
    this->add(function, std::ref(Args)...);
}

template<typename T>
template <typename F, typename ONE, typename ...A>
inline void Nexus<T>::add_job_val(F&& function, ONE& element, A&& ...Args)
{
    std::lock_guard<std::mutex> glock(m_get_mutex);
    auto binded_function = std::bind(function, std::ref(element), Args...);
    std::lock_guard <std::mutex> lock(m_mutex);

    m_task_deque.emplace_back(std::move(binded_function));

    m_sig_deque.notify_one();
    m_sig_get.notify_all();
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline void Nexus<T>::add_job_pair(F&& function, K& key, V& value, A&& ...Args)
{
    std::lock_guard<std::mutex> glock(m_get_mutex);
    auto binded_function = std::bind(function, std::ref(key), std::ref(value), Args...);
    std::lock_guard <std::mutex> lock(m_mutex);

    m_task_deque.emplace_back(std::move(binded_function));

    m_sig_deque.notify_one();
    m_sig_get.notify_all();
}
// ------------------------------------------------------------------------------------------

template<typename T>
inline Job<T> Nexus<T>::operator()(const std::string& input)
{
    std::unique_lock<std::mutex> glock(m_get_mutex);
    if (m_task_deque.size())
        m_sig_get.wait(glock);

    // If it is not already cached, is it queued?
    if (!m_str_inst_mp.count(input)) {
        for (const Task<T>& t : m_task_deque) {
            if (*t.name_ptr() == input)
                break;
        }
        throw std::runtime_error("Nexus Key Not Found!");
    }
    // wait until it is cached
    while (!m_str_inst_mp.count(input))
        this->sleep(1);

    size_t loc = m_str_inst_mp.at(input);
    while (!(*m_inst_job_mp)[loc].done()) 
        this->sleep(1);

    (*m_inst_job_mp)[loc].rethrow_exception();
    return (*m_inst_job_mp)[loc];
}


template<typename T>
inline Job<T> Nexus<T>::operator()(const size_t input)
{
    std::unique_lock<std::mutex> glock(m_get_mutex);
    if (m_task_deque.size())
        m_sig_get.wait(glock);

    if (m_inst_task_count + m_task_deque.size() < input + 1)
        throw std::runtime_error("Requested Job is Out of Range\n");

    while (!(*m_inst_job_mp).count(input)) this->sleep(1);
    while (!(*m_inst_job_mp)[input].done()) this->sleep(1);

    (*m_inst_job_mp)[input].rethrow_exception();
    return (*m_inst_job_mp)[input];
}

template<typename T>
inline Job<T> Nexus<T>::get(const std::string& input) {
    return this->operator()(input);
}

template<typename T>
inline Job<T> Nexus<T>::get(const char* input) {
    return this->operator()(std::string(input));
}

template<typename T>
inline Job<T> Nexus<T>::get_fast(const size_t val) noexcept
{
    return (*m_inst_job_mp)[val];
}

template<typename T>
inline Job<T> Nexus<T>::get(const size_t input) {
    return this->operator()(input);
}


// ------------------------------------------------------------------------------------------

template<typename T>
inline size_t Nexus<T>::size() const
{
    return m_inst_task_count;
}

template<typename T>
inline void Nexus<T>::wait_all() const
{
    while (m_task_deque.size()) this->sleep(1);
    while (NX_Threads::s_Threads_Used > 0) this->sleep(1);
}

template<typename T>
inline void Nexus<T>::clear()
{
    m_inst_job_mp->clear();
    m_inst_task_count = 0;
}

template<typename T>
inline void Nexus<T>::sleep(unsigned int extent) const
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    ::Sleep(extent);
#else
    ::usleep(extent);
#endif
}
