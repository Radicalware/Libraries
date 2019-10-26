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

#include "CPU_Threads.h"
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

#include "CPU_Threads.h"
#include "Task.h"
#include "Job.h"

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
#include<Windows.h>
#else 
#include<unistd.h>
#endif


// =========================================================================================

namespace util {
    template <typename T>
    using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;
};

template<typename T>
class Threader : private CPU_Threads
{
private:
    bool m_finish_tasks = false;
    size_t m_inst_task_count = 0;
    std::mutex m_mutex;
    std::condition_variable m_sig_deque;
    std::mutex m_get_mutex;
    std::condition_variable m_sig_get;
    std::vector<std::thread> m_threads;   // We add new tasks by succession from the m_task_deque here
    std::deque<Task<T>> m_task_deque; // This is where tasks are held before being run
    // deque chosen over queue because the deque has an iterator
    std::unordered_map<const std::string*, size_t> m_str_inst_xm; // KVP (std::string >> job inst)
    std::unordered_map<size_t, Job<T>>* m_inst_job_xm;  //                              (job inst >> Job)

    bool has_key(const std::string& key) const;
    void TaskLooper(int thread_idx);

    template <typename F, typename... A>
    void add(const std::string& key, F&& function, A&& ... Args);
public:

    Threader();
    ~Threader();

    template <typename F, typename... A>
    void add_job(const std::string& key, F&& function, A&& ... Args);
    template <typename F, typename... A>
    void add_job(const char* key, F&& function, A&& ... Args);
    template <typename F, typename... A> // the "enable_if_t" restraint was designed by "@Ben" from "Cpplang" on Slack!
    auto add_job(F&& function, A&& ... Args)->std::enable_if_t<!std::is_same_v<util::remove_cvref_t<F>, std::string>, void>;

    template <typename F, typename... A>
    void add_job_val(F&& function, T& element, A&& ... Args);

    template <typename K, typename V, typename F, typename... A>
    void add_job_pair(F&& function, K key, V& value, A&& ... Args);

    // Getters can't be const due to the mutex
    Job<T> operator()(const std::string& val);
    Job<T> operator()(const size_t val);
    // Getters can't be const due to the mutex
    Job<T> get(const std::string& val);
    Job<T> get(const size_t val);
    Job<T> get(const char* input);

    size_t size() const;
    void wait_all() const;
    void clear();
    void sleep(unsigned int extent) const;
};

// =========================================================================================

template<typename T>
inline bool Threader<T>::has_key(const std::string& key) const
{
    for (typename std::unordered_map<const std::string*, size_t>::const_iterator iter = m_str_inst_xm.begin(); iter != m_str_inst_xm.end(); ++iter) {
        if (*iter->first == key)
            return true;
    }
    return false;
}

template<typename T>
inline void Threader<T>::TaskLooper(int thread_idx)
{
    while (true) {
        size_t tsk_idx;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_sig_deque.wait(lock, [this]() {
                return ((m_finish_tasks || m_task_deque.size()) && CPU_Threads::threads_are_available());
                });

            if (m_task_deque.empty())
                return;

            CPU_Threads::Threads_Used++;
            tsk_idx = m_inst_task_count;
            if (m_task_deque.front().blank())
                continue;
            else
                (*m_inst_job_xm).insert({ tsk_idx, Job<T>(std::move(m_task_deque.front()), CPU_Threads::Task_Count) });

            const Task<T>* latest_task = (*m_inst_job_xm)[tsk_idx].task_ptr();
            if (latest_task->has_name())
                m_str_inst_xm.insert({ latest_task->name_ptr(), tsk_idx });

            CPU_Threads::Task_Count++;
            m_inst_task_count++;

            m_task_deque.pop_front();
            m_sig_get.notify_all();
        }
        (*m_inst_job_xm)[tsk_idx].init();
    }
}

template<typename T>
template<typename F, typename ...A>
inline void Threader<T>::add(const std::string& key, F&& function, A&& ...Args)
{
    std::lock_guard<std::mutex> glock(m_get_mutex);
    auto binded_function = std::bind(std::forward<F>(function), std::forward<A>(Args)...);
    std::lock_guard <std::mutex> lock(m_mutex);
    if (key.size())
        m_task_deque.emplace_back(std::move(binded_function), key);
    else
        m_task_deque.emplace_back(std::move(binded_function));
    m_sig_deque.notify_one();
    m_sig_get.notify_all();
}
// ------------------------------------------------------------------------------------------
template<typename T>
Threader<T>::Threader()
{
    int nested_variable = 5; // white 1; local

    m_inst_job_xm = new std::unordered_map<size_t, Job<T>>; // white 2; failed
    CPU_Threads::Inst_Count++;
    nested_variable; // white 1; local
    m_threads.reserve(CPU_Threads::Thread_Count); // white 1; member
    for (int i = 0; i < CPU_Threads::Thread_Count; ++i)
        m_threads.emplace_back(std::bind(&Threader<T>::TaskLooper, this, i));

    CPU_Threads::Threads_Used = 0;
}

template<typename T>
Threader<T>::~Threader()
{
    {
        std::unique_lock <std::mutex> lock(m_mutex);
        m_finish_tasks = true;
        m_sig_deque.notify_all();
    }
    for (auto& t : m_threads) t.join();
    delete m_inst_job_xm;
}
// ------------------------------------------------------------------------------------------
template<typename T>
template<typename F, typename ...A>
inline void Threader<T>::add_job(const std::string& key, F&& function, A&& ... Args)
{
    this->add(key, std::forward<F>(function), std::forward<A>(Args)...);
}

template<typename T>
template<typename F, typename ...A>
inline void Threader<T>::add_job(const char* key, F&& function, A&& ...Args)
{
    this->add(std::string(key), std::forward<F>(function), std::forward<A>(Args)...);
}

template<typename T>
template <typename F, typename... A>
inline auto Threader<T>::add_job(F&& function, A&& ... Args)->
    std::enable_if_t<!std::is_same_v<util::remove_cvref_t<F>, std::string>, void>
{
    this->add(std::string(), std::forward<F>(function), std::forward<A>(Args)...);
}

template<typename T>
template <typename F, typename... A>
inline void Threader<T>::add_job_val(F&& function, T& element, A&& ... Args)
{
    std::lock_guard<std::mutex> glock(m_get_mutex);
    auto binded_function = std::bind(std::forward<F>(function), element, std::forward<A>(Args)...);
    std::lock_guard <std::mutex> lock(m_mutex);

    m_task_deque.emplace_back(std::move(binded_function));

    m_sig_deque.notify_one();
    m_sig_get.notify_all();
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline void Threader<T>::add_job_pair(F&& function, K key, V& value, A&& ...Args)
{
    std::lock_guard<std::mutex> glock(m_get_mutex);
    auto binded_function = std::bind(std::forward<F>(function), key, value, std::forward<A>(Args)...);
    std::lock_guard <std::mutex> lock(m_mutex);

    m_task_deque.emplace_back(std::move(binded_function));

    m_sig_deque.notify_one();
    m_sig_get.notify_all();
}
// ------------------------------------------------------------------------------------------

template<typename T>
inline Job<T> Threader<T>::operator()(const std::string& input)
{
    std::unique_lock<std::mutex> glock(m_get_mutex);
    if (m_task_deque.size())
        m_sig_get.wait(glock);

    if (!this->has_key(input)) {
        for (const Task<T>& t : m_task_deque) {
            if (*t.name_ptr() == input)
                break;
        }
        throw std::runtime_error("Threader Key Not Found!");
    }

    while (!this->has_key(input))
        this->sleep(1);

    size_t loc;
    for (typename std::unordered_map<const std::string*, size_t>::iterator it = m_str_inst_xm.begin(); it != m_str_inst_xm.end(); it++) {
        if (*it->first == input)
            loc = it->second;
    }
    while (!(*m_inst_job_xm)[loc].done()) 
        this->sleep(1);

    (*m_inst_job_xm)[loc].rethrow_exception();
    return (*m_inst_job_xm)[loc];
}


template<typename T>
inline Job<T> Threader<T>::operator()(const size_t input)
{
    std::unique_lock<std::mutex> glock(m_get_mutex);
    if (m_task_deque.size())
        m_sig_get.wait(glock);

    if (m_inst_task_count + m_task_deque.size() < input + 1)
        throw std::runtime_error("Requested Job is Out of Range\n");

    while (!(*m_inst_job_xm).count(input)) this->sleep(1);
    while (!(*m_inst_job_xm)[input].done()) this->sleep(1);

    (*m_inst_job_xm)[input].rethrow_exception();
    return (*m_inst_job_xm)[input];
}

template<typename T>
inline Job<T> Threader<T>::get(const std::string& input) {
    return this->operator()(input);
}

template<typename T>
inline Job<T> Threader<T>::get(const char* input) {
    return this->operator()(std::string(input));
}

template<typename T>
inline Job<T> Threader<T>::get(const size_t input) {
    return this->operator()(input);
}


// ------------------------------------------------------------------------------------------

template<typename T>
inline size_t Threader<T>::size() const
{
    return m_inst_task_count;
}

template<typename T>
inline void Threader<T>::wait_all() const
{
    while (m_task_deque.size()) this->sleep(1);
    while (CPU_Threads::Threads_Used > 0) this->sleep(1);
}

template<typename T>
inline void Threader<T>::clear()
{
    m_inst_job_xm->clear();
    m_inst_task_count = 0;
}

template<typename T>
inline void Threader<T>::sleep(unsigned int extent) const
{
#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    ::Sleep(extent);
#else
    ::usleep(extent);
#endif
}
