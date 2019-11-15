#pragma once

#include<mutex>
#include<condition_variable>


template<typename T> class Nexus;

struct NX_Mutex // used to lock/unlock objects on a given index
{
    friend class Nexus<void>;

    size_t id = 0;
    std::mutex key; // key for a lock (unique_lock)
    std::condition_variable sig;
    bool mutex_unlocked = true;
    bool use_mutex = true;

    static size_t Mutex_Total; // = 0 

public:
    NX_Mutex();
    NX_Mutex(const NX_Mutex& other);  

    void mutex_on();
    void mutex_off();
};