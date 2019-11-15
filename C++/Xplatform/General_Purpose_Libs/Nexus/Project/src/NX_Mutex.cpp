#include "NX_Mutex.h"

size_t NX_Mutex::Mutex_Total = 0;


NX_Mutex::NX_Mutex() {
    Mutex_Total++;
    id = Mutex_Total; 
    // first mutex starts at 1 because 0 indicates no mutex
};


NX_Mutex::NX_Mutex(const NX_Mutex& other) 
{
    mutex_unlocked = other.mutex_unlocked;
    use_mutex = other.use_mutex;
    id = other.id;
}

void NX_Mutex::mutex_on(){
        use_mutex = true;
}

void NX_Mutex::mutex_off(){
        use_mutex = false;
}