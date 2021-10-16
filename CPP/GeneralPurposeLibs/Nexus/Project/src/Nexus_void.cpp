#include "Nexus_void.h"


bool   Nexus<void>::s_initialized     = false;
size_t Nexus<void>::s_inst_task_count = 0;
bool   Nexus<void>::s_finish_tasks    = false;

std::unordered_map<size_t, xptr<RA::Mutex>> Nexus<void>::s_lock_lst; // for objects in threads
std::mutex Nexus<void>::s_mutex; // for Nexus
std::condition_variable Nexus<void>::s_sig_queue;

std::vector<std::thread> Nexus<void>::s_threads; // these threads start in the constructor and don't stop until Nexus is over
std::queue<Nexus<void>::tsk_st> Nexus<void>::s_task_queue; // This is where tasks are held before being run

