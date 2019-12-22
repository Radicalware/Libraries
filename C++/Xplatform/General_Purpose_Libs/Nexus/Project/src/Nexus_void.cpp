#include "Nexus_void.h"


bool   Nexus<void>::m_initialized     = false;
size_t Nexus<void>::m_inst_task_count = 0;
bool   Nexus<void>::m_finish_tasks    = false;

std::vector<NX_Mutex*> Nexus<void>::m_lock_lst; // for objects in threads
std::mutex Nexus<void>::m_mutex; // for Nexus
std::condition_variable Nexus<void>::m_sig_queue;

std::vector<std::thread> Nexus<void>::m_threads; // these threads start in the constructor and don't stop until Nexus is over
std::queue<Nexus<void>::tsk_st> Nexus<void>::m_task_queue; // This is where tasks are held before being run

