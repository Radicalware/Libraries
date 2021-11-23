#include "MutexHandler.h"

RA::MutexHandler::~MutexHandler()
{
    if(MoMutexPtr)
        delete MoMutexPtr;
}

void RA::MutexHandler::CheckMutex()
{
    if (MoMutexPtr == nullptr)
        MoMutexPtr = new RA::Mutex;
}

RA::Mutex& RA::MutexHandler::GetMutex()
{
    The.CheckMutex();
    return *MoMutexPtr;
}

RA::Mutex* RA::MutexHandler::GetMutexPtr()
{
    The.CheckMutex();
    return MoMutexPtr;
}
