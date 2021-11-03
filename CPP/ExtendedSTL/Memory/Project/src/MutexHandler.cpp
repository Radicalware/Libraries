#include "MutexHandler.h"

void RA::MutexHandler::InitMutex()
{
    if (!MoMutexPtr)
        MoMutexPtr = RA::MakeShared<RA::Mutex>();
}

RA::Mutex& RA::MutexHandler::GetMutex()
{
    if (!MoMutexPtr)
        InitMutex();
    return MoMutexPtr.Get();
}

const RA::Mutex& RA::MutexHandler::GetMutex() const
{
    return MoMutexPtr.Get();
}

RA::SharedPtr<RA::Mutex>& RA::MutexHandler::GetMutexPtr()
{
    if (!MoMutexPtr)
        InitMutex();
    return MoMutexPtr;
}

const RA::SharedPtr<RA::Mutex>& RA::MutexHandler::GetMutexPtr() const
{
    return MoMutexPtr;
}
