#include "Thread/MutexHandler.h"

#include <unordered_map>
#include <sstream>

RA::Mutex& RA::MutexHandler::ReadyMutex()
{
    if (MnMutexID == 0)
    {
        MnMutexID = ++SnMutexCount;
        SmMutex.insert({ MnMutexID, RA::Mutex() });
    }
    return SmMutex[MnMutexID];
}

bool RA::MutexHandler::UsingMutex() const
{
    return MnMutexID > 0;
}

RA::Mutex& RA::MutexHandler::GetMutex() const
{
    if (MnMutexID == 0)
        throw "MutexHandler::GetMutex >> Has No Mutex ID";
    return SmMutex.at(MnMutexID);
}