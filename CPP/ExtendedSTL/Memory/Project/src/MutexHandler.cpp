#include "MutexHandler.h"

#include <unordered_map>

RA::MutexHandler::MutexHandler()
{
    MnMutexID = SnMutexCount++;
}

bool RA::MutexHandler::UsingMutex() const
{
    return SmMutex.count(MnMutexID);
}

RA::Mutex& RA::MutexHandler::GetMutex() const
{
    if (!SmMutex.count(MnMutexID))
        SmMutex.insert({ MnMutexID, RA::Mutex() });
    return SmMutex[MnMutexID];
}
