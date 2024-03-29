﻿#include "Thread/MutexHandler.h"

#include <unordered_map>
#include <sstream>

void RA::MutexHandler::CreateMTX()
{
    if (MnMutexID == 0)
    {
        MnMutexID = ++SnMutexCount;
        SmMutex.insert({ MnMutexID, RA::Mutex() });
    }
}

bool RA::MutexHandler::UsingMTX() const
{
    return MnMutexID > 0;
}

RA::Mutex& RA::MutexHandler::GetMTX() const
{
    if (MnMutexID == 0)
        throw "MutexHandler::GetMutex >> Has No Mutex ID";
    return SmMutex.at(MnMutexID);
}

RA::Mutex& RA::MutexHandler::GetMTX()
{
    if (MnMutexID == 0)
        CreateMTX();

    return SmMutex[MnMutexID];
}
