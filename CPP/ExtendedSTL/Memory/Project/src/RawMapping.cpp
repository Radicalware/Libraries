#include "RawMapping.h"
#include <cmath>


double RA::TrimZeros(const double FnNum, const double FnAccuracy)
{
    // xint to strip zeros, then back to double type
    const double Places = 1 / FnAccuracy;
    const double Result = static_cast<double>(static_cast<unsigned long long>(FnNum * Places));
    return Result / Places;
}

float RA::TrimZeros(const float FnNum, const float FnAccuracy)
{
    const float  Places = 1 / FnAccuracy;
    const double Result = static_cast<double>(static_cast<unsigned long long>(FnNum * Places));
    return Result / Places;
}

bool RA::Appx(const double FnFirst, const double FnSecond, const double FnAcceptibleRange)
{
    if ((FnFirst + FnAcceptibleRange) <= FnSecond || (FnFirst - FnAcceptibleRange) >= FnSecond)
        return false;
    return true;
}
