#include "RawMapping.h"


double RA::TrimZeros(const double FnNum, const double FnAccuracy)
{
    // pint to strip zeros, then back to double type
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