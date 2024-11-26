#pragma once


namespace RA
{
    enum class EStatOpt : int
    {
        NONE,
        AVG,     // Average
        RSI,     // Relative Strength Index
        STOCH,   // Stochastic
        Normals, // Normals
        MAD,     // Mean Absolute Deviation
        SD,      // Standard Deviation
        Literal  // Useful for Mapping
    };
}
