#pragma once


namespace RA
{
    enum class EStatOpt : int
    {
        NONE,
        AVG,     // Average
        ZEA,     // Zero-Lag Exp. Avg
        Omaha,   // Omaha High-Low Poker
        RSI,     // Relative Strength Index
        STOCH,   // Stochastic
        Normals, // Normals
        MAD,     // Mean Absolute Deviation
        SD,      // Standard Deviation
        Literal  // Useful for Mapping
    };
}
