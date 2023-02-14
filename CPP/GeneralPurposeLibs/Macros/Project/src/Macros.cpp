#include "Macros.h"



namespace RA
{
    template<>
    xvector<char> GetRandomSequence<char>(const uint Length, const uint Floor, const uint Ceiling)
    {
        std::random_device os_seed;
        const uint_least32_t seed = os_seed();

        std::mt19937 generator(seed);
        std::uniform_int_distribution<unsigned int> distribute(Floor, Ceiling);

        xvector<char> Sequence;
        for (int repetition = 0; repetition < Length; ++repetition)
            Sequence << (char)distribute(generator);

        return Sequence;
    }

    xstring GetRandomStr(const uint Length, const uint Floor, const uint Ceiling)
    {
        std::random_device os_seed;
        const uint_least32_t seed = os_seed();

        std::mt19937 generator(seed);
        std::uniform_int_distribution<unsigned int> distribute(Floor, Ceiling);

        xstring Sequence;
        Sequence.reserve(Length);
        for (int repetition = 0; repetition < Length; ++repetition)
            Sequence += (char)distribute(generator);

        return Sequence;
    }
}