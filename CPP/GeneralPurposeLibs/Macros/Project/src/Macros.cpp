#include "Macros.h"



namespace RA
{
    xvector<char> Rand::GetRandomCharSequence(const xint Length, const xint Floor, const xint Ceiling)
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

    xstring Rand::GetRandomStr(const xint Length, const xint Floor, const xint Ceiling)
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
