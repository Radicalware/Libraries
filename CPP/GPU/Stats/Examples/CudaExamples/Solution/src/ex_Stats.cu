// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Macros.h"
#include "xmap.h"

#include "StatsGPU.cuh"
#include "CudaBridge.cuh"

// #include "vld.h"

using std::cout;
using std::endl;

template<typename ...R>
void CudaShot(R... Args)
{
    RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), std::forward<R>(Args)...);
    RA::CudaBridge<>::SyncAll();
}


namespace Test
{
    namespace Kernel
    {
        namespace Add
        {
            static __global__ void Value(RA::StatsGPU* StatsPtr, const double FnValue)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                *StatsPtr << FnValue;
            }
            static __global__ void ValuesToRange(RA::StatsGPU* StatsPtr, const xint FnStart, const xint FnEnd)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                for (int i = FnStart; i <= FnEnd; i++)
                    Stats << i;
            }
            static __global__ void StaticValueToRange(RA::StatsGPU* StatsPtr, const double FnValue, const xint FnStart, const xint FnEnd)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                for (int i = FnStart; i <= FnEnd; i++)
                    Stats << FnValue;
            }


            static __global__ void AddIncreasingRSI(RA::StatsGPU* StatsPtr, double FnValue, const xint FnStart, const xint FnEnd)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;

                for (int i = FnStart; i <= FnEnd; i++)
                {
                    if (i % 2)
                        FnValue += 3;
                    else
                        FnValue -= 1;

                    if (FnValue < 0)
                        printf("Price below zero in " __CLASS__);

                    Stats << FnValue;
                }
            }

            static __global__ void AddDecreasingRSI(RA::StatsGPU* StatsPtr, double FnValue, const xint FnStart, const xint FnEnd)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;

                for (int i = FnStart; i <= FnEnd; i++)
                {
                    if (i % 2)
                        FnValue += 1;
                    else
                        FnValue -= 3;

                    if (FnValue < 0)
                        printf("Price below zero in " __CLASS__);

                    Stats << FnValue;
                }
            }
        }

        namespace Check
        {
            static __global__ void AverageEquals(RA::StatsGPU* StatsPtr, const double FnAvg)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                constexpr auto LnAllowedVariance = 0.0003;
                if (FnAvg > Stats.AVG().GetAVG() + LnAllowedVariance
                    || FnAvg < Stats.AVG().GetAVG() - LnAllowedVariance)
                    printf(RED "%fll != %fll\n" WHITE, Stats.AVG().GetAVG(), FnAvg);
            }

            static __global__ void StochEquals(RA::StatsGPU* StatsPtr, const double FnAvg)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                constexpr auto LnAllowedVariance = 0.0003;
                if (FnAvg > Stats.STOCH().GetSTOCH() + LnAllowedVariance
                    || FnAvg < Stats.STOCH().GetSTOCH() - LnAllowedVariance)
                    printf(RED "%fll != %fll\n" WHITE, Stats.STOCH().GetSTOCH(), FnAvg);
            }

            static __global__ void RSIEquals(RA::StatsGPU* StatsPtr, const double FnAvg)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                constexpr auto LnAllowedVariance = 0.0003;
                if (FnAvg > Stats.RSI().GetRSI() + LnAllowedVariance
                    || FnAvg < Stats.RSI().GetRSI() - LnAllowedVariance)
                    printf(RED "%fll != %fll\n" WHITE, Stats.RSI().GetRSI(), FnAvg);
            }

            static __global__ void MeanAbsoluteDeviationEquals(RA::StatsGPU* StatsPtr, const double FnAvg)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                constexpr auto LnAllowedVariance = 0.0003;
                if (FnAvg > Stats.MAD().GetDeviation() + LnAllowedVariance
                    || FnAvg < Stats.MAD().GetDeviation() - LnAllowedVariance)
                    printf(RED "%fll != %fll\n" WHITE, Stats.MAD().GetDeviation(), FnAvg);
            }
            static __global__ void StandardDeviationEquals(RA::StatsGPU* StatsPtr, const double FnAvg)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                constexpr auto LnAllowedVariance = 0.0003;
                if (FnAvg > Stats.SD().GetDeviation() + LnAllowedVariance
                    || FnAvg < Stats.SD().GetDeviation() - LnAllowedVariance)
                    printf(RED "%fll != %fll\n" WHITE, Stats.SD().GetDeviation(), FnAvg);
            }

            static __global__ void MeanAbsoluteDeviationOffsetEquals(RA::StatsGPU* StatsPtr, const double FnAvg)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                constexpr auto LnAllowedVariance = 0.0003;
                if (FnAvg > Stats.MAD().GetOffset() + LnAllowedVariance
                    || FnAvg < Stats.MAD().GetOffset() - LnAllowedVariance)
                    printf(RED "%fll != %fll\n" WHITE, Stats.MAD().GetOffset(), FnAvg);
            }
            static __global__ void StandardDeviationOffsetEquals(RA::StatsGPU* StatsPtr, const double FnAvg)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                constexpr auto LnAllowedVariance = 0.0003;
                if (FnAvg > Stats.SD().GetOffset() + LnAllowedVariance
                    || FnAvg < Stats.SD().GetOffset() - LnAllowedVariance)
                    printf(RED "%fll != %fll\n" WHITE, Stats.SD().GetOffset(), FnAvg);
            }
        }

        namespace Print
        {
            static __global__ void SumAndAVG(RA::StatsGPU* StatsPtr)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                printf("AVG: %lf\n", Stats.AVG().GetAVG());
                printf("SUM: %lf\n", Stats.AVG().GetSum());
                printf("\n");
            }
            static __global__ void MaxMinStoch(RA::StatsGPU* StatsPtr)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                printf("MAX: %lf\n", Stats.STOCH().GetMax());
                printf("MIN: %lf\n", Stats.STOCH().GetMin());
                printf("STO: %lf\n", Stats.STOCH().GetSTOCH());
                printf("\n");
            }
            static __global__ void RSI(RA::StatsGPU* StatsPtr)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                printf("RSI: %lf\n", Stats.GetRSI());
            }

            static __global__ void StandardDeviation(RA::StatsGPU* StatsPtr)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                printf("SD: %lf\n", Stats.SD().GetDeviation());
            }
            static __global__ void MeanAbsoluteDeviation(RA::StatsGPU* StatsPtr)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                printf("MAD: %lf\n", Stats.MAD().GetDeviation());
            }

            static __global__ void StandardDeviationOffset(RA::StatsGPU* StatsPtr)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                printf("SD Offset: %lf\n", Stats.SD().GetOffset());
            }
            static __global__ void MeanAbsoluteDeviationOffset(RA::StatsGPU* StatsPtr)
            {
                if (!StatsPtr)
                    printf(RED "Null Arg in: " __CLASS__ "\n" WHITE);
                auto& Stats = *StatsPtr;
                printf("MAD Offset: %lf\n", Stats.MAD().GetOffset());
            }

        }
    }

    namespace Storage
    {
        namespace GPU
        {
            void AVG()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 20;
                const auto LnLogicalSize = 10;

                auto Stats1 = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::EStatOpt, xint>({
                        {RA::EStatOpt::AVG, LnLogicalSize} // half storage size
                        })
                );

                const auto LnLogicalLoopSize = 20;

                RA::Host::ConfigureStatsSync(Stats1);
                CudaShot(&Kernel::Add::ValuesToRange, Stats1, 1, LnLogicalLoopSize);
                CudaShot(&Kernel::Print::SumAndAVG, Stats1);
                CudaShot(&Kernel::Check::AverageEquals, Stats1, 15.5);

                auto Stats2 = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::EStatOpt, xint>({
                        {RA::EStatOpt::AVG, LnStorageSize} // full storage size
                        })
                );
                RA::Host::ConfigureStatsSync(Stats2);
                CudaShot(&Kernel::Add::ValuesToRange, Stats2, 1, LnLogicalLoopSize * 2);
                CudaShot(&Kernel::Print::SumAndAVG, Stats2);
                CudaShot(&Kernel::Check::AverageEquals, Stats2, 30.5);
                Rescue();
            }

            void STOCH()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 20;
                const auto LnLogicalSize = 10;

                {
                    auto Stats = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                        xmap<RA::EStatOpt, xint>({
                            {RA::EStatOpt::AVG, LnLogicalSize},
                            {RA::EStatOpt::STOCH, LnLogicalSize},
                            })
                    );
                    RA::Host::ConfigureStatsSync(Stats);

                    double LnPrice = 50;
                    CudaShot(&Kernel::Add::StaticValueToRange, Stats, LnPrice, 0, LnStorageSize);
                    CudaShot(&Kernel::Add::Value, Stats, 25);
                    CudaShot(&Kernel::Add::Value, Stats, 35);

                    CudaShot(&Kernel::Print::MaxMinStoch, Stats);
                    CudaShot(&Kernel::Check::StochEquals, Stats, 40);
                }
                {
                    auto Stats = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                        xmap<RA::EStatOpt, xint>({
                            {RA::EStatOpt::AVG, LnLogicalSize},
                            {RA::EStatOpt::STOCH, LnLogicalSize}
                            })
                    );
                    RA::Host::ConfigureStatsSync(Stats);

                    double LnPrice = 50;
                    CudaShot(&Kernel::Add::StaticValueToRange, Stats, LnPrice, 0, LnStorageSize);
                    CudaShot(&Kernel::Add::Value, Stats, 25);
                    CudaShot(&Kernel::Add::Value, Stats, 60);
                    CudaShot(&Kernel::Add::Value, Stats, 55);

                    CudaShot(&Kernel::Print::MaxMinStoch, Stats);
                    CudaShot(&Kernel::Check::StochEquals, Stats, 85.7143);
                }
                Rescue();
            }

            void RSI()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 20;
                const auto LnLogicalSize = 14;


                auto Stats = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::EStatOpt, xint>({
                        {RA::EStatOpt::RSI, LnLogicalSize} })
                        );
                RA::Host::ConfigureStatsSync(Stats);

                double LnPrice = 50;

                CudaShot(&Kernel::Add::AddDecreasingRSI, Stats, LnPrice, 0, LnStorageSize);
                CudaShot(&Kernel::Print::RSI, Stats);
                CudaShot(&Kernel::Check::RSIEquals, Stats, 25);

                CudaShot(&Kernel::Add::AddIncreasingRSI, Stats, LnPrice, 0, LnStorageSize);
                CudaShot(&Kernel::Print::RSI, Stats);
                CudaShot(&Kernel::Check::RSIEquals, Stats, 75);

                CudaShot(&Kernel::Add::AddDecreasingRSI, Stats, LnPrice, 0, LnStorageSize);
                CudaShot(&Kernel::Print::RSI, Stats);
                CudaShot(&Kernel::Check::RSIEquals, Stats, 25);



                Rescue();
            }

            void StandardDeviation()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 4;
                const auto LnLogicalSize = 4;

                auto Stats = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::EStatOpt, xint>({
                        {RA::EStatOpt::AVG, LnLogicalSize},
                        {RA::EStatOpt::SD, LnLogicalSize}
                        }));
                RA::Host::ConfigureStatsSync(Stats);

                for (xint i = 0; i < 2; i++)
                {
                    CudaShot(&Kernel::Add::Value, Stats, 2);
                    CudaShot(&Kernel::Add::Value, Stats, 5);
                    CudaShot(&Kernel::Add::Value, Stats, 9);
                    CudaShot(&Kernel::Add::Value, Stats, 12);
                }
                CudaShot(&Kernel::Print::StandardDeviation, Stats);
                CudaShot(&Kernel::Print::StandardDeviationOffset, Stats);
                CudaShot(&Kernel::Check::StandardDeviationEquals, Stats, 4.39697);

                Rescue();
            }

            void MeanAbsoluteDeviation()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 6;
                const auto LnLogicalSize = 6;

                auto Stats = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::EStatOpt, xint>({
                        {RA::EStatOpt::AVG, LnLogicalSize},
                        {RA::EStatOpt::MAD, LnLogicalSize}
                        }));
                RA::Host::ConfigureStatsSync(Stats);

                for (xint i = 0; i < 2; i++)
                {
                    CudaShot(&Kernel::Print::MeanAbsoluteDeviation, Stats);
                    CudaShot(&Kernel::Add::Value, Stats, 3);
                    CudaShot(&Kernel::Print::MeanAbsoluteDeviation, Stats);
                    CudaShot(&Kernel::Add::Value, Stats, 3);
                    CudaShot(&Kernel::Print::MeanAbsoluteDeviation, Stats);
                    CudaShot(&Kernel::Add::Value, Stats, 5);
                    CudaShot(&Kernel::Print::MeanAbsoluteDeviation, Stats);
                    CudaShot(&Kernel::Add::Value, Stats, 8);
                    CudaShot(&Kernel::Print::MeanAbsoluteDeviation, Stats);
                    CudaShot(&Kernel::Add::Value, Stats, 8);
                    CudaShot(&Kernel::Print::MeanAbsoluteDeviation, Stats);
                    CudaShot(&Kernel::Add::Value, Stats, 12);
                    CudaShot(&Kernel::Print::MeanAbsoluteDeviation, Stats);
                }
                CudaShot(&Kernel::Print::MeanAbsoluteDeviation, Stats);
                CudaShot(&Kernel::Print::MeanAbsoluteDeviationOffset, Stats);
                CudaShot(&Kernel::Check::MeanAbsoluteDeviationEquals, Stats, 2.83333);

                Rescue();
            }
        }
    }

    namespace Logical
    {
        namespace GPU
        {
            void AVG()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 0;
                const auto LnLogicalSize = 0;

                auto Stats1 = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::EStatOpt, xint>({
                        {RA::EStatOpt::AVG, LnLogicalSize} // half storage size
                        })
                );

                RA::Host::ConfigureStatsSync(Stats1);
                CudaShot(&Kernel::Add::ValuesToRange, Stats1, 1, 20);
                CudaShot(&Kernel::Print::SumAndAVG, Stats1);
                CudaShot(&Kernel::Check::AverageEquals, Stats1, 10.5);
                Rescue();
            }

            void MeanAbsoluteDeviation()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 4;
                const auto LnLogicalSize = 4;

                auto LoStorage = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::EStatOpt, xint>({
                        {RA::EStatOpt::AVG, LnLogicalSize},
                        {RA::EStatOpt::MAD, LnLogicalSize}
                        }));
                RA::Host::ConfigureStatsSync(LoStorage);

                auto LoLogical = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(0,
                    xmap<RA::EStatOpt, xint>({
                        {RA::EStatOpt::AVG, 0},
                        {RA::EStatOpt::MAD, 0}
                        }));
                RA::Host::ConfigureStatsSync(LoLogical);


                auto LoDistributor = RA::Rand::GetDistributor<int>(0, 100);
                for (xint i = 0; i < 100; i++)
                {
                    cvar LnVal = (double)RA::Rand::GetValue(LoDistributor);
                    CudaShot(&Kernel::Add::Value, LoStorage, LnVal);
                    CudaShot(&Kernel::Add::Value, LoLogical, LnVal);
                }

                CudaShot(&Kernel::Print::MeanAbsoluteDeviation, LoStorage);
                CudaShot(&Kernel::Print::MeanAbsoluteDeviation, LoLogical);

                Rescue();
            }

            void StandardDeviation()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 4;
                const auto LnLogicalSize = 4;

                auto LoStorage = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::EStatOpt, xint>({
                        {RA::EStatOpt::AVG, LnLogicalSize},
                        {RA::EStatOpt::SD, LnLogicalSize}
                        }));
                RA::Host::ConfigureStatsSync(LoStorage);

                auto LoLogical = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(0,
                    xmap<RA::EStatOpt, xint>({
                        {RA::EStatOpt::AVG, 0},
                        {RA::EStatOpt::SD, 0}
                        }));
                RA::Host::ConfigureStatsSync(LoLogical);


                auto LoDistributor = RA::Rand::GetDistributor<int>(0, 100);
                for (xint i = 0; i < 100; i++)
                {
                    cvar LnVal = (double)RA::Rand::GetValue(LoDistributor);
                    CudaShot(&Kernel::Add::Value, LoStorage, LnVal);
                    CudaShot(&Kernel::Add::Value, LoLogical, LnVal);
                }

                CudaShot(&Kernel::Print::StandardDeviation, LoStorage);
                CudaShot(&Kernel::Print::StandardDeviation, LoLogical);

                Rescue();
            }
        }
    }

    namespace Miscellaneous
    {
        void Joinery()
        {
            Begin();
            cout << "Joinery on the GPU is Untested!\n";
            Rescue();
        }
    }
}



void RunGPU()
{
    Begin();
    Test::Storage::GPU::AVG();
    Test::Logical::GPU::AVG();

    Test::Storage::GPU::STOCH();
    Test::Storage::GPU::RSI();

    //Test::Miscellaneous::Joinery();

    Test::Storage::GPU::MeanAbsoluteDeviation();
    Test::Logical::GPU::MeanAbsoluteDeviation();

    Test::Storage::GPU::StandardDeviation();
    Test::Logical::GPU::StandardDeviation();

    Rescue();
}

int main()
{
    Nexus<>::Start();

    RunGPU();
    cout << "\n\n";

    Nexus<>::Stop();
    return 0;
}

