// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Macros.h"
#include "xmap.h"

#include "StatsCPU.cuh"
#include "StatsGPU.cuh"

#include "ImportCUDA.cuh"
#include "Host.cuh"
#include "CudaBridge.cuh"

#include "vld.h"

using std::cout;
using std::endl;


namespace Test
{
    namespace Kernel
    {
        static __global__ void AddValuesOneToTweanty(RA::StatsGPU *StatsPtr, const uint LoopCount)
        {
            auto& Stats = *StatsPtr;

            for (int i = 1; i <= LoopCount; i++)
                Stats << i;
        }

        static __global__ void AddValuesForSTOCH_1(RA::StatsGPU* StatsPtr, const uint LoopCount)
        {
            auto& Stats = *StatsPtr;

            for (int i = 1; i <= LoopCount; i++)
                Stats << i;

            Stats << 25;
            Stats << 35;
        }

        static __global__ void AddValuesForSTOCH_2(RA::StatsGPU* StatsPtr, const uint LoopCount)
        {
            auto& Stats = *StatsPtr;

            for (int i = 1; i <= LoopCount; i++)
                Stats << i;

            Stats << 25;
            Stats << 60;
            Stats << 55;
        }

        static __global__ void AddValuesForRSI_Up(RA::StatsGPU* StatsPtr, const uint LoopCount)
        {
            auto& Stats = *StatsPtr;

            double LnPrice = 50;
            for (uint i = 0; i <= Stats.GetStorageSize(); i++)
            {
                if (i % 2)
                    LnPrice -= 1;
                else
                    LnPrice += 3;

                Stats << LnPrice;
            }
        }

        static __global__ void AddValuesForRSI_Down(RA::StatsGPU* StatsPtr, const uint LoopCount)
        {
            auto& Stats = *StatsPtr;

            double LnPrice = 50;
            for (uint i = 0; i <= Stats.GetStorageSize(); i++)
            {
                if (i % 2)
                    LnPrice += 1;
                else
                    LnPrice -= 3;

                Stats << LnPrice;
            }
        }

        static __global__ void PrintAVG(RA::StatsGPU* StatsPtr)
        {
            auto& Stats = *StatsPtr;
            printf("AVG: %lf\n", Stats.AVG().GetAVG());
            printf("SUM: %lf\n", Stats.AVG().GetSum());
            printf("\n");
        }
        static __global__ void PrintSTOCH(RA::StatsGPU* StatsPtr)
        {
            auto& Stats = *StatsPtr;
            printf("MAX: %lf\n", Stats.STOCH().GetMax());
            printf("CUR: %lf\n", Stats.STOCH().GetCurrent());
            printf("MIN: %lf\n", Stats.STOCH().GetMin());
            printf("STO: %lf\n", Stats.STOCH().GetSTOCH());
            printf("\n");
        }
        static __global__ void PrintRSI(RA::StatsGPU* StatsPtr)
        {
            auto& Stats = *StatsPtr;
            printf("RSI: %lf\n", Stats.GetRSI());
            printf("\n");
        }
    }

    namespace Storage
    {
        namespace CPU
        {
            void AVG()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 20;
                const auto LnLogicalSize = 10;

                auto One = RA::StatsCPU(LnStorageSize,
                    {
                        {RA::Stats::EOptions::AVG,   LnLogicalSize},
                    });

                auto PrintVals = [](const RA::StatsCPU& Obj) ->void
                {
                    cout << "AVG: " << Obj.AVG().GetAVG() << endl;
                    cout << "SUM: " << Obj.AVG().GetSum() << endl;
                    cout << endl;
                };

                for (uint i = 1; i <= One.GetStorageSize(); i++)
                    One << i;

                auto Two = One;
                PrintVals(One);
                PrintVals(Two);
                Rescue();
            }

            void STOCH()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 20;
                const auto LnLogicalSize = 10;

                auto Stoch = RA::StatsCPU(LnStorageSize,
                    {
                        {RA::Stats::EOptions::STOCH, LnLogicalSize}
                    });


                {
                    double LnPrice = 50;
                    for (uint i = 0; i <= Stoch.GetStorageSize(); i++)
                        Stoch << LnPrice;

                    Stoch << 25;
                    Stoch << 35;

                    // will be under 50 because closing 35 is under the ATH of 50
                    cout << "MAX: " << Stoch.STOCH().GetMax() << endl;
                    cout << "MIN: " << Stoch.STOCH().GetMin() << endl;
                    cout << "STO: " << Stoch.STOCH().GetSTOCH() << endl;
                    cout << endl;
                }
                {
                    double LnPrice = 50;
                    for (uint i = 0; i <= Stoch.GetStorageSize(); i++)
                        Stoch << LnPrice;

                    Stoch << 25;
                    Stoch << 60;
                    Stoch << 55;

                    cout << "MAX: " << Stoch.STOCH().GetMax() << endl;
                    cout << "MIN: " << Stoch.STOCH().GetMin() << endl;
                    cout << "STO: " << Stoch.STOCH().GetSTOCH() << endl;
                }

                Rescue();
            }

            void RSI()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 20;
                const auto LnLogicalSize = 14;

                auto RSI = RA::StatsCPU(LnStorageSize,
                    {
                        {RA::Stats::EOptions::RSI, LnLogicalSize}
                    });


                {
                    double LnPrice = 50;
                    for (uint i = 0; i <= RSI.GetStorageSize(); i++)
                    {
                        if (i % 2)
                            LnPrice += 1;
                        else
                            LnPrice -= 3;

                        if (LnPrice < 0)
                            ThrowIt("Price below zero");

                        RSI << LnPrice;
                    }
                    cout << "RSI: " << RSI.GetRSI() << endl;
                }
                {
                    double LnPrice = 50;
                    for (uint i = 0; i <= RSI.GetStorageSize(); i++)
                    {
                        if (i % 2)
                            LnPrice += 3;
                        else
                            LnPrice -= 1;

                        if (LnPrice < 0)
                            ThrowIt("Price below zero");

                        RSI << LnPrice;
                    }
                    cout << "RSI: " << RSI.GetRSI() << endl;
                }
                Rescue();
            }
        }

        namespace GPU
        {
            void AVG()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 20;
                const auto LnLogicalSize = 10;
                // note: if storage size is 0, logical size must also be 0
                // logical size increases indefinatly if storage size is 0
                auto AvgGPU = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::Stats::EOptions, uint>({
                        {RA::Stats::EOptions::AVG, LnLogicalSize}
                        })
                    );

                const auto LnLogicalLoopSize = 20;

                RA::Host::ConfigureStatsSync(AvgGPU);

                RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::AddValuesOneToTweanty, AvgGPU, LnLogicalLoopSize);
                RA::CudaBridge<>::SyncAll();

                RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::PrintAVG, AvgGPU);
                RA::CudaBridge<>::SyncAll();
                Rescue();
            }

            void STOCH()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 20;
                const auto LnLogicalSize = 10;
                // note: if storage size is 0, logical size must also be 0
                // logical size increases indefinatly if storage size is 0
                auto Stoch = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::Stats::EOptions, uint>({
                        {RA::Stats::EOptions::STOCH, LnLogicalSize}
                        })
                    );

                const auto LnLogicalLoopSize = 20;

                RA::Host::ConfigureStatsSync(Stoch);

                RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::AddValuesForSTOCH_1, Stoch, LnLogicalLoopSize);
                RA::CudaBridge<>::SyncAll();

                RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::PrintSTOCH, Stoch);
                RA::CudaBridge<>::SyncAll();

                RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::AddValuesForSTOCH_2, Stoch, LnLogicalLoopSize);
                RA::CudaBridge<>::SyncAll();

                RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::PrintSTOCH, Stoch);
                RA::CudaBridge<>::SyncAll();
                Rescue();
            }

            void RSI()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 20;
                const auto LnLogicalSize = 10;
                // note: if storage size is 0, logical size must also be 0
                // logical size increases indefinatly if storage size is 0
                auto RSI = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                    xmap<RA::Stats::EOptions, uint>({
                        {RA::Stats::EOptions::RSI, LnLogicalSize}
                        })
                    );

                const auto LnLogicalLoopSize = 20;

                RA::Host::ConfigureStatsSync(RSI);

                cout << "RSI UP" << endl;
                RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::AddValuesForRSI_Up, RSI, LnLogicalLoopSize);
                RA::CudaBridge<>::SyncAll();

                RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::PrintRSI, RSI);
                RA::CudaBridge<>::SyncAll();


                cout << "RSI DOWN" << endl;
                RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::AddValuesForRSI_Down, RSI, LnLogicalLoopSize);
                RA::CudaBridge<>::SyncAll();

                RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::PrintRSI, RSI);
                RA::CudaBridge<>::SyncAll();
                Rescue();
            }
        }
    }

    namespace Logical
    {
        namespace CPU
        {
            void AVG()
            {
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 0;
                const auto LnLogicalSize = 0;
                // note: if storage size is 0, logical size must also be 0
                // logical size increases indefinatly if storage size is 0

                auto One = RA::StatsCPU(LnStorageSize,
                    {
                        {RA::Stats::EOptions::AVG, LnLogicalSize}
                    });

                auto PrintVals = [](const RA::StatsCPU& Obj) ->void
                {
                    cout << "AVG: " << Obj.AVG().GetAVG() << endl;
                    cout << "SUM: " << Obj.AVG().GetSum() << endl;
                    cout << endl;
                };

                for (uint i = 1; i <= 20; i++)
                    One << i;

                PrintVals(One);
            }
        }

        namespace GPU
        {
            void AVG()
            {
                Begin();
                {
                    cout << '\n' << __CLASS__ << '\n';

                    const auto LnStorageSize = 0;
                    const auto LnLogicalSize = 0;
                    // note: if storage size is 0, logical size must also be 0
                    // logical size increases indefinatly if storage size is 0
                    auto AvgGPU = RA::Host::AllocateObjOnDevice<RA::StatsGPU>(LnStorageSize,
                        xmap<RA::Stats::EOptions, uint>({
                            {RA::Stats::EOptions::AVG, LnLogicalSize}
                            })
                        );

                    const auto LnLogicalLoopSize = 20;

                    RA::Host::ConfigureStatsSync(AvgGPU);

                    RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::AddValuesOneToTweanty, AvgGPU, LnLogicalLoopSize);
                    RA::CudaBridge<>::SyncAll();

                    RA::CudaBridge<>::NONE::RunGPU(dim3(1), dim3(1), &Test::Kernel::PrintAVG, AvgGPU);
                    RA::CudaBridge<>::SyncAll();
                }
                Rescue();
            }
        }
    }
}

void RunCPU()
{
    Begin();
    Test::Storage::CPU::AVG();
    Test::Logical::CPU::AVG();

    Test::Storage::CPU::RSI();
    Test::Storage::CPU::STOCH();
    Rescue();
}

void RunGPU()
{
    Begin();
    Test::Storage::GPU::AVG();
    Test::Logical::GPU::AVG();

    Test::Storage::GPU::RSI();
    Test::Storage::GPU::STOCH();
    Rescue();
}

int main() 
{
    Nexus<>::Start();
    Begin();

    //RunCPU();
    RunGPU();

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}

