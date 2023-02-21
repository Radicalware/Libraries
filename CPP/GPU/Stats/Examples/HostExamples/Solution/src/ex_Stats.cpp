// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Macros.h"
#include "xmap.h"

#include "StatsCPU.h"

#include "vld.h"

using std::cout;
using std::endl;


namespace Test
{
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
                {
                    One << i;
                    //cout << CYAN   << "Price: " << i << endl;
                    //cout << YELLOW << "AVG:   " << One.GetAVG() << "\n\n";
                }

                PrintVals(One);
                if (One.GetAVG() != 15.5)
                    ThrowIt("Bad AVG 1 Calc");

                cout << WHITE;
                auto Two = One;

                Two.Construct(LnStorageSize, { {RA::Stats::EOptions::AVG,   LnStorageSize} });
                for (uint i = 1; i <= One.GetStorageSize() * 2; i++)
                {
                    Two << i;
                    //cout << CYAN << "Price: " << i << endl;
                    //cout << YELLOW << "AVG:   " << Two.GetAVG() << "\n\n";
                }

                if (Two.GetAVG() != 30.5)
                    ThrowIt("Bad AVG 2 Calc");

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
                if (Stoch.GetSTOCH() != 40)
                    ThrowIt("Bad STOCH 1 Calc");
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
                if (!RA::Appx(85.7143, Stoch.GetSTOCH()))
                    ThrowIt("Bad STOCH 2 Calc: ");
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
                    RSI.Reset();
                    double LnPrice = 50;
                    for (uint i = 0; i <= RSI.GetStorageSize(); i++)
                    {
                        if (i % 2)
                            LnPrice += 1;
                        else
                            LnPrice -= 3;

                        if (LnPrice < 0)
                            ThrowIt("Price below zero");

                        //cout << LnPrice << endl;
                        RSI << LnPrice;
                        //cout << CYAN   << "Price: " << LnPrice << endl;
                        //cout << YELLOW << "RSI:   " << RSI.GetRSI() << "\n\n";
                    }
                    cout << WHITE << "RSI: " << RSI.GetRSI() << endl;
                    if (RSI.GetRSI() != 25)
                        ThrowIt("Bad RSI 2 Calc");
                }
                {
                    //RSI.Reset();
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
                        //cout << CYAN   << "Price: " << LnPrice << endl;
                        //cout << YELLOW << "RSI:   " << RSI.GetRSI() << "\n\n";;
                    }
                    cout << WHITE << "RSI: " << RSI.GetRSI() << endl;
                    if (RSI.GetRSI() != 75)
                        ThrowIt("Bad RSI 2 Calc");
                }
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
                {
                    One << i;
                    //cout << CYAN   << "Price: " << i << endl;
                    //cout << YELLOW << "AVG:   " << One.GetAVG() << "\n\n";
                }

                PrintVals(One);
                if (One.GetAVG() != 10.5)
                    ThrowIt("Bad AVG Calc");
            }
        }
    }

    namespace Miscellaneous
    {
        void Joinery()
        {

            auto CheckVal = [](const RA::StatsCPU& Stats, const xstring& Val)->void
            {
                std::ostringstream SS;
                SS << Stats[0];
                cout << SS.str() << endl;
                if (SS.str() != Val)
                    ThrowIt("Does not Equate!");
            };

            {
                auto Stats = RA::StatsCPU(1, {});
                Stats.SetJoinerySize(0);

                for (uint i = 0; i < 5 * 5; i++)
                    Stats << 1;
                CheckVal(Stats, "1");

                for (uint i = 0; i < 5 * 5; i++)
                    Stats << 2; // 2 * 5 = 10
                CheckVal(Stats, "2");
            }
            {
                auto Stats = RA::StatsCPU(1, {});
                Stats.SetJoinerySize(1);

                for (uint i = 0; i < (5 * 5); i++)
                    Stats << 1;
                CheckVal(Stats, "1");

                for (uint i = 0; i < 5 * 5; i++)
                    Stats << 2; // 2 * 5 = 10
                CheckVal(Stats, "2");
            }
            {
                auto Stats = RA::StatsCPU(1, {});
                Stats.SetJoinerySize(5);

                for (uint i = 0; i < (5 * 5); i++)
                    Stats << 1; // 1 * 5 = 5 per join
                CheckVal(Stats, "5");

                for (uint i = 0; i < (5 * 5); i++)
                    Stats << 2; // 2 * 5 = 10 per join
                CheckVal(Stats, "10");
            }
            {
                auto Stats = RA::StatsCPU(1, {}); // 1 value stored
                Stats.SetJoinerySize(5); // 5 values summed per storage

                cout << "Inc 1 \n ";
                for (uint i = 1; i < (5 * 3); i++)
                {
                    Stats << i;
                    cout << " (+" << i << ") " << Stats[0] << ' ';
                    if (i % 5 == 0)
                        cout << endl;
                }
                cout << endl;
                cout << Stats[0] << endl;
            }
            {
                auto Stats = RA::StatsCPU(2, {}); // 2 values stored
                Stats.SetJoinerySize(5); // 5 values summed per storage

                for (uint i = 1; i < (5 * 3); i++) {
                    Stats << i;
                    cout << i << '(' << Stats[0] << '-' << Stats[1] << ')';
                    if (i % 5 == 0)
                        cout << endl;
                }
                cout << endl;
                cout << Stats[0] << ' ' << Stats[1] << endl;
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

    Test::Miscellaneous::Joinery();

    Rescue();
}
int main() 
{
    Nexus<>::Start();
    Begin();

    RunCPU();
    cout << "\n\n";

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}

