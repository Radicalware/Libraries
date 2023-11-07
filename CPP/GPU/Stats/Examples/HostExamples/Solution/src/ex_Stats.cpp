// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Macros.h"
#include "xmap.h"

#include "StatsCPU.h"

// #include "vld.h"

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
                auto One = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::AVG });

                auto PrintVals = [](const RA::StatsCPU& Obj) ->void
                {
                    cout << "AVG: " << Obj.AVG().GetAVG() << endl;
                    cout << "SUM: " << Obj.AVG().GetSum() << endl;
                    cout << endl;
                };

                for (xint i = 1; i <= One.GetStorageSize(); i++)
                    One << i;

                PrintVals(One);
                if (One.GetAVG() != 10.5)
                    ThrowIt("Bad AVG 1 Calc");

                cout << WHITE;
                auto Two = One;
                for (xint i = 1; i <= Two.GetStorageSize() * 2 ; i++)
                    Two << i;

                PrintVals(Two);
                if (Two.GetAVG() != 30.5)
                {
                    for (xint i = 0; i < Two.GetStorageSize(); i++)
                        cout << Two [i] << endl;
                    ThrowIt("Bad AVG 2 Calc");
                }

                Rescue();
            }

            void STOCH()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 20;
                auto Stoch = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::STOCH });


                {
                    double LnPrice = 50;
                    for (xint i = 0; i <= Stoch.GetStorageSize(); i++)
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
                {
                    for (xint i = 0; i < Stoch.GetStorageSize(); i++)
                        cout << Stoch[i] << endl;
                    ThrowIt("Bad STOCH 1 Calc");
                }


                {
                    double LnPrice = 50;
                    for (xint i = 0; i <= Stoch.GetStorageSize(); i++)
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

                {
                    const auto LnStorageSize = 3;
                    auto RSI = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::RSI });

                    RSI << 5;
                    RSI << 6;
                    RSI << 7;
                    if (RSI.GetRSI() != 100)
                        ThrowIt("Bad Call");
                    RSI << 8;
                    RSI << 7;
                    if (RSI.GetRSI() != 50)
                        ThrowIt("Bad Call");
                    RSI << 6;
                    if (RSI.GetRSI() != 0)
                        ThrowIt("Bad Call");
                }

                double LnPrice = 50;
                {
                    const auto LnStorageSize = 20;
                    auto RSI = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::RSI });


                    for (xint i = 0; i <= RSI.GetStorageSize(); i++)
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
                    {
                        for (xint i = 0; i < RSI.GetStorageSize(); i++)
                            cout << RSI[i] << endl;
                        ThrowIt("Bad RSI 2 Calc");
                    }
                }
                {
                    const auto LnStorageSize = 20;
                    auto RSI = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::RSI });

                    double LnPrice = 50;
                    for (xint i = 0; i <= RSI.GetStorageSize(); i++)
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
                {
                    //RSI.Reset();
                    const auto LnStorageSize = 20;
                    auto RSI = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::RSI });

                    for (xint i = 0; i <= RSI.GetStorageSize(); i++)
                    {
                        if (i % 2)
                            LnPrice += 1;
                        else
                            LnPrice -= 3;

                        if (LnPrice < 0)
                            ThrowIt("Price below zero");

                        RSI << LnPrice;
                        //cout << CYAN   << "Price: " << LnPrice << endl;
                        //cout << YELLOW << "RSI:   " << RSI.GetRSI() << "\n\n";;
                    }
                    cout << WHITE << "RSI: " << RSI.GetRSI() << endl;
                    if (RSI.GetRSI() != 25)
                        ThrowIt("Bad RSI 3 Calc");
                }
                Rescue();
            }

            void StandardDeviation()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 4;
                auto One = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::AVG, RA::EStatOpt::SD });

                for (xint i = 0; i < 2; i++)
                {
                    One << 2;
                    One << 5;
                    One << 9;
                    One << 12;

                    cout << "SD:  " << One.SD().GetDeviation() << endl;
                    if (!RA::Appx(4.39697, One.SD().GetDeviation()))
                        ThrowIt("Bad Value: ", One.SD().GetDeviation());
                }
                Rescue();
            }

            void MeanAbsoluteDeviation()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 6;
                auto One = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::AVG, RA::EStatOpt::MAD });
                

                for (xint i = 0; i < 2; i++)
                {
                    cout << "MAD: " << One.MAD().GetDeviation() << endl;
                    One << 3;
                    cout << "MAD: " << One.MAD().GetDeviation() << endl;

                    One << 3;
                    cout << "MAD: " << One.MAD().GetDeviation() << endl;

                    One << 5;
                    cout << "MAD: " << One.MAD().GetDeviation() << endl;

                    One << 8;
                    cout << "MAD: " << One.MAD().GetDeviation() << endl;

                    One << 8;
                    cout << "MAD: " << One.MAD().GetDeviation() << endl;
                    One << 12;

                    cout << "MAD: " << One.MAD().GetDeviation() << endl;
                    if (!RA::Appx(2.8333, One.MAD().GetDeviation()))
                        ThrowIt("Bad Value: ", One.MAD().GetDeviation());
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
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 0;
                auto One = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::AVG });

                auto PrintVals = [](const RA::StatsCPU& Obj) ->void
                {
                    cout << "AVG: " << Obj.AVG().GetAVG() << endl;
                    cout << "SUM: " << Obj.AVG().GetSum() << endl;
                    cout << endl;
                };

                for (xint i = 1; i <= 20; i++)
                    One << i;

                PrintVals(One);
                if (One.GetAVG() != 10.5)
                    ThrowIt("Bad AVG Calc");
                Rescue();
            }

            void MeanAbsoluteDeviation()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                xint LnSize = 100;
                const auto LnStorageSize = LnSize;
                auto LoStorage = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::AVG, RA::EStatOpt::MAD });
                auto LoLogical = RA::StatsCPU(0, { RA::EStatOpt::AVG, RA::EStatOpt::MAD });


                auto LnVal = 0.0;
                auto LoDistributor = RA::Rand::GetDistributor<int>(0, 100);
                for (xint i = 0; i < 100; i++)
                {
                    auto LnLast = LnVal;
                    LnVal = (double)RA::Rand::GetValue(LoDistributor);
                    LoStorage << LnVal;

                    LoLogical.MAD().SetCurrentValue(LnLast); // this works but not req
                    LoLogical << LnVal;

                    //auto LnDiff = LoLogical.MAD().GetDeviation() / LoStorage.MAD().GetDeviation();
                    //RA::Print(LnVal, ">> ", LoStorage.MAD().GetDeviation(), " <> ", LoLogical.MAD().GetDeviation(), " >> ", LnDiff);
                    //RA::Print(LnVal, ">> ", LoStorage.MAD().GetOffset(), " <> ", LoLogical.MAD().GetOffset(), " <> ", LoLogical.MAD().GetAvgOffset());
                    //cout << '\n';
                }

                // Note: Has an accuracy up to the 0.5 from 100
                if (!RA::Appx(LoStorage.MAD().GetDeviation(), LoLogical.MAD().GetDeviation(), 2)) 
                    ThrowIt("No Equate: ", LoStorage.MAD().GetDeviation(), " != ", LoLogical.MAD().GetDeviation());

                if (!RA::Appx(LoStorage.MAD().GetOffset(), LoLogical.MAD().GetOffset(), 2))
                    ThrowIt("No Equate: ", LoStorage.MAD().GetOffset(), " != ", LoLogical.MAD().GetOffset());

                Rescue();
            }

            void StandardDeviation()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                xint LnSize = 100;
                const auto LnStorageSize = LnSize;
                auto LoStorage = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::AVG, RA::EStatOpt::SD });
                auto LoLogical = RA::StatsCPU(0, { RA::EStatOpt::AVG, RA::EStatOpt::SD });

                auto LoDistributor = RA::Rand::GetDistributor<int>(0, 100);
                for (xint i = 0; i < 100; i++)
                {
                    auto LnVal = (double)RA::Rand::GetValue(LoDistributor);
                    LoStorage << LnVal;
                    LoLogical << LnVal;

                    //auto LnDiff = LoLogical.SD().GetDeviation() / LoStorage.SD().GetDeviation();
                    //RA::Print(LnVal, ">> ", LoStorage.SD().GetDeviation(), " <> ", LoLogical.SD().GetDeviation(), " >> ", LnDiff);
                    //RA::Print(LnVal, ">> ", LoStorage.SD().GetOffset(),    " <> ", LoLogical.SD().GetOffset(), " <> ", LoLogical.SD().GetAvgOffset());
                    //cout << '\n';
                }

                // Note: Has an accuracy up to the 0.7 from 100
                if (!RA::Appx(LoStorage.SD().GetDeviation(), LoLogical.SD().GetDeviation(), 2))
                    ThrowIt("No Equate: ", LoStorage.SD().GetDeviation(), " != ", LoLogical.SD().GetDeviation());

                if (!RA::Appx(LoStorage.SD().GetOffset(), LoLogical.SD().GetOffset(), 2))
                    ThrowIt("No Equate: ", LoStorage.SD().GetOffset(), " != ", LoLogical.SD().GetOffset());


                Rescue();
            }
        }
    }

    namespace Miscellaneous
    {
        void Joinery()
        {
            Begin();
            cout << __CLASS__ << endl;

            auto CheckVal = [](const RA::StatsCPU& Stats, const xstring& Val, 
                bool FnPrint = true, xint Idx = 0)->void
            {
                std::ostringstream SS;
                SS << Stats[Idx];
                if(FnPrint)
                    cout << SS.str() << endl;
                if (SS.str() != Val)
                    ThrowIt("Does not Equate!");
            };

            {
                Begin();
                auto Stats = RA::StatsCPU(0, {});
                Stats.SetJoinerySize(0);

                for (xint i = 0; i < 5 * 5; i++)
                    Stats << 1;
                CheckVal(Stats, "1");

                for (xint i = 0; i < 5 * 5; i++)
                    Stats << 2; // 2 * 5 = 10
                CheckVal(Stats, "2");
                Rescue();
            }
            {
                Begin();
                auto Stats = RA::StatsCPU(0, {});
                Stats.SetJoinerySize(1);

                for (xint i = 0; i < (5 * 5); i++)
                    Stats << 1;
                CheckVal(Stats, "1");

                for (xint i = 0; i < 5 * 5; i++)
                    Stats << 2; // 2 * 5 = 10
                CheckVal(Stats, "2");
                Rescue();
            }
            {
                Begin();
                auto Stats = RA::StatsCPU(0, {});
                Stats.SetJoinerySize(5);

                for (xint i = 0; i < (5 * 5); i++)
                    Stats << 1; // 1 * 5 = 5 per join
                CheckVal(Stats, "5");

                for (xint i = 0; i < (5 * 5); i++)
                    Stats << 2; // 2 * 5 = 10 per join
                CheckVal(Stats, "10");
                Rescue();
            }
            {
                Begin();
                auto Stats = RA::StatsCPU(0, {}); // 1 value stored
                Stats.SetJoinerySize(5); // 5 values summed per storage

                int LnCount = 0;
                cout << "Inc 1 \n ";
                for (xint i = 1; i <= (5 * 3); i++)
                {
                    Stats << i;
                    cout << " (+" << i << ") " << Stats [0] << ' ';
                    if (i % 5 == 0)
                    {
                        ++LnCount;
                        if (LnCount == 1) CheckVal(Stats, "15", false);
                        else if (LnCount == 2) CheckVal(Stats, "40", false);
                        else if (LnCount == 3) CheckVal(Stats, "65", false);
                        cout << endl;
                    }
                }

                cout << endl;
                for (xint i = 1; i <= 3; i++)
                {
                    Stats << 0;
                    cout << " (+" << i << ") " << Stats[0] << ' '; // storage size is 1
                }
                CheckVal(Stats, "29", false);

                cout << "\n\n";
                Rescue();
            }
            {
                Begin();
                auto Stats = RA::StatsCPU(3, {}); // 2 values stored
                Stats.SetJoinerySize(5); // 5 values summed per storage

                for (xint i = 1; i < (5 * 3); i++) {
                    Stats << i;
                    cout << ' ' << i << '(' << Stats[0] << '-' << Stats[1] << '-' << Stats[2] << ')';
                    if (i % 5 == 0)
                        cout << endl;
                }
                CheckVal(Stats, "60", false);
                CheckVal(Stats, "40", false, 1);
                cout << endl;
                for (xint i = 1; i <= 5*2; i++)
                {
                    Stats << 0;
                    cout << i << '(' << Stats[0] << '-' << Stats[1] << '-' << Stats[2] << ')';
                    if (i % 5 == 0)
                        cout << endl;
                }
                CheckVal(Stats, "0", false);
                CheckVal(Stats, "0", false, 1);

                cout << endl;
                cout << Stats[0] << ' ' << Stats[1] << ' ' << Stats[2] << endl;
                Rescue();
            }
            Rescue();
        }

        void Slippage()
        {
            const auto LnStorageSize = 3;
            auto LoStorage = RA::StatsCPU(LnStorageSize, { {} });
            LoStorage.SetSlippageSize(4);
            LoStorage.SetJoinerySize(3);

            for (xint i = 1; i < 30; i++)
            {
                LoStorage << i;
                cout << '(' << LoStorage[0] << ':' << LoStorage[1] << ':' << LoStorage[2] << ')' << '\n';
            }

            for (xint i = 0; i < LoStorage.GetSkipDataLeng(); i++)
                cout << LoStorage.GetSkippedNum(i) << ' ' << endl;
        }
    }
}



void RunCPU()
{
    Begin();
    Test::Storage::CPU::AVG();
    Test::Logical::CPU::AVG();

    Test::Storage::CPU::STOCH();
    Test::Storage::CPU::RSI();

    Test::Miscellaneous::Joinery();
    Test::Miscellaneous::Slippage();

    cout << "----------------------------------------------------" << endl;
    Test::Storage::CPU::MeanAbsoluteDeviation();
    Test::Logical::CPU::MeanAbsoluteDeviation();

    Test::Storage::CPU::StandardDeviation();
    Test::Logical::CPU::StandardDeviation();
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

