// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Macros.h"
#include "xmap.h"

#include "StatsCPU.h"
#include "Normals.h"

#include <algorithm>
#include <iomanip>

// #include "vld.h"

using std::cout;
using std::endl;


namespace Test
{
    istatic double SnComp = 1.2;
    namespace Storage
    {
        namespace CPU
        {
            void Getters()
            {
                const auto LnStorage = 5;
                auto LoStat = RA::StatsCPU(LnStorage, { {} });
                xvector<xint> LvVec;
                for (xint i = 0; i < LnStorage * 5; i++)
                {
                    LoStat << i;
                    LvVec  << i;
                    const auto LnInsertIdx = LoStat.GetInsertIdx();
                    cout << "Insert Idx : Value >> (" << LnInsertIdx << ":" << i << ')' << endl;
                    if (i < LnStorage)
                        continue;
                    if (LvVec.Size() >= LnStorage)
                        LvVec.erase(LvVec.begin());
                    for (xint j = 0; j < LnStorage; j++)
                    {
                        cvar LnStatStart = LoStat.ValueFromStart(j);
                        cvar LnStatEnd = LoStat.ValueFromEnd(j);

                        cvar LnVecFront = LvVec.First(j);
                        cvar LnVecEnd = LvVec.Last(j);

                        //cout << "First: " << LnStatStart << " <> " << LnVecFront << endl;
                        //cout << "Last:  " << LnStatEnd << " <> " << LnVecEnd << endl;

                        AssertEqual(LnStatStart, LnVecFront);
                        AssertEqual(LnStatEnd, LnVecEnd);
                    }
                }
                cout << endl;
            }

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
                        cout << Two.ValueFromEnd(i) << endl;
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
                    for (xint i = 0; i < 4; i++)
                        Stoch << LnPrice;

                    Stoch << 20;
                    Stoch << 23;

                    // will be under 50 because closing 35 is under the ATH of 50
                    cout << "MAX: " << Stoch.STOCH().GetMax() << endl;
                    cout << "MIN: " << Stoch.STOCH().GetMin() << endl;
                    cout << "STO: " << Stoch.STOCH().GetSTOCH() << endl;
                    cout << endl;
                }

                if (Stoch.GetSTOCH() != 10)
                {
                    for (xint i = 0; i < Stoch.GetStorageSize(); i++)
                        cout << Stoch.ValueFromEnd(i) << endl;
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
                if (!RA::Appx(85.7143, Stoch.GetSTOCH(), 0.0001))
                    ThrowIt("Bad STOCH 2 Calc: ");

                Rescue();
            }

            void Normals()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 10;
                auto LoNormals = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::Normals });
                xvector<double> LvValues = { 1,2,3,4,5,6,7,8,9,10 };
                for (auto& Val : LvValues)
                    LoNormals << Val;
                for (auto& LnNormal : LoNormals.GetNormals())
                    cout << LnNormal << " <> " << LoNormals.Normals().ToRawLinear(LnNormal) << endl;
                auto LnVal = LvValues.Last();
                for (xint i = 0; i < 10; i++)
                    LoNormals << (LnVal + i);
                for (auto& LnNormal : LoNormals.GetNormals())
                    cout << LnNormal << " <> " << LoNormals.Normals().ToRawLinear(LnNormal) << endl;
                Rescue();
            }

            void Omaha()
            {
                {
                    auto LvVals = std::vector{ 1,2,3,4,5,4,3,4,5,6,7,6,5 };
                    auto LoStats = RA::StatsCPU(LvVals.size(), { RA::EStatOpt::Omaha });
                    for (auto Val : LvVals)
                    {
                        LoStats << Val;
                        cout << Val
                            << " Low  Idx: " << LoStats.Omaha().OldIndexFor(LoStats.Omaha().GetMin())
                            << " High Idx: " << LoStats.Omaha().OldIndexFor(LoStats.Omaha().GetMax())
                            << " Running:  " << LoStats.Omaha().GetRunningSize() - 1
                            << endl;

                    }
                    cout << "High: " << LoStats.Omaha().BxHigh() << endl;
                    cout << "Low:  " << LoStats.Omaha().BxLow() << endl;
                }
                {
                    auto LvVals  = std::vector{ 1,2,3,4,5,4,3,4,5,6,7,6,5 };
                    auto LoStats = RA::StatsCPU(LvVals.size(), { RA::EStatOpt::Omaha });
                    for (auto Val : LvVals)
                    {
                        LoStats << Val;
                        cout << Val
                            << " Low  Idx: " << LoStats.Omaha().GetLowIdxScaled()
                            << " High Idx: " << LoStats.Omaha().GetHighIdxScaled()
                            << " Running:  " << LoStats.Omaha().GetRunningSize()
                            << endl;
                    }
                    cout << "High: " << LoStats.Omaha().BxHigh() << endl;
                    cout << "Low:  " << LoStats.Omaha().BxLow() << endl;
                }
            }

            class Boundaries
            {
            public:
                istatic xdbl SnLinearLoss = 0.0L;
                istatic xdbl SnLogLoss = 0.0L;
                istatic void AssertLinear(const xdbl FVal, const xdbl FMin, const xdbl FMax)
                {
                    cvar LnMidpoint = (FMin + FMax) / 2.0L;
                    auto LnNormal = RA::Normals::ToNormalLinear(FVal, RA::Normals::Config(SnComp, FMin, FMax));
                    auto LnRaw    = RA::Normals::ToRawLinear(LnNormal, RA::Normals::Config(SnComp, FMin, FMax));
                    AssertEqualDbl(FVal, LnRaw, FMin, FMax);
                    if (FVal < FMin)
                        assert(LnNormal < -1.0 /SnComp);
                    if (FVal > FMax)
                        assert(LnNormal > 1.0 / SnComp);
                    if (FVal > LnMidpoint)
                        assert(LnNormal > 0);
                    if (FVal < LnMidpoint)
                        assert(LnNormal < 0);

                    SnLinearLoss += RA::Abs(FVal - LnRaw);
                }

                istatic void AssertLog(const xdbl FVal, const xdbl FMin, const xdbl FMax)
                {
                    cvar LnMidpoint = (FMin + FMax) / 2.0L;
                    auto LnNormal = RA::Normals::ToNormalLinear(FVal, RA::Normals::Config(SnComp, FMin, FMax));
                    auto LnRaw = RA::Normals::ToRawLinear(LnNormal, RA::Normals::Config(SnComp, FMin, FMax));
                    AssertEqualDbl(FVal, LnRaw, FMin, FMax);
                    if (FVal < FMin)
                        assert(LnNormal < -1.0 / SnComp);
                    if (FVal > FMax)
                        assert(LnNormal > 1.0 / SnComp);
                    if (FVal > LnMidpoint)
                        assert(LnNormal > 0);
                    if (FVal < LnMidpoint)
                        assert(LnNormal < 0);

                    SnLogLoss += RA::Abs(FVal - LnRaw);
                }
            };

            void NormalLinearBoundaries()
            {
                assert(RA::Appx(5.0, 5.0) == true);
                assert(RA::Appx(-5.0, -5.0) == true);
                assert(RA::Appx(-5.0, 5.0) == false);

                auto LoMin = -5.0;
                auto LoMax = 5.0;

                cout << std::fixed << std::setprecision(12) << endl;
                cout << "-6 : " << RA::Normals::ToNormalLinear(-6, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-5 : " << RA::Normals::ToNormalLinear(-5, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-4 : " << RA::Normals::ToNormalLinear(-4, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-1 : " << RA::Normals::ToNormalLinear(-1, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << " 0 : " << RA::Normals::ToNormalLinear(0, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+1 : " << RA::Normals::ToNormalLinear(1, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+4 : " << RA::Normals::ToNormalLinear(4, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+5 : " << RA::Normals::ToNormalLinear(5, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+6 : " << RA::Normals::ToNormalLinear(6, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-----" << endl;

                cout << "-6 : " << RA::Normals::ToRawLinear(-1.2, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-5 : " << RA::Normals::ToRawLinear(-1, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-4 : " << RA::Normals::ToRawLinear(-0.8, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-1 : " << RA::Normals::ToRawLinear(-0.2, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << " 0 : " << RA::Normals::ToRawLinear(0, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+1 : " << RA::Normals::ToRawLinear(0.2, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+4 : " << RA::Normals::ToRawLinear(0.8, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+5 : " << RA::Normals::ToRawLinear(1, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+6 : " << RA::Normals::ToRawLinear(1.2, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;


                {
                    // equal min/max under/over
                    const auto LnMin = -5.0L;
                    const auto LnMax = 5.0L;
                    for (auto LnVal = LnMin - 1.0L; LnVal < LnMax + 2.0L; LnVal++) {
                        Boundaries::AssertLinear(LnVal, LnMin, LnMax);
                    }
                }
                {
                    // both neg
                    const auto LnMin = -11.0L;
                    const auto LnMax = -1.0L;
                    for (auto LnVal = LnMin - 1.0L; LnVal < LnMax + 2.0L; LnVal++) {
                        Boundaries::AssertLinear(LnVal, LnMin, LnMax);
                    }
                }
                {
                    // both pos
                    const auto LnMin = 11.0L;
                    const auto LnMax = 1.0L;
                    for (auto LnVal = LnMin - 1.0L; LnVal < LnMax + 2.0L; LnVal++) {
                        Boundaries::AssertLinear(LnVal, LnMin, LnMax);
                    }
                }
                {
                    // more neg
                    const auto LnMin = -15.0L;
                    const auto LnMax = 5.0L;
                    for (auto LnVal = LnMin - 1.0L; LnVal < LnMax + 2.0L; LnVal++) {
                        Boundaries::AssertLinear(LnVal, LnMin, LnMax);
                    }
                }
                {
                    // more pos
                    const auto LnMin = -5.0L;
                    const auto LnMax = 15.0L;
                    for (auto LnVal = LnMin - 1.0L; LnVal < LnMax + 2.0L; LnVal++) {
                        Boundaries::AssertLinear(LnVal, LnMin, LnMax);
                    }
                }
                {
                    // more pos
                    const auto LnMin = -1e10L;
                    const auto LnMax = 1e4L;
                    for (auto LnVal = LnMax - 100000.0L; LnVal < LnMax + 100000.0L; LnVal++) {
                        Boundaries::AssertLinear(LnVal, LnMin, LnMax);
                    }
                }
                cout << endl;
                cout << "Loss: " << Boundaries::SnLinearLoss << endl;
            }

            void NormalLogBoundaries()
            {
                assert(RA::Appx(5.0, 5.0) == true);
                assert(RA::Appx(-5.0, -5.0) == true);
                assert(RA::Appx(-5.0, 5.0) == false);

                auto LoMin = -5.0;
                auto LoMax = 5.0;

                cout << std::fixed << std::setprecision(12) << endl;
                cout << "-6 : " << RA::Normals::ToNormalLog(-6, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-5 : " << RA::Normals::ToNormalLog(-5, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-4 : " << RA::Normals::ToNormalLog(-4, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-1 : " << RA::Normals::ToNormalLog(-1, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << " 0 : " << RA::Normals::ToNormalLog(0, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+1 : " << RA::Normals::ToNormalLog(1, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+4 : " << RA::Normals::ToNormalLog(4, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+5 : " << RA::Normals::ToNormalLog(5, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+6 : " << RA::Normals::ToNormalLog(6, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-----" << endl;

                cout << "-6 : " << RA::Normals::ToRawLog(-1.2, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-5 : " << RA::Normals::ToRawLog(-1, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-4 : " << RA::Normals::ToRawLog(-0.8, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "-1 : " << RA::Normals::ToRawLog(-0.2, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << " 0 : " << RA::Normals::ToRawLog(0, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+1 : " << RA::Normals::ToRawLog(0.2, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+4 : " << RA::Normals::ToRawLog(0.8, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+5 : " << RA::Normals::ToRawLog(1, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;
                cout << "+6 : " << RA::Normals::ToRawLog(1.2, RA::Normals::Config(SnComp, LoMin, LoMax)) << endl;


                {
                    // equal min/max under/over
                    const auto LnMin = -5.0L;
                    const auto LnMax = 5.0L;
                    for (auto LnVal = LnMin - 1.0L; LnVal < LnMax + 2.0L; LnVal++) {
                        Boundaries::AssertLog(LnVal, LnMin, LnMax);
                    }
                }
                {
                    // both neg
                    const auto LnMin = -11.0L;
                    const auto LnMax = -1.0L;
                    for (auto LnVal = LnMin - 1.0L; LnVal < LnMax + 2.0L; LnVal++) {
                        Boundaries::AssertLog(LnVal, LnMin, LnMax);
                    }
                }
                {
                    // both pos
                    const auto LnMin = 11.0L;
                    const auto LnMax = 1.0L;
                    for (auto LnVal = LnMin - 1.0L; LnVal < LnMax + 2.0L; LnVal++) {
                        Boundaries::AssertLog(LnVal, LnMin, LnMax);
                    }
                }
                {
                    // more neg
                    const auto LnMin = -15.0L;
                    const auto LnMax = 5.0L;
                    for (auto LnVal = LnMin - 1.0L; LnVal < LnMax + 2.0L; LnVal++) {
                        Boundaries::AssertLog(LnVal, LnMin, LnMax);
                    }
                }
                {
                    // more pos
                    const auto LnMin = -5.0L;
                    const auto LnMax = 15.0L;
                    for (auto LnVal = LnMin - 1.0L; LnVal < LnMax + 2.0L; LnVal++) {
                        Boundaries::AssertLog(LnVal, LnMin, LnMax);
                    }
                }
                {
                    // more pos
                    const auto LnMin = -1e10L;
                    const auto LnMax = 1e4L;
                    for (auto LnVal = LnMax - 100000.0L; LnVal < LnMax + 100000.0L; LnVal++) {
                        Boundaries::AssertLog(LnVal, LnMin, LnMax);
                    }
                }
                cout << endl;
                cout << "Loss: " << Boundaries::SnLogLoss << endl;
            }

            void RSI_STOCH()
            {
                Begin();
                cout << '\n' << __CLASS__ << '\n';

                const auto LnStorageSize = 5;
                auto LoMore = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::RSI, RA::EStatOpt::STOCH });
                auto LoLess = RA::StatsCPU(LnStorageSize, { RA::EStatOpt::RSI, RA::EStatOpt::STOCH });

                auto LvMore = xvector<double>{ 624.45, 23451.2, 34525.0,    198.9, 202.13, 201.28, 204.11, 202.59 };
                auto LvLess = xvector<double>{ 198.9, 202.13, 201.28, 204.11, 202.59 };

                for (xint i = 0; i < LvLess.Size(); i++)
                    LoLess << LvLess[i];

                for (xint i = 0; i < LvMore.Size(); i++)
                    LoMore << LvMore[i];

                cout << LoLess.GetSTOCH() << "    " << LoLess.GetRSI() << endl;
                cout << LoMore.GetSTOCH() << "    " << LoMore.GetRSI() << endl;

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
                            cout << RSI.ValueFromEnd(i) << endl;
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
                    if (!RA::Appx(4.39697, One.SD().GetDeviation(), 0.00001))
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
                    if (!RA::Appx(2.8333, One.MAD().GetDeviation(), 0.0001))
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
                auto LnLast = 0.0;
                auto LoDistributor = RA::Rand::GetDistributor<int>(0, 100);

                // average dividor 

                // absolute offset then use neg if 

                auto LvVals = xvector<double>{};
                RA::Print("Val : Deviation : Directional : Difference");
                for (xint i = 0; i < 100; i++)
                {
                    LnVal = (double)RA::Rand::GetValue(LoDistributor);
                    if (RA::Rand::GetValue(LoDistributor) < 50)
                        LnVal *= -1;

                    LvVals << LnVal;

                    LoStorage << LnVal;

                    LoLogical.MAD().SetCurrentValue(LnLast); // this works but not req
                    LoLogical << LnVal;


                    cvar LnDeviation   = LoLogical.MAD().GetDeviation();
                    cvar LnDirectional = LoLogical.MAD().GetDirectionalOffset();
                    cvar LnDifferece   = LoLogical.MAD().GetDifference();


                    RA::Print(LnVal, " : ", LnDeviation, " : ", LnDirectional, " : ", LnDifferece);

                    //auto LnDiff = LoLogical.MAD().GetDeviation() / LoStorage.MAD().GetDeviation();
                    //RA::Print(LnVal, ">> ", LoStorage.MAD().GetDeviation(), " <> ", LoLogical.MAD().GetDeviation(), " >> ", LnDiff);
                    //RA::Print(LnVal, ">> ", LoStorage.MAD().GetOffset(), " <> ", LoLogical.MAD().GetOffset(), " <> ", LoLogical.MAD().GetAvgOffset());
                    //cout << '\n';

                    auto LnLast = LnVal;
                }

                if(LvVals.Size())
                    cout << LvVals.GetString() << endl;

                //// Note: Has an accuracy up to the 0.5 from 100
                //if (!RA::Appx(LoStorage.MAD().GetDeviation(), LoLogical.MAD().GetDeviation(), 2)) 
                //    ThrowIt("No Equate: ", LoStorage.MAD().GetDeviation(), " != ", LoLogical.MAD().GetDeviation());

                //if (!RA::Appx(LoStorage.MAD().GetOffset(), LoLogical.MAD().GetOffset(), 2))
                //    ThrowIt("No Equate: ", LoStorage.MAD().GetOffset(), " != ", LoLogical.MAD().GetOffset());

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
                SS << Stats.ValueFromEnd(Idx);
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
                    cout << " (+" << i << ") " << Stats.ValueFromEnd(0) << ' ';
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
                    cout << " (+" << i << ") " << Stats.ValueFromEnd(0) << ' '; // storage size is 1
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
                    cout << ' ' << i << '(' 
                        << Stats.ValueFromEnd(0) << '-' 
                        << Stats.ValueFromEnd(1) << '-' 
                        << Stats.ValueFromEnd(2) << ')';
                    if (i % 5 == 0)
                        cout << endl;
                }
                CheckVal(Stats, "60", false);
                CheckVal(Stats, "40", false, 1);
                cout << endl;
                for (xint i = 1; i <= 5*2; i++)
                {
                    Stats << 0;
                    cout << i << '('
                        << Stats.ValueFromEnd(0) << '-'
                        << Stats.ValueFromEnd(1) << '-'
                        << Stats.ValueFromEnd(2) << ')';
                    if (i % 5 == 0)
                        cout << endl;
                }
                CheckVal(Stats, "0", false);
                CheckVal(Stats, "0", false, 1);

                cout << endl;
                cout << Stats.ValueFromEnd(0) << ' ' << Stats.ValueFromEnd(1) << ' ' << Stats.ValueFromEnd(0) << endl;
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
                cout << '('
                    << LoStorage.ValueFromEnd(0) << '-'
                    << LoStorage.ValueFromEnd(1) << '-'
                    << LoStorage.ValueFromEnd(2) << ')'
                    << '\n';
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

    Test::Storage::CPU::Normals();
    Test::Storage::CPU::NormalLinearBoundaries();
    Test::Storage::CPU::NormalLogBoundaries();

    Test::Storage::CPU::Omaha();

    Test::Storage::CPU::RSI_STOCH();

    Test::Miscellaneous::Joinery();
    Test::Miscellaneous::Slippage();

    cout << "----------------------------------------------------" << endl;
    Test::Storage::CPU::Getters();

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

