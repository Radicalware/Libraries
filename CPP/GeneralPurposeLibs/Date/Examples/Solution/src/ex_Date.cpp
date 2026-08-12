
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#include "Nexus.h"
#include "xvector.h"
#include "xstring.h"
#include "Date.h"

using std::cout;
using std::endl;
using RA::Date;

constexpr RA::ETimeZone LOC = RA::ETimeZone::LOC;
constexpr RA::ETimeZone UTC = RA::ETimeZone::UTC;

void PrintDate(Date& FoDate)
{
    Begin();
    cout << "FnDate                = " << FoDate << endl;
    cout << "FnDate.GetStr         = " << FoDate.GetStr() << endl;
    cout << "FnDate.GetEpochTime   = " << FoDate.GetEpochTime() << endl;
    cout << "FnDate.GetNumericTime = " << FoDate.RetStr(RA::ETimeFormat::Numeric) << endl;
    Rescue();
}

void TestDates(Date& One, Date& Two)
{
    Begin();
    if (One == Two)
        xstring("success\n").ToGreen().Print(0);
    else
    {
        xstring("Failed\n").ToRed().Print(0);
        cout << "\tone: " << One << endl;
        cout << "\ttwo: " << Two << endl;
    }

    xstring().ResetColor();
    Rescue();
}

void Loop()
{
    Begin();
    Date  Start(LOC, 2020, 11, 25);
    Date Finish(LOC, 2021,  2 , 3);

    for (Date Dt = Start; Dt.GetEpochTime() < Finish.GetEpochTime(); Dt = Dt.AdjDay(1))
    {
        cout << "Loop By Day = " << Dt << endl;
    }
    Rescue();
}

void Test()
{
    Begin();
    xstring("\r").ResetColor().Print();
    Date LoDate(LOC);
    Date LoLocal1(LOC, LoDate.GetEpochTime());
    TestDates(LoLocal1, LoDate);

    Date LoLocal2(LOC, LoLocal1.GetStr());
    TestDates(LoLocal1, LoLocal2);

    auto& LoLocalStruct = LoLocal2.GetFormat();
    auto LoLocal3 = Date(
        RA::ETimeZone::LOC,
        LoLocalStruct.GetYear(),
        LoLocalStruct.GetMonth(),
        LoLocalStruct.GetDay(),
        LoLocalStruct.GetHour(),
        LoLocalStruct.GetMin(),
        LoLocalStruct.GetSec());

    TestDates(LoLocal2, LoLocal3);

    auto LoLocal4 = Date(LOC, LoLocal3.RetStr(RA::ETimeFormat::Epoch));
    TestDates(LoLocal3, LoLocal4);
    // ------------------------------------------------------------
    auto LoUtc1 = Date(UTC, LoLocal4.GetEpochTime() + LoLocal4.GetFlippedOffsetSeconds());

    auto LoUtc2 = Date(UTC, LoUtc1.GetStr());
    TestDates(LoUtc1, LoUtc2);

    auto& LoUtcStruct = LoUtc2.GetFormat();
    auto LoUtc3 = Date(
        RA::ETimeZone::UTC,
        LoUtcStruct.GetYear(),
        LoUtcStruct.GetMonth(),
        LoUtcStruct.GetDay(),
        LoUtcStruct.GetHour(),
        LoUtcStruct.GetMin(),
        LoUtcStruct.GetSec());

    TestDates(LoUtc2, LoUtc3);

    {
        {
            auto Mod1 = Date(LOC, 2021, 1, 31);
            auto Mod2 = Mod1.Dup().AdjMonth(-1); // 2020, 12, 31
            Mod1.SetYear(2020).SetMonth(12);
            TestDates(Mod1, Mod2);
        }
        {
            auto Mod1 = Date(LOC, 2021, 1, 31);
            auto Mod2 = Mod1.Dup().AdjMonth(-1); // 2020, 12, 31
            Mod1.SetYear(2020).SetMonth(12); // 2020, 12, 31
            TestDates(Mod1, Mod2);
        }
    }
    {
        {
            Date Mod1(LOC, 2020, 12, 25);
            Date Mod2 = Mod1.Dup().AdjYear(1).AdjMonth(1); // 2022, 1, 25
            Mod1.SetYear(2022).SetMonth(1);
            TestDates(Mod1, Mod2);
        }
        {
            Date Mod1(LOC, 2020, 1, 25);
            Date Mod2 = Mod1.Dup().AdjMonth(-1);
            Mod1.SetYear(2019).SetMonth(12);
            TestDates(Mod1, Mod2);
        }
    }
    {
        {
            Date Mod1(LOC, 2020, 12, 25);
            Date Mod2 = Mod1.Dup().AdjMonth(13);
            Mod1.SetYear(2022).SetMonth(1);
            TestDates(Mod1, Mod2);
        }
        {
            Date Mod1(LOC, 2020, 12, 25);
            Date Mod2 = Mod1.Dup().AdjMonth(-13);
            Mod1.SetYear(2019).SetMonth(11);
            TestDates(Mod1, Mod2);
        }
        {
            Date Mod1(LOC, 2020, 12, 25);
            Date Mod2 = Mod1.Dup().AdjMonth(1);
            Mod1.AdjYear(1).AdjMonth(-11);
            TestDates(Mod1, Mod2);
        }
    }
    {
        Date Num(LOC, 2020, 12, 31, 23, 59, 0);
        Date Str(LOC, Num.RetStr(RA::ETimeFormat::DateTime));
        TestDates(Num, Str);
    }
    xstring("").ResetColor().Print();
    Rescue();
}

int main()
{
    Begin();
    Nexus<>::Start();

    RA::Date::SetAutoCreateStrings(true);
    RA::Date::SetAutoFormat(true);

    cout << xstring::White() << endl;

    cout << RA::GetTodayLOC() << endl;
    cout << RA::GetTodayLOC().GetFormat() << endl;

    auto LoDate = RA::Date(RA::ETimeZone::LOC);

    auto LoBackMin = RA::Date(
        RA::ETimeZone::LOC, 
        LoDate.GetEpochTime() - SC<int>(RA::ESecondsIn::Minutes)); // back 60 sec or 1 min
    cout << LoDate.AdjMin(-1) << endl; // All AdjMin are in place (so use Dup() to get a copy)
    cout << LoBackMin << endl;
    if (LoDate != LoBackMin)
    {
        cout << "LoDate.AdjMin(-1) != LoBackMin" << endl;
        cout << LoDate << " != " << LoBackMin << endl;
        ThrowIt("Badd Offset");
    }

    cout << "UTC OFFSET   (" << Date::GetLocalOffsetFromUTC() << ")\n";
    cout << "LOCal:        " << Date(LOC).CreateStr() << "LOCal Time = LOC" << endl;
    cout << "UTC:          " << Date(UTC).ToUTC() << "Universal Time Coordinated = UTC" << endl;
    cout << "Local to UTC: " << Date(UTC, Date(LOC)) << " = From Local back to UTC" << endl;
    cout << "UTC to Local: " << Date(LOC, Date(UTC)) << " = From UTC back to Local" << endl;

    Date LoDate1(LOC);
    cout << "=======================================================\n";
    cout << "Hour Offset = " << LoDate1.GetHoursOffset() << endl;
    cout << "Secs Offset = " << LoDate1.GetSecondsOffset() << endl;
    cout << "=======================================================\n";
    cout << "                     LOCAL TIME \n";
    cout << "=======================================================\n";
    cout << "----------- From Int ---------------------\n";
    PrintDate(LoDate1);
    cout << "----------- From Str ---------------------\n";
    auto LoDate2 = Date(LOC, LoDate1.RetStr(RA::ETimeFormat::DateTime));
    PrintDate(LoDate2);
    TestDates(LoDate1, LoDate2);
    cout << "----------- From EPC ---------------------\n";
    auto LoDate3 = Date(LOC, LoDate2.GetEpochTime());
    PrintDate(LoDate3);
    TestDates(LoDate2, LoDate3);
    cout << "=======================================================\n";
    cout << "                     UTC TIME \n";
    cout << "=======================================================\n";
    cout << "----------- From EPC ---------------------\n";
    auto LoDate4 = Date(
        RA::ETimeZone::UTC, LoDate3.GetEpochTime() + LoDate3.GetFlippedOffsetSeconds()); // from local to UTC
    PrintDate(LoDate4);
    cout << "----------- From Str ---------------------\n";
    auto LoDate5 = Date(RA::ETimeZone::UTC, LoDate4.RetStr(RA::ETimeFormat::DateTime));
    PrintDate(LoDate5);
    cout << "=======================================================\n";
    cout << "                     Convert TIME \n";
    cout << "=======================================================\n";
    auto LoDateToUTC = Date(UTC, 2021, 7, 9, 0, 1, 2); // literal time
    PrintDate(LoDateToUTC);
    cout << '\n';
    auto LoDateToLocal = Date(LOC, LoDateToUTC);
    PrintDate(LoDateToLocal);
    assert(LoDateToUTC.GetEpochTime() != LoDateToLocal.GetEpochTime()); // diff zone means diff epoch
    assert(LoDateToUTC == LoDateToLocal); // but true time is the same
    cout << "=======================================================\n";
    cout << "                     Set Short Time \n";
    cout << "=======================================================\n";
    auto LoShortTimeUTC = Date(UTC, 2021, 7, 9);
    PrintDate(LoShortTimeUTC);
    cout << '\n';
    auto LoShortTimeLocal = Date(LOC, LoShortTimeUTC);
    PrintDate(LoShortTimeLocal);
    TestDates(LoShortTimeUTC, LoShortTimeLocal);
    cout << "=======================================================\n";
    cout << "                     Chrono Time \n";
    cout << "=======================================================\n";
    auto LoStandard = Date(UTC, 2025, 5, 4);
    auto LoMS = RA::Date(UTC, LoStandard.GetChronoMilliseconds());
    auto LoSec = RA::Date(UTC, LoStandard.GetChronoSeconds());
    assert(LoMS == LoSec);
    cout << '\n';
    auto LoMs2 = RA::Date(UTC, LoSec.GetEpochTimeEvenMilliseconds());
    assert(LoSec == LoMs2);
    PrintDate(LoMs2);
    cout << "=======================================================\n";;
    cout << "Date Standard: " << RA::GetTodayLOC().GetStr() << endl;
    cout << "Date Full:     " << RA::GetTodayLOC().NewStr(RA::ETimeFormat::Weekday) << endl;
    cout << "=======================================================\n";

    Loop();
    Test();

    FinalRescue();
    Nexus<>::Stop();
    return 0;
}
