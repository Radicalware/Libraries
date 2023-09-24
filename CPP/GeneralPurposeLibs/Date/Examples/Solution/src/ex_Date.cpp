
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#include "Nexus.h"
#include "xvector.h"
#include "xstring.h"
#include "Date.h"

using std::cout;
using std::endl;
using RA::Date;

void PrintDate(Date& FoDate)
{
    Begin();
    cout << "FnDate                = " << FoDate << endl;
    cout << "FnDate.GetStr         = " << FoDate.GetStr() << endl;
    cout << "FnDate.GetEpochTime   = " << FoDate.GetEpochTime() << endl;
    cout << "FnDate.GetNumericTime = " << FoDate.GetNumericTimeStr() << endl;
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
    Date  Start(2020, 11, 25);
    Date Finish(2021,  2 , 3);

    for (Date Dt = Start; Dt.GetEpochTime() < Finish.GetEpochTime(); Dt = Dt.Day(1))
    {
        cout << "Loop By Day = " << Dt << endl;
    }
    Rescue();
}

void Test()
{
    Begin();
    xstring("\r").ResetColor().Print();
    Date LoDate;
    Date LoLocal1(LoDate.GetEpochTime());
    TestDates(LoLocal1, LoDate);

    Date LoLocal2(LoLocal1.GetStr());
    TestDates(LoLocal1, LoLocal2);

    auto& LoLocalStruct = LoLocal2.GetLayout();
    Date LoLocal3(
        LoLocalStruct.Year,
        LoLocalStruct.Month,
        LoLocalStruct.Day,
        LoLocalStruct.Hour,
        LoLocalStruct.Min,
        LoLocalStruct.Sec,
        Date::Offset::None);

    TestDates(LoLocal2, LoLocal3);

    Date LoLocal4(LoLocal3.GetEpochTime());
    TestDates(LoLocal3, LoLocal4);
    // ------------------------------------------------------------
    Date LoUtc1(LoLocal4.GetEpochTime(), Date::Offset::ToUTC);

    Date LoUtc2(LoUtc1.GetStr());
    TestDates(LoUtc1, LoUtc2);

    auto& LoUtcStruct = LoUtc2.GetLayout();
    Date LoUtc3(
        LoUtcStruct.Year,
        LoUtcStruct.Month,
        LoUtcStruct.Day,
        LoUtcStruct.Hour,
        LoUtcStruct.Min,
        LoUtcStruct.Sec,
        Date::Offset::None);

    TestDates(LoUtc2, LoUtc3);

    Date LoUtc4(LoUtc3.GetEpochTime());
    TestDates(LoUtc3, LoUtc4);

    Date Dejavu(LoUtc4.GetEpochTime(), Date::Offset::ToLocal);

    TestDates(Dejavu, LoDate);

    {
        Date Mod1(2021, 1, 31);
        Date Mod2 = Mod1.Month(-1);
        Mod1.SetYear(2020);
        Mod1.SetMonth(12);
        TestDates(Mod1, Mod2);
    }
    {
        Date Mod1(2020, 12, 25);
        Date Mod2 = Mod1.Month(1);
        Mod1.SetYear(2021);
        Mod1.SetMonth(1);
        TestDates(Mod1, Mod2);
    }
    {
        Date Mod1(2020, 12, 25);
        Date Mod2 = Mod1.Month(1);
        Mod1 = Mod1.Year(1);
        Mod1 = Mod1.Month(-11);
        TestDates(Mod1, Mod2);
    }
    {
        Date Num(2020, 12, 31, 23, 59);
        Date Str = Num.GetStr();
        TestDates(Num, Str);
    }
    xstring("").ResetColor().Print();
    Rescue();
}

int main()
{
    Begin();
    Nexus<>::Start();

    auto LoDate = RA::Date();

    auto LoBackMin = RA::Date(LoDate.GetEpochTime() - (60)); // back 60 sec or 1 min
    cout << LoDate << endl;
    cout << LoDate.Min(-1) << endl;
    cout << LoBackMin << endl;
    if (LoDate.Min(-1) != LoBackMin)
        ThrowIt("Badd Offset");

    cout << "UTC OFFSET (" << Date::GetComputerOffset() << ")\n";
    cout << "None:      " << Date(Date::Offset::None)    << " = Default = UTC Time (otherwise the input time with no change)" << endl;
    cout << "ToLocal:   " << Date(Date::Offset::ToLocal) << " = From UTC To Local" << endl;
    cout << "ToLocal:   " << Date(Date::Offset::None).ToLocal() << " = From UTC To Local" << endl;
    cout << "ToUTC:     " << Date(Date(Date::Offset::ToLocal), Date::Offset::ToUTC)   << " = From Local back to UTC" << endl;


    Date LoDate1(Date::Offset::None);
    cout << "=======================================================\n";
    cout << "Hour Offset = " << LoDate1.GetHoursOffset() << endl;
    cout << "Secs Offset = " << LoDate1.GetSecondsOffset() << endl;
    cout << "=======================================================\n";
    cout << "                     LOCAL TIME \n";
    cout << "=======================================================\n";
    cout << "----------- From Int ---------------------\n";
    PrintDate(LoDate1);
    cout << "----------- From Str ---------------------\n";
    Date LoDate2(LoDate1.GetStr());
    PrintDate(LoDate2);
    cout << "----------- From EPC ---------------------\n";
    Date LoDate3(LoDate2.GetEpochTime());
    PrintDate(LoDate3);
    cout << "=======================================================\n";
    cout << "                     UTC TIME \n";
    cout << "=======================================================\n";
    cout << "----------- From EPC ---------------------\n";
    Date LoDate4(LoDate3.GetEpochTime(), Date::Offset::ToUTC); // from local
    PrintDate(LoDate4);
    cout << "----------- From Str ---------------------\n";
    Date LoDate5(LoDate4.GetStr(), Date::Offset::None);
    PrintDate(LoDate5);
    cout << "=======================================================\n";
    cout << "                     Convert TIME \n";
    cout << "=======================================================\n";
    Date LoDateToUTC(2021, 7, 9, 0, 1, 2); // literal time
    PrintDate(LoDateToUTC);
    cout << '\n';
    Date LoDateToLocal(LoDateToUTC, Date::Offset::ToUTC);
    PrintDate(LoDateToLocal);
    cout << "=======================================================\n";
    cout << "                     Set Short Time \n";
    cout << "=======================================================\n";
    Date LoShortTimeUTC(2021, 7, 9, Date::Offset::None);
    PrintDate(LoShortTimeUTC);
    cout << '\n';
    Date LoShortTimeLocal(LoShortTimeUTC, Date::Offset::ToLocal);
    PrintDate(LoShortTimeLocal);
    cout << "=======================================================\n";
    Date CurrentDate;
    cout << "Date Seconds:      " << CurrentDate.GetEpochTimeStr() << endl;
    cout << "Date Milliseconds: " << CurrentDate.GetEpochTimeMillisecondsStr() << endl;
    cout << "=======================================================\n";

    Loop();
    Test();

    Nexus<>::Stop();
    return 0;
    Rescue();
}
