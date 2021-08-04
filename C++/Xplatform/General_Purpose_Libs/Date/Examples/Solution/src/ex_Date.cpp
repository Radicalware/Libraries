
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#include "Nexus.h"
#include "xvector.h"
#include "xstring.h"
#include "Date.h"

using std::cout;
using std::endl;

void PrintDate(Date& FoDate)
{
    cout << "FnDate                = " << FoDate << endl;
    cout << "FnDate.GetStr         = " << FoDate.GetStr() << endl;
    cout << "FnDate.GetEpochTime   = " << FoDate.GetEpochTime() << endl;
    cout << "FnDate.GetNumericTime = " << FoDate.GetNumericTimeStr() << endl;
}


void TestDates(Date& One, Date& Two)
{
    if (One == Two)
        xstring("success\n").ToGreen().Print();
    else
        xstring("Failed\n").ToRed().Print();

    xstring().ResetColor();
}

void Test()
{
    xstring("\r").ResetColor().Print();
    Date LoDate;
    Date LoLocal1(LoDate.GetEpochTime());
    TestDates(LoLocal1, LoDate);

    Date LoLocal2(LoLocal1.GetStr());
    TestDates(LoLocal1, LoLocal2);

    auto& LoLocalStruct = LoLocal2.GetTime();
    Date LoLocal3(
        LoLocalStruct.tm_year,
        LoLocalStruct.tm_mon,
        LoLocalStruct.tm_mday,
        LoLocalStruct.tm_hour,
        LoLocalStruct.tm_min,
        LoLocalStruct.tm_sec,
        Date::Offset::None);

    TestDates(LoLocal2, LoLocal3);

    Date LoLocal4(LoLocal3.GetEpochTime());
    TestDates(LoLocal3, LoLocal4);
    // ------------------------------------------------------------
    Date LoUtc1(LoLocal4.GetEpochTime(), Date::Offset::ToUTC);

    Date LoUtc2(LoUtc1.GetStr());
    TestDates(LoUtc1, LoUtc2);

    auto& LoUtcStruct = LoUtc2.GetTime();
    Date LoUtc3(
        LoUtcStruct.tm_year,
        LoUtcStruct.tm_mon,
        LoUtcStruct.tm_mday,
        LoUtcStruct.tm_hour,
        LoUtcStruct.tm_min,
        LoUtcStruct.tm_sec,
        Date::Offset::None);

    TestDates(LoUtc2, LoUtc3);

    Date LoUtc4(LoUtc3.GetEpochTime());
    TestDates(LoUtc3, LoUtc4);

    Date Dejavu(LoUtc4.GetEpochTime(), Date::Offset::ToLocal);

    TestDates(Dejavu, LoDate);
    xstring("").ResetColor().Print();
}

int main()
{
    Nexus<>::Start();

    Date LoDate1(Date::Offset::Local);
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
    Date LoDate4(LoDate3.GetEpochTime(), Date::Offset::ToUTC);
    PrintDate(LoDate4);
    cout << "----------- From Str ---------------------\n";
    Date LoDate5(LoDate4.GetStr(), Date::Offset::None);
    PrintDate(LoDate5);
    cout << "=======================================================\n";
    cout << "                     Convert TIME \n";
    cout << "=======================================================\n";
    Date LoDateToUTC(2021, 7, 9, 3, 0, 0, Date::Offset::ToLocal);
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

    Test();

    Nexus<>::Stop();
    return 0;
}
