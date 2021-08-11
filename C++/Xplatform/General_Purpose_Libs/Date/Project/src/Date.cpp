#include "Date.h"
#include <ctime>
#include <stdlib.h>
#include <iomanip>
#include <ctime>
#include <sstream>


bool Date::SbAppliedLocalOffset = false;
int  Date::SnLocalOffset = 0;


Date::Date(Offset FeOffset)
{
    const long long int LnSecondsOffset = GetSecondsOffset(FeOffset);
    MoEpochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) + LnSecondsOffset * 60;
    MoTime.Year = 0;
}

Date::Date(const Date& Other, Offset FeOffset)
{
    const long long int LnSecondsOffset = GetSecondsOffset(FeOffset);
    MoEpochTime = Other.GetEpochTime() + LnSecondsOffset * 60;
    MoTime.Year = 0;
}


Date::Date(uint FnEpochTime, Offset FeOffset)
{
    const long long int LnSecondsOffset = GetSecondsOffset(FeOffset);

    if (!FnEpochTime)
        MoEpochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) + LnSecondsOffset * 60;
    else
        MoEpochTime = std::time_t(FnEpochTime + LnSecondsOffset * 60);
    MoTime.Year = 0;
}

// Format: "2021-06-23 20:00:00"
Date::Date(const xstring& FsDateTime, Offset FeOffset)
{
    int LnHoursOffset = GetSecondsOffset(FeOffset);
    if (LnHoursOffset)
        LnHoursOffset /= 60;

    xvector<int> FvnDigits = FsDateTime.Search(R"(^(\d\d\d\d)-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)$)").ForEach<int>([](const xstring& Item) { return Item.ToInt(); });
    MoTime.Year = 0;
    SetDateTime(FvnDigits[0], FvnDigits[1], FvnDigits[2], FvnDigits[3] + LnHoursOffset, FvnDigits[4], FvnDigits[5]);
}


Date::Date(int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond, Offset FeOffset)
{
    int LnHoursOffset = GetSecondsOffset(FeOffset);
    if (LnHoursOffset)
        LnHoursOffset /= 60;

    MoTime.Year = 0;
    SetDateTime(FnYear, FnMonth, FnDay, FnHour + LnHoursOffset, FnMin, FnSecond);
}

Date::Date(int FnYear, int FnMonth, int FnDay, Offset FeOffset)
{
    int LnHoursOffset = GetSecondsOffset(FeOffset);
    if (LnHoursOffset)
        LnHoursOffset /= 60;

    MoTime.Year = 0;
    SetDateTime(FnYear, FnMonth, FnDay, LnHoursOffset, 0, 0);
}


Date::Date(const Date& Other)
{
    *this = Other;
}

Date::Date(Date&& Other) noexcept
{
    *this = std::move(Other);
}

void Date::operator=(const Date& Other)
{
    if (Other.MsStr)
    {
        if (MsStr)
            *MsStr = *Other.MsStr;
        else
            MsStr = new xstring(*Other.MsStr);
    }
    else if (MsStr)
        Clear();

    MoEpochTime = Other.MoEpochTime;
    MoTime = Other.MoTime;
}

void Date::operator=(Date&& Other) noexcept
{
    if (Other.MsStr)
    {
        if (MsStr)
            *MsStr = std::move(*Other.MsStr);
        else
            MsStr = new xstring(std::move(*Other.MsStr));
    }
    else if (MsStr)
        Clear();

    MoEpochTime    = Other.MoEpochTime;
    MoTime         = Other.MoTime;
}

void Date::Clear()
{
    if (MsStr)
    {
        delete MsStr;
        MsStr = nullptr;
    }

    MoTime.Year = 0;
}

Date::~Date()
{
    Clear();
}

void Date::CreateStr()
{
    Clear();

    // Wed Jun 23 23:30:38 2021
    // MsStr = new xstring(ToXString(std::asctime(std::localtime(&MoEpochTime))));

    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&MoEpochTime));
    MsStr = new xstring(buffer);
}

const xstring& Date::GetStr() 
{
    if (!MsStr)
        CreateStr();

    return *MsStr;
}

Date::Layout Date::GetLayout()
{
    if (MoTime.Year)
        return MoTime;

    MoTime = *reinterpret_cast<Date::Layout*>(std::localtime(&MoEpochTime));
    MoTime.Year += 1900;
    MoTime.Month++;
    return MoTime;
}

int Date::GetDaysInMonth(const int FnYear, const int FnMonth)
{
    switch (FnMonth)
    {
    case 1:
        return 31;
    case 2:
        if (Date::IsLeapYear(FnYear))
            return 29;
        else
            return 28;
    case 3:
        return 31;
    case 4:
        return 30;
    case 5:
        return 31;
    case 6:
        return 30;
    case 7:
        return 31;
    case 8:
        return 31;
    case 9:
        return 30;
    case 10:
        return 31;
    case 11:
        return 30;
    case 12:
        return 31;
    default:
        throw "Invalid Month";
    }
}

int Date::GetDaysInMonth()
{
    Date::Layout LoLayout = GetLayout();
    return Date::GetDaysInMonth(LoLayout.Year, LoLayout.Month);
}

void Date::ClampDayToMonth()
{
    Date::Layout& LoTime = GetLayout();
    int LoMaxDays = Date::GetDaysInMonth(LoTime.Year, LoTime.Month);
    if (LoTime.Day > LoMaxDays)
        SetEpochTime(MoEpochTime + ((static_cast<Date::EpochTime>(LoMaxDays) - LoTime.Day) * 60 * 60 * 24));
}

bool Date::IsLeapYear(int FnYear)
{
    if (((FnYear % 4 == 0) && (FnYear % 100 != 0)) || (FnYear % 400 == 0))
        return true;
    return false;
}

bool Date::IsLeapYear()
{
    return Date::IsLeapYear(GetLayout().Year);
}

Date::EpochTime Date::GetEpochTime() const {
    return MoEpochTime;
}

xstring Date::GetEpochTimeStr() const {
    return ToXString(MoEpochTime);
}

xstring Date::GetNumericTimeStr()
{
    std::stringstream LoSS;
    auto AddStreamValue = [&LoSS](int FnInt) ->void
    {
        if (FnInt <= 9)
            LoSS << '0';
        LoSS << FnInt;
    };
    Date::Layout LoTime = GetLayout();
    LoSS << LoTime.Year;
    AddStreamValue(LoTime.Month);
    AddStreamValue(LoTime.Day);
    LoSS << 'T';
    if (LoTime.Hour == 0 && LoTime.Min == 0 && LoTime.Sec == 0)
    {
        LoSS << "000000";
        return LoSS.str();
    }

    AddStreamValue(LoTime.Hour);
    AddStreamValue(LoTime.Min);
    AddStreamValue(LoTime.Sec);

    return LoSS.str();
}

int Date::GetSecondsOffset()
{
    return SnLocalOffset;
}

int Date::GetHoursOffset()
{
    return SnLocalOffset / 60;
}

int Date::GetSecondsOffset(Offset FeOffset)
{
    if (FeOffset == Offset::None)
        return 0;

    if (!SbAppliedLocalOffset)
    {
        SbAppliedLocalOffset = true;
        time_t LoZeroTime(0);
        tm* LoEpochOffset = std::localtime(&LoZeroTime);
        SnLocalOffset = (23 - LoEpochOffset->tm_hour) * 60;
    }

    int LnTimeOffset = 0;
    switch (FeOffset)
    {
        case(Offset::UTC):      LnTimeOffset =  SnLocalOffset; break;
        case(Offset::Local):    LnTimeOffset =  0; break;
        case(Offset::ToUTC):    LnTimeOffset =  SnLocalOffset; break;
        case(Offset::ToLocal):  LnTimeOffset = -SnLocalOffset; break;
        default:                LnTimeOffset = 0;
    }
    return LnTimeOffset;
}

void Date::SetEpochTime(const Date::EpochTime FnEpochTime)
{
    MoEpochTime = FnEpochTime;
    MoTime.Year = 0;
}

void Date::SetDateTime(int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond)
{
    Clear();

    MoTime.Year    = FnYear - 1900;
    MoTime.Month   = FnMonth - 1; // -1 due to indexing
    MoTime.Day     = FnDay;
    MoTime.Hour    = FnHour;
    MoTime.Min     = FnMin;
    MoTime.Sec     = FnSecond;
    MoTime.DaylightSavingsTimeFlag   = 1; // daylight saving (on/off)

    int LoMaxDays = Date::GetDaysInMonth(FnYear, FnMonth);
    if (MoTime.Day > LoMaxDays)
        MoTime.Day = LoMaxDays;

    MoEpochTime = std::mktime(reinterpret_cast<std::tm*>(&MoTime));

    MoTime.Year += 1900;
    MoTime.Month++;
}

void Date::SetDateTime(const Date::Layout& FnTime)
{
    SetDateTime(FnTime.Year, FnTime.Month, FnTime.Day, FnTime.Hour + 1, FnTime.Min, FnTime.Sec);
}

bool Date::operator==(const Date& Other) const {
    return MoEpochTime == Other.MoEpochTime;
}
bool Date::operator!=(const Date& Other) const {
    return MoEpochTime != Other.MoEpochTime;
}
bool Date::operator>=(const Date& Other) const {
    return MoEpochTime >= Other.MoEpochTime;
}
bool Date::operator<=(const Date& Other) const {
    return MoEpochTime <= Other.MoEpochTime;
}
bool Date::operator>(const Date& Other) const {
    return MoEpochTime > Other.MoEpochTime;
}
bool Date::operator<(const Date& Other) const {
    return MoEpochTime < Other.MoEpochTime;
}
// ------------------------------------------------------
Date Date::Year(int FnYear) const
{
    Date RoDate = *this;
    Date::Layout LoTime = RoDate.GetLayout();
    LoTime.Year += FnYear;
    RoDate.SetDateTime(LoTime);
    return RoDate;
}

Date Date::Month(int FnMonth) const
{
    Date RoDate = *this;
    Date::Layout LoLayout = RoDate.GetLayout();
    int CurentMonth = LoLayout.Month;
    int MoveMonths  = CurentMonth + FnMonth;

    if (MoveMonths >= 1 && MoveMonths <= 12)
    {
        LoLayout.Month += FnMonth;
    }
    else if (MoveMonths < 1)
    {
        MoveMonths     -= LoLayout.Month;
        LoLayout.Month  = 1; // month goes down to the last year
        LoLayout.Year  -= static_cast<int>((MoveMonths * -1) / 12) + 1;
        LoLayout.Month += 12 + MoveMonths;
    }
    else if (MoveMonths > 12)
    {
        MoveMonths     -= (12 - LoLayout.Month);
        LoLayout.Year  += static_cast<int>(MoveMonths / 12);
        LoLayout.Month  = MoveMonths - 12;
    }
    else
        throw "This won't happen";

    RoDate.SetDateTime(LoLayout);
    return RoDate;
}
// ------------------------------------------------------
Date Date::Day(int FnDay) const {
    return Second(FnDay * 60 * 60 * 24);
}

Date Date::Hour(int FnHour) const {
    return Second(FnHour * 60 * 60);
}

Date Date::Min(int FnMin) const {
    return Second(FnMin * 60);
}

Date Date::Second(int FnSecond) const {
    return Date(MoEpochTime + FnSecond);
}
// ------------------------------------------------------

void Date::SetYear(int FnYear)
{
    Date::Layout Layout = GetLayout();
    Layout.Year = FnYear;
    SetDateTime(Layout);
}

void Date::SetMonth(int FnMonth)
{
    if (FnMonth < 1) FnMonth = 1;
    else if (FnMonth > 12) FnMonth = 12;

    Date::Layout Layout = GetLayout();
    Layout.Month = FnMonth;
    SetDateTime(Layout);
}

void Date::SetDay(int FnDay)
{
    int Days = GetDaysInMonth();
    if (FnDay > Days) FnDay = Days;
    else if (FnDay < 1) FnDay = 1;

    Date::Layout Layout = GetLayout();
    Layout.Day = FnDay;
    SetDateTime(Layout);
}

void Date::SetHour(int FnHour)
{
    if (FnHour > 60) FnHour = 60;
    else if (FnHour < 1) FnHour = 1;

    MoTime.Year = 0;
    MoEpochTime += static_cast<Date::EpochTime>(FnHour) * 60 * 60;
}

void Date::SetMin(int FnMin)
{
    if (FnMin > 60) FnMin = 60;
    else if (FnMin < 1) FnMin = 1;

    MoTime.Year = 0;
    MoEpochTime += static_cast<Date::EpochTime>(FnMin) * 60;
}

void Date::SetSecond(int FnSecond)
{
    if (FnSecond > 60) FnSecond = 60;
    else if (FnSecond < 1) FnSecond = 1;

    MoTime.Year = 0;
    MoEpochTime += FnSecond;
}
// ------------------------------------------------------

std::ostream& operator<<(std::ostream& out, Date& obj)
{
    out << obj.GetStr();
    return out;
}