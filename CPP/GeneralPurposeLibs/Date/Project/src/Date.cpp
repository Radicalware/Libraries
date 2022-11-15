#include "Date.h"
#include <ctime>
#include <stdlib.h>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <sstream>

RA::Date::Date(Offset FeOffset)
{
    MbDateCurrent = true;
    if (!SbOffsetCalculated) RA::Date::GetComputerOffset();
    if (RA::Date::Offset::ToUTC == FeOffset)
        ThrowIt("UTC - UTC is Invalid and should never be done");
    const long long int LnSecondsOffset = GetSecondsOffset(FeOffset);
    MoEpochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    if (LnSecondsOffset)
    {
        MoEpochTime += LnSecondsOffset * 60;
    }
}

RA::Date::Date(const Date& Other, Offset FeOffset)
{
    if (!SbOffsetCalculated) RA::Date::GetComputerOffset();
    const long long int LnSecondsOffset = GetSecondsOffset(FeOffset);
    MoEpochTime = Other.GetEpochTime();
    MbDateCurrent = Other.MbDateCurrent;
    if (LnSecondsOffset)
    {
        MoEpochTime += LnSecondsOffset * 60;
    }
    MoTime.Year = 0;
}


RA::Date::Date(uint FnEpochTime, Offset FeOffset)
{
    if (FnEpochTime > 9999999999)
        FnEpochTime = FnEpochTime / 1000;
    if (!SbOffsetCalculated) RA::Date::GetComputerOffset();
    const long long int LnSecondsOffset = GetSecondsOffset(FeOffset);

    if (!FnEpochTime)
    {
        MbDateCurrent = true;
        MoEpochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) + (LnSecondsOffset * 60);
    }
    else
        MoEpochTime = std::time_t(FnEpochTime);
    MoTime.Year = 0;
}

// Format: "2021-06-23 20:00:00"
RA::Date::Date(const xstring& FsDateTime, Offset FeOffset)
{
    if (!SbOffsetCalculated) RA::Date::GetComputerOffset();
    int LnHoursOffset = GetSecondsOffset(FeOffset);
    if (LnHoursOffset)
        LnHoursOffset /= 60;

    xvector<int> FvnDigits = FsDateTime.Search(R"(^(\d\d\d\d)-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)$)").ForEach<int>([](const xstring& Item) { return Item.ToInt(); });
    MoTime.Year = 0;
    SetDateTime(FvnDigits[0], FvnDigits[1], FvnDigits[2], FvnDigits[3] + LnHoursOffset, FvnDigits[4], FvnDigits[5]);
}


RA::Date::Date(int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond, Offset FeOffset)
{
    if (!SbOffsetCalculated) RA::Date::GetComputerOffset();
    int LnHoursOffset = GetSecondsOffset(FeOffset);
    if (LnHoursOffset)
        LnHoursOffset /= 60;

    MoTime.Year = 0;
    SetDateTime(FnYear, FnMonth, FnDay, FnHour + LnHoursOffset, FnMin, FnSecond);
}

RA::Date::Date(int FnYear, int FnMonth, int FnDay, Offset FeOffset)
{
    if (!SbOffsetCalculated) RA::Date::GetComputerOffset();
    int LnHoursOffset = GetSecondsOffset(FeOffset);
    if (LnHoursOffset)
        LnHoursOffset /= 60;

    MoTime.Year = 0;
    SetDateTime(FnYear, FnMonth, FnDay, LnHoursOffset, 0, 0);
}


RA::Date::Date(const Date& Other)
{
    *this = Other;
}

RA::Date::Date(Date&& Other) noexcept
{
    *this = std::move(Other);
}

void RA::Date::operator=(const uint FnTime)
{
    This = RA::Date(FnTime);
}

void RA::Date::operator=(const Date& Other)
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
    
    MoEpochTime    = Other.MoEpochTime;
    MoTime         = Other.MoTime;
    MbDateCurrent  = Other.MbDateCurrent;
}

void RA::Date::operator=(Date&& Other) noexcept
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
    MbDateCurrent  = Other.MbDateCurrent;
}

void RA::Date::Clear()
{
    if (MsStr)
    {
        delete MsStr;
        MsStr = nullptr;
    }

    MoTime.Year = 0;
}

RA::Date::~Date()
{
    Clear();
}

void RA::Date::CreateStr()
{
    Clear();

    char buffer[80];
    RA::Date::Layout Layout = GetLayout();
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::gmtime(&MoEpochTime));
    MsStr = new xstring(buffer);
}

const xstring& RA::Date::GetStr() 
{
    if (!MsStr)
        CreateStr();

    return *MsStr;
}

xstring RA::Date::GetStr() const
{
    Date Copy = *this;
    return Copy.GetStr();
}

RA::Date::Layout& RA::Date::GetLayout()
{
    if (MoTime.Year > 0)
        return MoTime;

    MoTime = *reinterpret_cast<RA::Date::Layout*>(std::gmtime(&MoEpochTime));
    MoTime.Year += 1900;
    MoTime.Month++;
    return MoTime;
}

int RA::Date::GetDaysInMonth(const int FnYear, const int FnMonth)
{
    switch (FnMonth)
    {
    case 1:
        return 31;
    case 2:
        if (RA::Date::IsLeapYear(FnYear))
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

int RA::Date::GetDaysInMonth()
{
    RA::Date::Layout LoLayout = GetLayout();
    return RA::Date::GetDaysInMonth(LoLayout.Year, LoLayout.Month);
}

void RA::Date::ClampDayToMonth()
{
    RA::Date::Layout LoTime = GetLayout();
    int LoMaxDays = RA::Date::GetDaysInMonth(LoTime.Year, LoTime.Month);
    if (LoTime.Day > LoMaxDays)
        SetEpochTime(MoEpochTime + ((static_cast<RA::Date::EpochTime>(LoMaxDays) - LoTime.Day) * 60 * 60 * 24));
}

bool RA::Date::IsLeapYear(int FnYear)
{
    if (((FnYear % 4 == 0) && (FnYear % 100 != 0)) || (FnYear % 400 == 0))
        return true;
    return false;
}

bool RA::Date::IsLeapYear()
{
    return RA::Date::IsLeapYear(GetLayout().Year);
}

bool RA::Date::IsEvenHour()
{
    const auto& Layout = This.GetLayout();
    if (Layout.Sec == 0 && Layout.Min == 0)
        return true;
    return false;
}

bool RA::Date::IsEvenHour(const int FnRound)
{
    const auto& Layout = This.GetLayout();
    if (Layout.Sec == 0 && Layout.Min == 0 && ((Layout.Hour + 1) % FnRound) == 0)
        return true;
    return false;
}
    
RA::Date::EpochTime RA::Date::GetEpochTime() const {
    return MoEpochTime;
}

int RA::Date::GetEpochTimeInt() const
{
    return static_cast<int>(MoEpochTime);
}

pint RA::Date::GetEpochTimePint() const
{
    return static_cast<pint>(MoEpochTime);
}

pint RA::Date::GetEpochTimeMilliseconds() const
{
    if (!MbDateCurrent)
        return GetEpochTimePint() * 1000;

    auto Rounded = GetEpochTimePint();
    size_t Milliseconds =  static_cast<pint>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()) % 1000;
    return Rounded * 1000 + Milliseconds;
}

xstring RA::Date::GetEpochTimeStr() const {
    return RA::ToXString(MoEpochTime);
}

xstring RA::Date::GetNumericTimeStr()
{
    std::stringstream LoSS;
    auto AddStreamValue = [&LoSS](int FnInt) ->void
    {
        if (FnInt <= 9)
            LoSS << '0';
        LoSS << FnInt;
    };
    RA::Date::Layout LoTime = GetLayout();
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

xstring RA::Date::GetEpochTimeMillisecondsStr() const
{
    return RA::ToXString(This.GetEpochTimeMilliseconds());
}

int RA::Date::GetComputerOffset()
{
    if (SbOffsetCalculated)
        return SnOffsetUTC;
    SbOffsetCalculated = true;

    long LocalOffset = 0;
    long UTCOffset = 0;
    {
        std::time_t current_time;
        std::time(&current_time);
        struct std::tm* timeinfo = std::localtime(&current_time);
        LocalOffset = timeinfo->tm_hour;
    }
    {
        std::time_t current_time;
        std::time(&current_time);
        struct std::tm* timeinfo = std::gmtime(&current_time);
        UTCOffset = timeinfo->tm_hour;
    }
    SnOffsetUTC = (LocalOffset - UTCOffset);
    return SnOffsetUTC;
}

int RA::Date::GetSecondsOffset() {
    return SnOffsetUTC * 60;
}

int RA::Date::GetHoursOffset() {
    return SnOffsetUTC;
}

RA::Date RA::Date::ToLocal() const {
    return This.Hour(SnOffsetUTC);
}

RA::Date RA::Date::FromLocal() const {
    return This.Hour(-SnOffsetUTC);
}

RA::Date RA::Date::ToUTC() const {
    return This.Hour(-SnOffsetUTC);
}

RA::Date RA::Date::FromUTC() const {
    return This.Hour(+SnOffsetUTC);
}

int RA::Date::GetSecondsOffset(Offset FeOffset)
{
    // None    = (Input is UTC left as UTC)
    // ToUTC   = (Input is Local but converted to UTC)
    // ToLocal = (Input UTC but converted to Local)
    if (FeOffset == Offset::None)
        return 0;

    int LnHoursOffset = 0;
    switch (FeOffset)
    {
        //case(Offset::UTC):      LnHoursOffset =  SnOffsetUTC; break;
        //case(Offset::Local):    LnHoursOffset =  0; break;
        case(Offset::ToUTC):    LnHoursOffset = -SnOffsetUTC; break;
        case(Offset::ToLocal):  LnHoursOffset = SnOffsetUTC; break;
        default:                LnHoursOffset = 0;
    }
    return LnHoursOffset * 60;
}

void RA::Date::SetEpochTime(const RA::Date::EpochTime FnEpochTime)
{
    MoEpochTime = FnEpochTime;
    MoTime.Year = 0;
}

void RA::Date::SetDateTime(int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond)
{
    Clear();

    int LoMaxDays = RA::Date::GetDaysInMonth(FnYear, FnMonth);

    auto SetArgTime = [&]()->void {
        MoTime.Year = FnYear - 1900;
        MoTime.Month = FnMonth - 1; // -1 due to indexing
        MoTime.Day = FnDay;
        MoTime.Hour = FnHour;
        MoTime.Min = FnMin;
        MoTime.Sec = FnSecond;
        MoTime.DaylightSavingsTimeFlag = 1; // daylight saving (on/off)

        if (MoTime.Day > LoMaxDays)
            MoTime.Day = LoMaxDays;
    };

    SetArgTime();
    MoEpochTime = _mkgmtime(reinterpret_cast<std::tm*>(&MoTime));

    MoTime.Year += 1900;
    MoTime.Month++;
}

void RA::Date::SetDateTime(const RA::Date::Layout& FnTime)
{
    SetDateTime(FnTime.Year, FnTime.Month, FnTime.Day, FnTime.Hour, FnTime.Min, FnTime.Sec);
}

bool RA::Date::operator==(const Date& Other) const {
    return MoEpochTime == Other.MoEpochTime;
}
bool RA::Date::operator!=(const Date& Other) const {
    return MoEpochTime != Other.MoEpochTime;
}
bool RA::Date::operator>=(const Date& Other) const {
    return MoEpochTime >= Other.MoEpochTime;
}
bool RA::Date::operator<=(const Date& Other) const {
    return MoEpochTime <= Other.MoEpochTime;
}
bool RA::Date::operator>(const Date& Other) const {
    return MoEpochTime > Other.MoEpochTime;
}
bool RA::Date::operator<(const Date& Other) const {
    return MoEpochTime < Other.MoEpochTime;
}
// ------------------------------------------------------
RA::Date RA::Date::Year(int FnYear) const
{
    Date RoDate = *this;
    RA::Date::Layout LoTime = RoDate.GetLayout();
    LoTime.Year += FnYear;
    RoDate.SetDateTime(LoTime);
    return RoDate;
}

RA::Date RA::Date::Month(int FnMonth) const
{
    Date RoDate = *this;
    RA::Date::Layout LoLayout = RoDate.GetLayout();
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
RA::Date RA::Date::Day(int FnDay) const {
    return Second(FnDay * 60 * 60 * 24);
}

RA::Date RA::Date::Hour(int FnHour) const {
    return Second(FnHour * 60 * 60);
}

RA::Date RA::Date::Min(int FnMin) const {
    return Second(FnMin * 60);
}

RA::Date RA::Date::Second(int FnSecond) const {
    return Date(MoEpochTime + FnSecond);
}
// ------------------------------------------------------

void RA::Date::SetYear(int FnYear)
{
    RA::Date::Layout Layout = GetLayout();
    Layout.Year = FnYear;
    SetDateTime(Layout);
}

void RA::Date::SetMonth(int FnMonth)
{
    if (FnMonth < 1) FnMonth = 1;
    else if (FnMonth > 12) FnMonth = 12;

    RA::Date::Layout Layout = GetLayout();
    Layout.Month = FnMonth;
    SetDateTime(Layout);
}

void RA::Date::SetDay(int FnDay)
{
    int Days = GetDaysInMonth();
    if (FnDay > Days) FnDay = Days;
    else if (FnDay < 1) FnDay = 1;

    RA::Date::Layout Layout = GetLayout();
    Layout.Day = FnDay;
    SetDateTime(Layout);
}

void RA::Date::SetHour(int FnHour)
{
    if (FnHour > 60) FnHour = 60;
    else if (FnHour < 1) FnHour = 1;

    MoTime.Year = 0;
    MoEpochTime += static_cast<RA::Date::EpochTime>(FnHour) * 60 * 60;
}

void RA::Date::SetMin(int FnMin)
{
    if (FnMin > 60) FnMin = 60;
    else if (FnMin < 1) FnMin = 1;

    MoTime.Year = 0;
    MoEpochTime += static_cast<RA::Date::EpochTime>(FnMin) * 60;
}

void RA::Date::SetSecond(int FnSecond)
{
    if (FnSecond > 60) FnSecond = 60;
    else if (FnSecond < 1) FnSecond = 1;

    MoTime.Year = 0;
    MoEpochTime += FnSecond;
}
// ------------------------------------------------------

std::ostream& operator<<(std::ostream& out, const RA::Date& obj)
{
    out << obj.GetStr();
    return out;
}