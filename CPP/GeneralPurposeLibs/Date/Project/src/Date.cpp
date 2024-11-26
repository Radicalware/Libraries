#include "Date.h"
#include <ctime>
#include <stdlib.h>
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


RA::Date::Date(xint FnEpochTime, Offset FeOffset)
{
    if (FnEpochTime > 9999999999)
        FnEpochTime /= 1000;

    if (!SbOffsetCalculated)
        RA::Date::GetComputerOffset();

    if (!FnEpochTime)
    {
        const long long int LnSecondsOffset = GetSecondsOffset(FeOffset);

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

    xvector<int> FvnDigits = (FsDateTime.Match(SnMatchDateTimeCaptureFull))
        ? FsDateTime.Search(SnDateTimeCaptureFull).ForEach<int>([](const xstring& Item) { return Item.ToInt(); })
        : FsDateTime.Search(SnDateTimeCaptureHalf).ForEach<int>([](const xstring& Item) { return Item.ToInt(); }) + xvector<int>{0, 0, 0};
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

void RA::Date::operator=(const xint FnTime)
{
    The = RA::Date(FnTime);
}

void RA::Date::operator=(const Date& Other)
{
    if (Other.MsStrPtr)
    {
        if (MsStrPtr)
            *MsStrPtr = *Other.MsStrPtr;
        else
            MsStrPtr = new xstring(*Other.MsStrPtr);
    }
    else if (MsStrPtr)
        Clear();
    
    MoEpochTime    = Other.MoEpochTime;
    MoTime         = Other.MoTime;
    MbDateCurrent  = Other.MbDateCurrent;
}

void RA::Date::operator=(Date&& Other) noexcept
{
    if (Other.MsStrPtr)
    {
        if (MsStrPtr)
            *MsStrPtr = std::move(*Other.MsStrPtr);
        else
            MsStrPtr = new xstring(std::move(*Other.MsStrPtr));
    }
    else if (MsStrPtr)
        Clear();

    MoEpochTime    = Other.MoEpochTime;
    MoTime         = Other.MoTime;
    MbDateCurrent  = Other.MbDateCurrent;
}

void RA::Date::Clear()
{
    if (MsStrPtr)
    {
        delete MsStrPtr;
        MsStrPtr = nullptr;
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
    MsStrPtr = new xstring(buffer);
}

const xstring& RA::Date::GetStr() 
{
    if (!MsStrPtr)
        CreateStr();

    return *MsStrPtr;
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
    const auto& Layout = The.GetLayout();
    if (Layout.Sec == 0 && Layout.Min == 0)
        return true;
    return false;
}

bool RA::Date::IsEvenHour(const int FnRound)
{
    const auto& Layout = The.GetLayout();
    if (Layout.Sec == 0 && Layout.Min == 0 && ((Layout.Hour + 1) % FnRound) == 0)
        return true;
    return false;
}
    
RA::Date::EpochTime RA::Date::GetEpochTime() const {
    return MoEpochTime;
}

xint RA::Date::GetEpochTimeInt() const
{
    return static_cast<xint>(MoEpochTime);
}

int RA::Date::GetEpochTimeInt32() const
{
    return static_cast<int>(MoEpochTime);
}

xint RA::Date::GetEpochTimeMilliseconds() const
{
    if (!MbDateCurrent)
        return GetEpochTimeInt() * 1000;

    auto Rounded = GetEpochTimeInt();
    size_t Milliseconds =  static_cast<xint>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()) % 1000;
    return Rounded * 1000 + Milliseconds;
}

xint RA::Date::GetEpochTimeEvenMilliseconds() const
{
    return GetEpochTimeInt() * 1000;
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
    return RA::ToXString(The.GetEpochTimeMilliseconds());
}

xstring RA::Date::GetEpochTimeEvenMillisecondsStr() const
{
    return RA::ToXString(The.GetEpochTimeEvenMilliseconds());
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
    return The.Hour(SnOffsetUTC);
}

RA::Date RA::Date::FromLocal() const {
    return The.Hour(-SnOffsetUTC);
}

RA::Date RA::Date::ToUTC() const {
    return The.Hour(-SnOffsetUTC);
}

RA::Date RA::Date::FromUTC() const {
    return The.Hour(+SnOffsetUTC);
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
        throw "The won't happen";

    RoDate.SetDateTime(LoLayout);
    return RoDate;
}
// ------------------------------------------------------
RA::Date RA::Date::Day(int FnDay) const {
    return Second(FnDay * SecondsTo::Days);
}

RA::Date RA::Date::Hour(int FnHour) const {
    return Second(FnHour * SecondsTo::Hours);
}

RA::Date RA::Date::Min(int FnMin) const {
    return Second(FnMin * SecondsTo::Minutes);
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
    HostDelete(MsStrPtr);

    if (FnMonth < 1) FnMonth = 1;
    else if (FnMonth > 12) FnMonth = 12;

    RA::Date::Layout Layout = GetLayout();
    Layout.Month = FnMonth;
    SetDateTime(Layout);
}

void RA::Date::SetDay(int FnDay)
{
    HostDelete(MsStrPtr);
    if (FnDay < 1) FnDay = 1;
    RA::Date::Layout Layout = GetLayout();
    Layout.Day = FnDay;
    SetDateTime(Layout); // will throw exception if day is out of range
}

void RA::Date::SetHour(int FnHour)
{
    HostDelete(MsStrPtr);

    if (FnHour < 1) FnHour = 1;
    else if (FnHour > 24) FnHour = 24;

    RA::Date::Layout Layout = GetLayout();
    Layout.Month = FnHour;
    SetDateTime(Layout);
}

void RA::Date::SetMin(int FnMin)
{
    HostDelete(MsStrPtr);

    if (FnMin < 1) FnMin = 1;
    else if (FnMin > 60) FnMin = 60;

    RA::Date::Layout Layout = GetLayout();
    Layout.Min = FnMin;
    SetDateTime(Layout);
}

void RA::Date::SetSecond(int FnSecond)
{
    HostDelete(MsStrPtr);

    if (FnSecond < 1) FnSecond = 1;
    else if (FnSecond > 60) FnSecond = 60;

    RA::Date::Layout Layout = GetLayout();
    Layout.Sec = FnSecond;
    SetDateTime(Layout);
}
// ------------------------------------------------------

void RA::Date::ModYear(int FnYear)
{
    RA::Date::Layout Layout = GetLayout();
    Layout.Year += FnYear;
    SetDateTime(Layout);
}

void RA::Date::ModMonth(int FnMonth)
{
    RA::Date::Layout Layout = GetLayout();
    if (FnMonth > 0)
    {

        if (FnMonth >= 12)
        {
            Layout.Year  += (FnMonth / 12);
            FnMonth = FnMonth % 12;
        }
        if (Layout.Month + FnMonth <= 12)
            Layout.Month += FnMonth;
        else
        {
            FnMonth -= (12 - Layout.Month);
            Layout.Month = 0;
            ++Layout.Year;
            Layout.Month += FnMonth;
        }
    }
    else
    {
        nvar LnAbsModMonth = std::abs(FnMonth);
        if (LnAbsModMonth >= 12)
        {
            Layout.Year   -= (LnAbsModMonth / 12);
            LnAbsModMonth -= (LnAbsModMonth % 12);
        }
        if (Layout.Month > LnAbsModMonth)
            Layout.Month -= LnAbsModMonth;
        else
        {
            LnAbsModMonth -= Layout.Month;
            --Layout.Year;
            Layout.Month = (12 - LnAbsModMonth);
        }
    }
    SetDateTime(Layout);
}

void RA::Date::ModDay(int FnDay)
{
    MoTime.Year = 0;
    HostDelete(MsStrPtr);

    MoEpochTime += (SecondsTo::Days * FnDay);
}

void RA::Date::ModHour(int FnHour)
{
    MoTime.Year = 0;
    HostDelete(MsStrPtr);

    MoEpochTime += (SecondsTo::Hours * FnHour);
}

void RA::Date::ModMin(int FnMin)
{
    MoTime.Year = 0;
    HostDelete(MsStrPtr);

    MoEpochTime += (SecondsTo::Minutes * FnMin);
}

void RA::Date::ModSecond(int FnSecond)
{
    MoTime.Year = 0;
    HostDelete(MsStrPtr);

    MoEpochTime += FnSecond;
}

bool operator==(const RA::Date& Left, const xint Right) { return Left.GetEpochTime() == Right; }
bool operator!=(const RA::Date& Left, const xint Right) { return Left.GetEpochTime() != Right; }
bool operator>=(const RA::Date& Left, const xint Right) { return Left.GetEpochTime() >= Right; }
bool operator<=(const RA::Date& Left, const xint Right) { return Left.GetEpochTime() <= Right; }
bool operator> (const RA::Date& Left, const xint Right) { return Left.GetEpochTime() >  Right; }
bool operator< (const RA::Date& Left, const xint Right) { return Left.GetEpochTime() <  Right; }


bool operator==(const xint& Left, const RA::Date& Right) { return Left == Right.GetEpochTime(); }
bool operator!=(const xint& Left, const RA::Date& Right) { return Left != Right.GetEpochTime(); }
bool operator>=(const xint& Left, const RA::Date& Right) { return Left >= Right.GetEpochTime(); }
bool operator<=(const xint& Left, const RA::Date& Right) { return Left <= Right.GetEpochTime(); }
bool operator> (const xint& Left, const RA::Date& Right) { return Left >  Right.GetEpochTime(); }
bool operator< (const xint& Left, const RA::Date& Right) { return Left <  Right.GetEpochTime(); }

std::ostream& operator<<(std::ostream& out, const RA::Date& obj)
{
    out << obj.GetStr();
    return out;
}
