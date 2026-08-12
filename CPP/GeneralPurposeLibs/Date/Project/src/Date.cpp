#include "Date.h"
#include <ctime>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <sstream>


// std::tm starts at 1, not 0, so use None
const xvector<RA::EMonth> RA::SvMonths{
    RA::EMonth::None,
    RA::EMonth::Jan,
    RA::EMonth::Feb,
    RA::EMonth::Mar,
    RA::EMonth::Apr,
    RA::EMonth::May,
    RA::EMonth::Jun,
    RA::EMonth::Jul,
    RA::EMonth::Aug,
    RA::EMonth::Sep,
    RA::EMonth::Oct,
    RA::EMonth::Nov,
    RA::EMonth::Dec
};

const xvector<RA::EDay> RA::SvDays{
    RA::EDay::None,
    RA::EDay::Sun,
    RA::EDay::Mon,
    RA::EDay::Tue,
    RA::EDay::Wed,
    RA::EDay::Thr,
    RA::EDay::Fri,
    RA::EDay::Sat
};

const xmap<RA::EMonth, xstring> RA::SmMonthToStr{
    { RA::EMonth::Jan, "Jan" },
    { RA::EMonth::Feb, "Feb" },
    { RA::EMonth::Mar, "Mar" },
    { RA::EMonth::Apr, "Apr" },
    { RA::EMonth::May, "May" },
    { RA::EMonth::Jun, "Jun" },
    { RA::EMonth::Jul, "Jul" },
    { RA::EMonth::Aug, "Aug" },
    { RA::EMonth::Sep, "Sep" },
    { RA::EMonth::Oct, "Oct" },
    { RA::EMonth::Nov, "Nov" },
    { RA::EMonth::Dec, "Dec" },
};

const xmap<RA::EMonth, xstring> RA::SmLongMonthToStr{
    { RA::EMonth::Jan,   "January" },
    { RA::EMonth::Feb,  "February" },
    { RA::EMonth::Mar,     "March" },
    { RA::EMonth::Apr,     "April" },
    { RA::EMonth::May,       "May" },
    { RA::EMonth::Jun,      "June" },
    { RA::EMonth::Jul,      "July" },
    { RA::EMonth::Aug,    "August" },
    { RA::EMonth::Sep, "September" },
    { RA::EMonth::Oct,   "October" },
    { RA::EMonth::Nov,  "November" },
    { RA::EMonth::Dec,  "December" },
};

const xmap<RA::EDay, xstring> RA::SmDayToStr{
    { RA::EDay::Sun, "Sun" },
    { RA::EDay::Mon, "Mon" },
    { RA::EDay::Tue, "Tue" },
    { RA::EDay::Wed, "Wed" },
    { RA::EDay::Thr, "Thr" },
    { RA::EDay::Fri, "Fri" },
    { RA::EDay::Sat, "Sat" },
};

const xmap<RA::EDay, xstring> RA::SmLongDayToStr{
    { RA::EDay::Sun,    "Sunday" },
    { RA::EDay::Mon,    "Monday" },
    { RA::EDay::Tue,   "Tuesday" },
    { RA::EDay::Wed, "Wednesday" },
    { RA::EDay::Thr,  "Thursday" },
    { RA::EDay::Fri,    "Friday" },
    { RA::EDay::Sat,  "Saturday" },
};

EXI std::ostream& RA::operator<<(std::ostream& out, const RA::Date::Format& obj)
{
    out << 
        "Year:    " << obj.GetYear() << '\n' << 
        "Month:   " << obj.GetMonth() << '\n' << 
        "Day:     " << obj.GetDay() << '\n' << 
        "Hour:    " << obj.GetHour() << '\n' << 
        "Min:     " << obj.GetMin() << '\n' << 
        "Sec:     " << obj.GetSec() << '\n' <<
        "Weekday: " << RA::SmDayToStr.Key(RA::SvDays.At(obj.GetWeekDay())) << '\n' <<
        "Yearday: " << obj.GetYearDay() << '\n' <<
        "DST Set: " << obj.GetDaylightSavingsTimeFlag() << '\n';
    return out;
}

void RA::Date::Clear()
{
    MsStrPtr = nullptr;
    MoFormatPtr = nullptr;
}

RA::Date& RA::Date::UpdateAll()
{
    if (MsStrPtr)
    {
        MsStrPtr = nullptr;
        RetStr();
    }
    else if (SbAutoCreateString)
        RetStr();

    if(MoFormatPtr)
    {
        MoFormatPtr = nullptr;
        RetFormat();
    }
    else if (SbAutoFormat)
        RetFormat();

    return The;
}

RA::Date& RA::Date::UpdateStr()
{
    if (MsStrPtr)
    {
        MsStrPtr = nullptr;
        RetStr();
    }
    else if (SbAutoCreateString)
        RetStr();

    return The;
}

RA::Date& RA::Date::CreateStr()
{
    RetStr();
    return The;
}

RA::Date& RA::Date::CreateAll()
{
    RetStr();
    RetFormat();
    return The;
}

void RA::Date::AutoCreateAll()
{
    MeStrFormat = SeDefaultStrFormat;
    if (SbAutoCreateString)
        RetStr();
    if (SbAutoFormat)
        RetFormat();
}

RA::Date::~Date()
{
    Clear();
}

RA::Date::Date(ETimeZone FeEZone)
{
    MeZone = FeEZone;
    if (!SbEZoneCalculated) 
        RA::Date::SetComputerEZone();
    MnEpochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    MnEpochTime += GetSecondsOffset();
    AutoCreateAll();
}

RA::Date::Date(ETimeZone FeEZone, xint FnEpochTime)
{
    MeZone = FeEZone;
    if (FnEpochTime > 9999999999)
        FnEpochTime /= 1000;

    if (!FnEpochTime)
    {
        const long long int LnOffset = GetSecondsOffset();
        MnEpochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) + (LnOffset);
    }
    else
        MnEpochTime = std::time_t(FnEpochTime);
    MoFormatPtr = nullptr;
    AutoCreateAll();
}

RA::Date::Date(ETimeZone FeEZone, ChronoMilliseconds FnChronoTime)
{
    MeZone = FeEZone;
    const long long LnMsEZone = GetSecondsOffset() * 1000;
    auto tp = std::chrono::system_clock::time_point(FnChronoTime);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
    MnEpochTime = ms + LnMsEZone;
    MoFormatPtr = nullptr;
    AutoCreateAll();
}

RA::Date::Date(ETimeZone FeEZone, ChronoSeconds FnChronoTime)
{
    MeZone = FeEZone;
    const long long LnMsEZone = GetSecondsOffset() * 1000;
    // Convert chrono::seconds ΓåÆ integer milliseconds since epoch
    auto tp = std::chrono::system_clock::time_point(FnChronoTime);
    auto ms = std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
    MnEpochTime = ms + LnMsEZone;
    MoFormatPtr = nullptr;
    AutoCreateAll();
}


// Format: "2021-06-23 20:00:00"
RA::Date::Date(ETimeZone FeEZone, const xstring& FsDateTime)
{
    MeZone = FeEZone;
    xvector<int> FvnDigits = (FsDateTime.Match(SnMatchDateTimeCaptureFull))
        ? FsDateTime.Search(SnDateTimeCaptureFull).ForEach<int>([](const xstring& Item) { return Item.ToInt(); })
        : FsDateTime.Search(SnDateTimeCaptureHalf).ForEach<int>([](const xstring& Item) { return Item.ToInt(); }) + xvector<int>{0, 0, 0};
    SetDateTime(FvnDigits[0], FvnDigits[1], FvnDigits[2], FvnDigits[3], FvnDigits[4], FvnDigits[5]);
    AutoCreateAll();
}

RA::Date::Date(ETimeZone FeZone, const Format& FoFormat)
{
    MeZone = FeZone;
    const std::tm* LoTM = &FoFormat;
    MnEpochTime = _mkgmtime(const_cast<std::tm*>(LoTM));
    AutoCreateAll();
}

RA::Date::Date(ETimeZone FeZone, int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond)
{
    MeZone = FeZone;
    SetDateTime(FnYear, FnMonth, FnDay, FnHour, FnMin, FnSecond);
    AutoCreateAll();
}

RA::Date::Date(ETimeZone FeZone, int FnYear, int FnMonth, int FnDay)
{
    MeZone = FeZone;
    SetDateTime(FnYear, FnMonth, FnDay, 0, 0, 0);
    AutoCreateAll();
}

RA::Date::Date(ETimeZone FeZone, const Date& Other)
{
    MeZone = FeZone;
    MnEpochTime = Other.GetEpochTime();
    if (MeZone != Other.MeZone)
        MnEpochTime += Other.GetFlippedOffsetSeconds();
    if (Other.BxFormatted())
        RetFormat();
    if (Other.BxHasString())
        RetStr();
    AutoCreateAll();
}

RA::Date::Date(ETimeZone FeZone, Date&& Other) noexcept
{
    The = std::move(Other);
    if (MeZone != FeZone)
    {
        MeZone = FeZone;
        MnEpochTime += Other.GetFlippedOffsetSeconds();
        if(Other.BxFormatted())
            RetFormat();
        if (Other.BxHasString())
            RetStr();
    }
    AutoCreateAll();
}

RA::Date::Date(const Date& Other)
{
    The = Other;
}

RA::Date::Date(Date&& Other) noexcept
{
    The = std::move(Other);
}

void RA::Date::operator=(const xint FnTime)
{
    The = RA::Date(MeZone, FnTime);
}

void RA::Date::operator=(const RA::Date::EpochTime FnTime)
{
    The = RA::Date(MeZone, FnTime);
}

void RA::Date::operator=(const Date& Other)
{
    MeZone = Other.MeZone;
    if (MeZone == Other.MeZone)
    {
        if (Other.MsStrPtr)
            MsStrPtr.Clone(Other.MsStrPtr);
        else
            MsStrPtr = nullptr;;
        if (Other.MoFormatPtr)
            MoFormatPtr.Clone(Other.MoFormatPtr);
        else
            MoFormatPtr = nullptr;

        MnEpochTime = Other.MnEpochTime;
        AutoCreateAll();
    }
    else
    {
        MoFormatPtr = nullptr;
        MsStrPtr = nullptr;
        MnEpochTime = Other.MnEpochTime + Other.GetFlippedOffsetSeconds();
        AutoCreateAll();
    }
}

void RA::Date::operator=(Date&& Other) noexcept
{
    if (MeZone == Other.MeZone)
    {
        MeZone = Other.MeZone;
        if (Other.MsStrPtr)
            MsStrPtr = std::move(Other.MsStrPtr);
        else
            MsStrPtr = nullptr;;
        if (Other.MoFormatPtr)
            MoFormatPtr = std::move(Other.MoFormatPtr);
        else
            MoFormatPtr = nullptr;

        MnEpochTime = Other.MnEpochTime;
        AutoCreateAll();
    }
    else
    {
        MoFormatPtr = nullptr;
        MsStrPtr = nullptr;
        MnEpochTime = Other.MnEpochTime + Other.GetFlippedOffsetSeconds();
        AutoCreateAll();
    }
}

const xstring& RA::Date::RetDateStr()
{
    if (MsStrPtr)
        return *MsStrPtr;
    char buffer[32];
    RA::Date::Format Format = RetFormat();
    if (MeZone == RA::ETimeZone::LOC)
    {
        if (MeStrFormat == RA::ETimeFormat::DateNZ)
            strftime(buffer, sizeof(buffer), "%Y-%m-%d", std::gmtime(&MnEpochTime));
        else
            strftime(buffer, sizeof(buffer), "%Y-%m-%d (LOC)", std::gmtime(&MnEpochTime));
    }
    else
    {
        if (MeStrFormat == RA::ETimeFormat::DateNZ)
            strftime(buffer, sizeof(buffer), "%Y-%m-%d", std::gmtime(&MnEpochTime));
        else
            strftime(buffer, sizeof(buffer), "%Y-%m-%d (UTC)", std::gmtime(&MnEpochTime));
    }
    return MsStrPtr.New(buffer);
}

const xstring& RA::Date::RetDateTimeStr()
{
    if (MsStrPtr)
        return *MsStrPtr;
    char buffer[64];
    RA::Date::Format Format = RetFormat();
    if (MeZone == RA::ETimeZone::LOC)
    {
        if (MeStrFormat == RA::ETimeFormat::DateTimeNZ)
            strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::gmtime(&MnEpochTime));
        else
            strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S (LOC)", std::gmtime(&MnEpochTime));
    }
    else
    {
        if (MeStrFormat == RA::ETimeFormat::DateTimeNZ)
            strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::gmtime(&MnEpochTime));
        else
            strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S (UTC)", std::gmtime(&MnEpochTime));
    }
    return MsStrPtr.New(buffer);
}
void RA::Date::SetAutoFormat(bool SbEnable){ SbAutoFormat = SbEnable; }

void RA::Date::SetAutoCreateStrings(bool SbEnable){ SbAutoCreateString = SbEnable; }

void RA::Date::SetDefaultStrFormat(RA::ETimeFormat FeFormat){ SeDefaultStrFormat = FeFormat; }

void RA::Date::SetUseDST(bool SbEnable){ SbUseDST = SbEnable; }

const xstring& RA::Date::RetWeekdayTimeStr()
{
    if (MsStrPtr)
        return *MsStrPtr;
    char buffer[128];
    auto LoFormat = RetFormat();

    constexpr const char* Months[] = {
        "none", "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    };

    constexpr const char* Weekdays[] = {
        "none","Sun","Mon","Tue","Wed","Thr","Fri","Sat"
    };

    const char* LsPattern = (MeStrFormat == RA::ETimeFormat::WeekdayNZ)
        ? "(%04d)-(%d|%s)-(%d|%s) (%02d:%02d:%02d)"
        : (MeZone == RA::ETimeZone::LOC)
            ? "(%04d)-(%d|%s)-(%d|%s) (%02d:%02d:%02d) (LOC)"
            : "(%04d)-(%d|%s)-(%d|%s) (%02d:%02d:%02d) (UTC)";

    snprintf(buffer,
        sizeof(buffer),
        LsPattern,

        LoFormat.tm_year,

        LoFormat.tm_mon,
        Months[LoFormat.tm_mon],

        LoFormat.tm_mday,
        Weekdays[LoFormat.tm_wday],

        LoFormat.tm_hour,
        LoFormat.tm_min,
        LoFormat.tm_sec);

    return MsStrPtr.New(buffer);
}

const xstring& RA::Date::RetEpochTimeStr()
{
    if (MsStrPtr)
        return *MsStrPtr;
    char buffer[32];
    RA::Date::Format Format = RetFormat();
    if (MeStrFormat == RA::ETimeFormat::EpochNZ)
        strftime(buffer, sizeof(buffer), "%s", std::gmtime(&MnEpochTime));
    else if (MeZone == RA::ETimeZone::LOC)
        strftime(buffer, sizeof(buffer), "%s (LOC)", std::gmtime(&MnEpochTime));
    else
        strftime(buffer, sizeof(buffer), "%s (UTC)", std::gmtime(&MnEpochTime));
    return MsStrPtr.New(buffer);
}

const xstring& RA::Date::RetNumericTimeStr()
{
    if (MsStrPtr)
        return *MsStrPtr;
    char buffer[64];
    RA::Date::Format Format = RetFormat();
    if (MeStrFormat == RA::ETimeFormat::NumericNZ)
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", std::gmtime(&MnEpochTime));
    else if (MeZone == RA::ETimeZone::LOC)
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S (LOC)", std::gmtime(&MnEpochTime));
    else
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S (UTC)", std::gmtime(&MnEpochTime));
    return MsStrPtr.New(buffer);
}

const xstring& RA::Date::RetEpochTimeEvenMillisecondsStr()
{
    if (MsStrPtr)
        return *MsStrPtr;
    char buffer[32];
    long long secs = static_cast<long long>(MnEpochTime);
    if (MeStrFormat == RA::ETimeFormat::EpochEvenMillisecondsNZ)
        std::snprintf(buffer, sizeof(buffer), "%lld0000", secs);
    else if (MeZone == RA::ETimeZone::LOC)
        std::snprintf(buffer, sizeof(buffer), "%lld0000 (LOC)", secs);
    else
        std::snprintf(buffer, sizeof(buffer), "%lld0000 (UTC)", secs);
    return MsStrPtr.New(buffer);
}

const xstring& RA::Date::RetStr(RA::ETimeFormat FeFormat)
{
    if(MsStrPtr)
        return *MsStrPtr;
    if (FeFormat != RA::ETimeFormat::None)
        MeStrFormat = FeFormat;
    switch (MeStrFormat)
    {
    case RA::ETimeFormat::Date:
        return RetDateStr();
    case RA::ETimeFormat::DateNZ:
        return RetDateStr();
    case RA::ETimeFormat::DateTime:
        return RetDateTimeStr();
    case RA::ETimeFormat::DateTimeNZ:
        return RetDateTimeStr();
    case RA::ETimeFormat::Weekday:
        return RetWeekdayTimeStr();
    case RA::ETimeFormat::WeekdayNZ:
        return RetWeekdayTimeStr();
    case RA::ETimeFormat::Epoch:
        return RetEpochTimeStr();
    case RA::ETimeFormat::EpochNZ:
        return RetEpochTimeStr();
    case RA::ETimeFormat::Numeric:
        return RetNumericTimeStr();
    case RA::ETimeFormat::NumericNZ:
        return RetNumericTimeStr();
    case RA::ETimeFormat::EpochEvenMilliseconds:
        return RetEpochTimeEvenMillisecondsStr();
    case RA::ETimeFormat::EpochEvenMillisecondsNZ:
        return RetEpochTimeEvenMillisecondsStr();
    default:
        ThrowIt("Invalid String Format");
    }
}

xstring RA::Date::RetStr(RA::ETimeFormat FeFormat) const
{
    auto LoCopy = The;
    return LoCopy.RetStr(FeFormat);
}

const xstring& RA::Date::NewStr(RA::ETimeFormat FeFormat)
{
    MsStrPtr = nullptr;
    return RetStr(FeFormat);
}

xstring RA::Date::NewStr(RA::ETimeFormat FeFormat) const
{
    auto LoCopy = The;
    return LoCopy.NewStr(FeFormat);
}

RA::Date::Format& RA::Date::RetFormat()
{
    if (MoFormatPtr)
        return *MoFormatPtr;

    SET(MoFormat, *static_cast<RA::Date::Format*>(std::gmtime(&MnEpochTime)));
    MoFormat.UpdateAfterMakeTime();
    return *MoFormatPtr;
}

int RA::Date::GetDaysInMonth(const int FnYear, const int FnMonth)
{
    switch (FnMonth)
    {
    case 1:
        return 31;
    case 2:
        if (RA::Date::BxLeapYear(FnYear))
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
    RA::Date::Format LoFormat = RetFormat();
    return RA::Date::GetDaysInMonth(LoFormat.tm_year, LoFormat.tm_mon);
}

void RA::Date::ClampDayToMonth()
{
    RA::Date::Format LoTime = RetFormat();
    int LoMaxDays = RA::Date::GetDaysInMonth(LoTime.tm_year, LoTime.tm_mon);
    if (LoTime.tm_mday > LoMaxDays)
        SetEpochTime(MnEpochTime + ((static_cast<RA::Date::EpochTime>(LoMaxDays) - LoTime.tm_mday) * 60 * 60 * 24));
}

bool RA::Date::BxLeapYear(int FnYear)
{
    if (((FnYear % 4 == 0) && (FnYear % 100 != 0)) || (FnYear % 400 == 0))
        return true;
    return false;
}

bool RA::Date::BxLeapYear()
{
    return RA::Date::BxLeapYear(RetFormat().tm_year);
}

bool RA::Date::BxEvenHour()
{
    const auto& Format = The.RetFormat();
    if (Format.tm_sec == 0 && Format.tm_min == 0)
        return true;
    return false;
}

bool RA::Date::BxEvenHour(const int FnRound)
{
    const auto& Format = The.RetFormat();
    if (Format.tm_sec == 0 && Format.tm_min == 0 && ((Format.tm_hour + 1) % FnRound) == 0)
        return true;
    return false;
}

RA::Date::EpochTime RA::Date::GetEpochTime() const {
    return MnEpochTime;
}

RA::Date::ChronoMilliseconds RA::Date::GetChronoMilliseconds() const
{
    return RA::Date::ChronoMilliseconds{
        std::chrono::milliseconds{ MnEpochTime }
    };
}

RA::Date::ChronoSeconds RA::Date::GetChronoSeconds() const
{
    return RA::Date::ChronoSeconds{
        std::chrono::seconds{ MnEpochTime }
    };
}

xint RA::Date::GetEpochTimeInt() const
{
    return static_cast<xint>(MnEpochTime);
}

int RA::Date::GetEpochTimeInt32() const
{
    return static_cast<int>(MnEpochTime);
}


void RA::Date::SetComputerEZone()
{
    if (SbEZoneCalculated)
        return;
    SbEZoneCalculated = true;

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
    SnLocalHoursOffset = (LocalOffset - UTCOffset);
    if (SnLocalHoursOffset > 12)
        SnLocalHoursOffset = (SnLocalHoursOffset - 24);

    SnLocalSecondsOffset = SnLocalHoursOffset * ESecondsIn::Hours;
}

int RA::Date::GetLocalOffsetFromUTC()
{
    return SnLocalHoursOffset;
}

int RA::Date::ConvertOffsetToHours(ETimeZone FeEZone)
{
    switch (FeEZone)
    {
    case(ETimeZone::UTC): return -SnLocalHoursOffset;
    case(ETimeZone::LOC): return  SnLocalHoursOffset;
    default: return 0;
    }
}

int RA::Date::GetRelativeOffsetSeconds(const ETimeZone FeFrom, const ETimeZone FeTo)
{
    if (FeFrom == FeTo)
        return 0;
    if (FeFrom == ETimeZone::UTC && FeTo == ETimeZone::LOC)
        return SnLocalSecondsOffset;
    if (FeFrom == ETimeZone::LOC && FeTo == ETimeZone::UTC)
        return -SnLocalSecondsOffset;
    ThrowIt("Bad Code");
}


int RA::Date::GetRelativeOffsetHours(const ETimeZone FeFrom, const ETimeZone FeTo)
{
    if (FeFrom == FeTo)
        return 0;
    if (FeFrom == ETimeZone::UTC && FeTo == ETimeZone::LOC)
        return SnLocalHoursOffset;
    if (FeFrom == ETimeZone::LOC && FeTo == ETimeZone::UTC)
        return -SnLocalHoursOffset;
    ThrowIt("Bad Code");
}

int RA::Date::GetSecondsOffset() const
{
    switch (MeZone)
    {
    case(ETimeZone::UTC): return 0;
    case(ETimeZone::LOC): return SnLocalSecondsOffset;
    default: ThrowIt("No Zone Set");
    }
}

int RA::Date::GetHoursOffset() const
{
    switch (MeZone)
    {
    case(ETimeZone::UTC): return 0;
    case(ETimeZone::LOC): return SnLocalHoursOffset;
    default: ThrowIt("No Zone Set");
    }
}

int RA::Date::GetFlippedOffsetHours() const
{
    if (MeZone == ETimeZone::UTC)
        return GetRelativeOffsetHours(ETimeZone::UTC, ETimeZone::LOC);
    if (MeZone == ETimeZone::LOC)
        return GetRelativeOffsetHours(ETimeZone::LOC, ETimeZone::UTC);
    ThrowIt("No Zone Set");
}

int RA::Date::GetFlippedOffsetSeconds() const
{
    if (MeZone == ETimeZone::UTC)
        return GetRelativeOffsetSeconds(ETimeZone::UTC, ETimeZone::LOC);
    if (MeZone == ETimeZone::LOC)
        return GetRelativeOffsetSeconds(ETimeZone::LOC, ETimeZone::UTC);
    ThrowIt("No Zone Set");
}

RA::Date RA::Date::ToLocal() const {
    if (MeZone == ETimeZone::LOC)
        return The;
    return Date(ETimeZone::LOC, MnEpochTime + GetFlippedOffsetSeconds());
}

RA::Date RA::Date::ToUTC() const {
    if (MeZone == ETimeZone::UTC)
        return The;
    return Date(ETimeZone::UTC, MnEpochTime + GetFlippedOffsetSeconds());
}

void RA::Date::SetEpochTime(const RA::Date::EpochTime FnEpochTime)
{
    Clear();
    MnEpochTime = FnEpochTime;
}

void RA::Date::SetDateTime(int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond)
{
    Clear();
    int LoMaxDays = RA::Date::GetDaysInMonth(FnYear, FnMonth);
    auto& LoFormat = MoFormatPtr.New();
    LoFormat.tm_year = FnYear;
    LoFormat.tm_mon = FnMonth;
    LoFormat.tm_mday = FnDay;
    LoFormat.tm_hour = FnHour;
    LoFormat.tm_min = FnMin;
    LoFormat.tm_sec = FnSecond;
    LoFormat.tm_isdst = Cast<int>(SbUseDST);

    if (LoFormat.tm_mday > LoMaxDays)
        LoFormat.tm_mday = LoMaxDays;

    LoFormat.UpdateBeforeMakeTime();
    MnEpochTime = _mkgmtime(SCast<std::tm*>(&LoFormat));
    LoFormat.UpdateAfterMakeTime();
}

void RA::Date::SetDateTime()
{
    MsStrPtr = nullptr;
    GET(MoFormat);
    MoFormat.UpdateBeforeMakeTime();
    std::tm* LoTimePtr = &MoFormat;
    // never localtime to prevent double time conversion
    MnEpochTime = _mkgmtime(SCast<std::tm*>(LoTimePtr));
    MoFormat.UpdateAfterMakeTime();
    UpdateStr();
}

bool RA::Date::operator==(const Date& Other) const {
    if (MeZone == Other.MeZone)
        return MnEpochTime == Other.MnEpochTime;
    return (MnEpochTime) == (Other.MnEpochTime + Other.GetFlippedOffsetSeconds());
}
bool RA::Date::operator==(const xint FnTime) const {
    return The.GetEpochTime() == FnTime;
}
bool RA::Date::operator==(const Date::EpochTime FnTime) const {
    return The.GetEpochTime() == FnTime;
}

bool RA::Date::operator!=(const Date& Other) const {
    if (MeZone == Other.MeZone)
        return MnEpochTime != Other.MnEpochTime;
    return (MnEpochTime) != (Other.MnEpochTime + Other.GetFlippedOffsetSeconds());
}
bool RA::Date::operator>=(const Date& Other) const {
    if (MeZone == Other.MeZone)
        return MnEpochTime >= Other.MnEpochTime;
    return (MnEpochTime) >= (Other.MnEpochTime + Other.GetFlippedOffsetSeconds());
}
bool RA::Date::operator<=(const Date& Other) const {
    if (MeZone == Other.MeZone)
        return MnEpochTime <= Other.MnEpochTime;
    return (MnEpochTime) <= (Other.MnEpochTime + Other.GetFlippedOffsetSeconds());
}
bool RA::Date::operator>(const Date& Other) const {
    if (MeZone == Other.MeZone)
        return MnEpochTime > Other.MnEpochTime;
    return (MnEpochTime) > (Other.MnEpochTime + Other.GetFlippedOffsetSeconds());
}
bool RA::Date::operator<(const Date& Other) const {
    if (MeZone == Other.MeZone)
        return MnEpochTime < Other.MnEpochTime;
    return (MnEpochTime) < (Other.MnEpochTime + Other.GetFlippedOffsetSeconds());
}

RA::Date RA::Date::operator+(const xint FnSeconds) const {
    return Date(MeZone, Cast<xint>(MnEpochTime) + FnSeconds);
}
RA::Date RA::Date::operator-(const xint FnSeconds) const {
    return Date(MeZone, Cast<xint>(MnEpochTime) - FnSeconds);
}
RA::Date RA::Date::operator*(const xint FnFactor) const {
    return Date(MeZone, Cast<xint>(MnEpochTime) * FnFactor);
}
RA::Date RA::Date::operator/(const xint FnDivisor) const {
    return Date(MeZone, Cast<xint>(MnEpochTime) / FnDivisor);
}
xint RA::Date::operator-(const Date& Other) const {
    return Cast<xint>(MnEpochTime) - Cast<xint>(Other.MnEpochTime);
}

RA::Date& RA::Date::operator+=(const xint FnSeconds) {
    The = The + FnSeconds;
    return The;
}
RA::Date& RA::Date::operator-=(const xint FnSeconds) {
    The = The - FnSeconds;
    return The;
}
RA::Date& RA::Date::operator*=(const xint FnFactor) {
    The = The * FnFactor;
    return The;
}
RA::Date& RA::Date::operator/=(const xint FnDivisor) {
    The = The / FnDivisor;
    return The;
}
// ------------------------------------------------------
RA::Date& RA::Date::SetYear(int FnYear)
{
    if (FnYear < 1900)
        ThrowIt("Year must be >= 1900");
    RetFormat().tm_year = FnYear;
    SetDateTime();
    return The;
}

RA::Date& RA::Date::SetMonth(int FnMonth)
{
    RA::Clamp(FnMonth, 1, 12);
    RetFormat().tm_mon = FnMonth;
    SetDateTime();
    return The;
}

RA::Date& RA::Date::SetDay(int FnDay)
{
    RA::Clamp(FnDay, 1, GetDaysInMonth());
    RetFormat().tm_mday = FnDay;
    SetDateTime(); // will throw exception if day is out of range
    return The;
}

RA::Date& RA::Date::SetHour(int FnHour)
{
    RA::Clamp(FnHour, 0, 23);
    RetFormat().tm_hour = FnHour;
    SetDateTime();
    return The;
}

RA::Date& RA::Date::SetMin(int FnMin)
{
    MsStrPtr = nullptr;
    RA::Clamp(FnMin, 0, 59);
    RetFormat().tm_min = FnMin;
    SetDateTime();
    return The;
}

RA::Date& RA::Date::SetSecond(int FnSecond)
{
    MsStrPtr = nullptr;
    RA::Clamp(FnSecond, 0, 59);
    RetFormat().tm_sec = FnSecond;
    SetDateTime();
    return The;
}
// ------------------------------------------------------

RA::Date& RA::Date::AdjYear(int FnYear)
{
    RetFormat().tm_year += FnYear;
    SetDateTime();
    UpdateStr();
    return The;
}

RA::Date& RA::Date::AdjMonth(int FnMonth)
{
    RetFormat().tm_mon += FnMonth;
    SetDateTime();
    UpdateStr();
    return The;
}

RA::Date& RA::Date::AdjDay(int FnDay)
{
    MnEpochTime += (ESecondsIn::Days * FnDay);
    UpdateAll();
    return The;
}

RA::Date& RA::Date::AdjHour(int FnHour)
{
    MnEpochTime += (ESecondsIn::Hours * FnHour);
    UpdateAll();
    return The;
}

RA::Date& RA::Date::AdjMin(int FnMin)
{
    MnEpochTime += (ESecondsIn::Minutes * FnMin);
    UpdateAll();
    return The;
}

RA::Date& RA::Date::AdjSecond(int FnSecond)
{
    MnEpochTime += FnSecond;
    UpdateAll();
    return The;
}

RA::Date& RA::Date::AdjBusinessDay(const int FnDay)
{
    if (FnDay == 0)
        return The;

    // RetFormat().tm_wday is 1-based (1=Sunday ... 7=Saturday), matching the
    // "everything starts at 1" wall-calendar convention used by RA::Date::Format.
    int LnIsoWeekDay = (RetFormat().tm_wday + 5) % 7; // ISO: 0=Mon..6=Sun
    int LnCalendarDays = 0;
    int LnRemainingDays = FnDay;

    if (FnDay > 0)
    {
        // If starting on a weekend, the snap to the next Monday IS the 1st
        // business day of movement, so it consumes one unit of FnDay.
        if (LnIsoWeekDay == 5) // Sat -> Mon
        {
            LnCalendarDays += 2;
            LnIsoWeekDay = 0;
            --LnRemainingDays;
        }
        else if (LnIsoWeekDay == 6) // Sun -> Mon
        {
            LnCalendarDays += 1;
            LnIsoWeekDay = 0;
            --LnRemainingDays;
        }

        if (LnRemainingDays > 0)
        {
            const int LnWeeks = LnRemainingDays / 5;
            int       LnRemainder = LnRemainingDays % 5;

            int LnNewIsoWeekDay = LnIsoWeekDay + LnRemainder;
            if (LnNewIsoWeekDay >= 5) // crosses into/through the weekend
                LnRemainder += 2;

            LnCalendarDays += (LnWeeks * 7) + LnRemainder;
        }
    }
    else
    {
        // If starting on a weekend, the snap to the previous Friday IS the 1st
        // business day of movement, so it consumes one unit of FnDay.
        if (LnIsoWeekDay == 5) // Sat -> Fri
        {
            LnCalendarDays -= 1;
            LnIsoWeekDay = 4;
            ++LnRemainingDays;
        }
        else if (LnIsoWeekDay == 6) // Sun -> Fri
        {
            LnCalendarDays -= 2;
            LnIsoWeekDay = 4;
            ++LnRemainingDays;
        }

        if (LnRemainingDays < 0)
        {
            const int LnAbsDay = -LnRemainingDays;
            const int LnWeeks = LnAbsDay / 5;
            int       LnRemainder = LnAbsDay % 5;

            int LnNewIsoWeekDay = LnIsoWeekDay - LnRemainder;
            if (LnNewIsoWeekDay < 0) // crosses into/through the weekend
                LnRemainder += 2;

            LnCalendarDays -= (LnWeeks * 7) + LnRemainder;
        }
    }

    MnEpochTime += (ESecondsIn::Days * LnCalendarDays);
    UpdateAll();
    return The;
}

RA::Date RA::Date::ToYear(const int FnYear) const { return The.Dup().AdjYear(FnYear); }

RA::Date RA::Date::ToMonth(const int FnMonth) const { return The.Dup().AdjMonth(FnMonth); }

RA::Date RA::Date::ToDay(const int FnDay) const { return The.Dup().AdjDay(FnDay); }

RA::Date RA::Date::ToHour(const int FnHour) const { return The.Dup().AdjHour(FnHour); }

RA::Date RA::Date::ToMin(const int FnMin) const { return The.Dup().AdjMin(FnMin); }

RA::Date RA::Date::ToSecond(const int FnSecond) const { return The.Dup().AdjSecond(FnSecond); }

RA::Date RA::Date::ToBusinessDay(const int FnDay) const { return The.Dup().AdjBusinessDay(FnDay); }

RA::Date& RA::Date::SetToNoon()
{
    auto& LoFormat = RetFormat();
    LoFormat.tm_sec = 0;
    LoFormat.tm_min = 0;
    LoFormat.tm_hour = 12;
    SetDateTime();
    return The;
}

RA::Date& RA::Date::ZeroOutHourMinSec()
{
    auto& LoFormat = RetFormat();
    LoFormat.tm_sec = 0;
    LoFormat.tm_min = 0;
    LoFormat.tm_hour = 0;
    SetDateTime();
    return The;
}

bool operator==(const RA::Date& Left, const xint Right) { return Left.GetEpochTime() == Right; }
bool operator!=(const RA::Date& Left, const xint Right) { return Left.GetEpochTime() != Right; }
bool operator>=(const RA::Date& Left, const xint Right) { return Left.GetEpochTime() >= Right; }
bool operator<=(const RA::Date& Left, const xint Right) { return Left.GetEpochTime() <= Right; }
bool operator> (const RA::Date& Left, const xint Right) { return Left.GetEpochTime() > Right; }
bool operator< (const RA::Date& Left, const xint Right) { return Left.GetEpochTime() < Right; }


bool operator==(const xint& Left, const RA::Date& Right) { return Left == Right.GetEpochTime(); }
bool operator!=(const xint& Left, const RA::Date& Right) { return Left != Right.GetEpochTime(); }
bool operator>=(const xint& Left, const RA::Date& Right) { return Left >= Right.GetEpochTime(); }
bool operator<=(const xint& Left, const RA::Date& Right) { return Left <= Right.GetEpochTime(); }
bool operator> (const xint& Left, const RA::Date& Right) { return Left > Right.GetEpochTime(); }
bool operator< (const xint& Left, const RA::Date& Right) { return Left < Right.GetEpochTime(); }

std::ostream& RA::operator<<(std::ostream& out, const RA::Date& obj)
{
    out << obj.GetStr();
    return out;
}

std::ostream& RA::operator<<(std::ostream& out, const xp<RA::Date> ptr)
{
    out << ptr->GetStr();
    return out;
}

void RA::Date::Format::UpdateBeforeMakeTime()
{
    // note: std::tm has tm_mday start at 1, not 0
    tm_year -= 1900;
    --tm_mon;
    tm_isdst = Cast<int>(SbUseDST); // safety
}

void RA::Date::Format::UpdateAfterMakeTime()
{
    // note: std::tm has tm_mday start at 1, not 0
    tm_year += 1900;
    ++tm_mon;
    ++tm_wday; // ignored by _mkgmtime
    ++tm_yday; // ignored by _mkgmtime
}
