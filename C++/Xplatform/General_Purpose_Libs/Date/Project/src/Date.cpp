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
    MoTime.tm_year = 0;
}

Date::Date(const Date& Other, Offset FeOffset)
{
    const long long int LnSecondsOffset = GetSecondsOffset(FeOffset);
    MoEpochTime = Other.GetEpochTime() + LnSecondsOffset * 60;
    MoTime.tm_year = 0;
}


Date::Date(uint FnEpochTime, Offset FeOffset)
{
    const long long int LnSecondsOffset = GetSecondsOffset(FeOffset);

    if (!FnEpochTime)
        MoEpochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) + LnSecondsOffset * 60;
    else
        MoEpochTime = std::time_t(FnEpochTime + LnSecondsOffset * 60);
    MoTime.tm_year = 0;
}

// Format: "2021-06-23 20:00:00"
Date::Date(const xstring& FsDateTime, Offset FeOffset)
{
    int LnHoursOffset = GetSecondsOffset(FeOffset);
    if (LnHoursOffset)
        LnHoursOffset /= 60;

    xvector<int> FvnDigits = FsDateTime.Search(R"(^(\d\d\d\d)-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)$)").ForEach<int>([](const xstring& Item) { return Item.ToInt(); });
    MoTime.tm_year = 0;
    SetDateTime(FvnDigits[0], FvnDigits[1], FvnDigits[2], FvnDigits[3] + LnHoursOffset, FvnDigits[4], FvnDigits[5]);
}


Date::Date(int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond, Offset FeOffset)
{
    int LnHoursOffset = GetSecondsOffset(FeOffset);
    if (LnHoursOffset)
        LnHoursOffset /= 60;

    MoTime.tm_year = 0;
    SetDateTime(FnYear, FnMonth, FnDay, FnHour + LnHoursOffset, FnMin, FnSecond);
}

Date::Date(int FnYear, int FnMonth, int FnDay, Offset FeOffset)
{
    int LnHoursOffset = GetSecondsOffset(FeOffset);
    if (LnHoursOffset)
        LnHoursOffset /= 60;

    MoTime.tm_year = 0;
    SetDateTime(FnYear, FnMonth, FnDay, LnHoursOffset, 0, 0);
}


Date::Date(const Date& Other)
{
    self = Other;
}

Date::Date(Date&& Other) noexcept
{
    self = std::move(Other);
}

void Date::operator=(const Date& Other)
{
    if (self.MsStr)
        *self.MsStr = *Other.MsStr;

    self.MoEpochTime = Other.MoEpochTime;
    self.MoTime = Other.MoTime;
}

void Date::operator=(Date&& Other) noexcept
{
    if(self.MsStr)
        *self.MsStr     = std::move(*Other.MsStr);

    self.MoEpochTime    = Other.MoEpochTime;
    self.MoTime         = Other.MoTime;
}

Date::~Date()
{
    if (MsStr)
        delete MsStr;
}

void Date::CreateStr()
{
    if (MsStr) 
        delete MsStr;

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

std::tm Date::GetTime()
{
    if (MoTime.tm_year)
        return MoTime;

    MoTime = *std::localtime(&MoEpochTime);
    MoTime.tm_year += 1900;
    MoTime.tm_mon++;
    return MoTime;
}

std::time_t Date::GetEpochTime() const {
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
    std::tm LoTime = GetTime();
    LoSS << LoTime.tm_year;
    AddStreamValue(LoTime.tm_mon);
    AddStreamValue(LoTime.tm_mday);
    LoSS << 'T';
    if (LoTime.tm_hour == 0 && LoTime.tm_min == 0 && LoTime.tm_sec == 0)
    {
        LoSS << "000000";
        return LoSS.str();
    }

    AddStreamValue(LoTime.tm_hour);
    AddStreamValue(LoTime.tm_min);
    AddStreamValue(LoTime.tm_sec);

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

void Date::SetDateTime(int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond)
{
    MoTime.tm_year    = FnYear - 1900;
    MoTime.tm_mon     = FnMonth - 1; // -1 due to indexing
    MoTime.tm_mday    = FnDay;
    MoTime.tm_hour    = FnHour;
    MoTime.tm_min     = FnMin;
    MoTime.tm_sec     = FnSecond;
    MoTime.tm_isdst   = 1; // daylight saving (on/off)

    MoEpochTime = std::mktime(&MoTime);

    MoTime.tm_year += 1900;
    MoTime.tm_mon++;
}

void Date::SetDateTime(const std::tm& FnTime)
{
    SetDateTime(FnTime.tm_year, FnTime.tm_mon, FnTime.tm_mday, FnTime.tm_hour, FnTime.tm_min, FnTime.tm_sec);
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

Date Date::Year(int FnYear) {
    Date RoDate = *this;
    std::tm LoTime = RoDate.GetTime();
    LoTime.tm_year += FnYear;
    RoDate.SetDateTime(LoTime);
    return RoDate;
}

Date Date::Month(int FnMonth) {
    Date RoDate = *this;
    std::tm LoTime = RoDate.GetTime();
    LoTime.tm_mon += FnMonth;
    RoDate.SetDateTime(LoTime);
    return RoDate;
}

Date Date::Day(int FnDay) {
    Date RoDate = *this;
    std::tm LoTime = RoDate.GetTime();
    LoTime.tm_mday += FnDay;
    RoDate.SetDateTime(LoTime);
    return RoDate;
}

Date Date::Hour(int FnHour) {
    Date RoDate = *this;
    std::tm LoTime = RoDate.GetTime();
    LoTime.tm_hour += FnHour;
    RoDate.SetDateTime(LoTime);
    return RoDate;
}

Date Date::Min(int FnMin) {
    Date RoDate = *this;
    std::tm LoTime = RoDate.GetTime();
    LoTime.tm_min += FnMin;
    RoDate.SetDateTime(LoTime);
    return RoDate;
}

Date Date::Second(int FnSecond) {
    Date RoDate = *this;
    std::tm LoTime = RoDate.GetTime();
    LoTime.tm_sec += FnSecond;
    RoDate.SetDateTime(LoTime);
    return RoDate;
}


std::ostream& operator<<(std::ostream& out, Date& obj)
{
    out << obj.GetStr();
    return out;
}