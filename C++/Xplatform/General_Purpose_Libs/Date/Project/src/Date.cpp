#include "Date.h"
#include <ctime>
#include <stdlib.h>


Date::Date(uint FnEpochTime)
{
    if(!FnEpochTime)
        MoTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    else
        MoTime = std::time_t(FnEpochTime);
}


// Format: "2021-06-23 20:00:00"
Date::Date(const xstring& FsDateTime)
{
    xvector<int> FvnDigits = FsDateTime.Search(R"(^(\d\d\d\d)-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)$)").ForEach<int>([](const xstring& Item) { return Item.ToInt(); });
    SetDateTime(FvnDigits[0], FvnDigits[1], FvnDigits[2], FvnDigits[3], FvnDigits[4], FvnDigits[5]);
}


Date::Date(int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond)
{
    SetDateTime(FnYear, FnMonth, FnDay, FnHour, FnMin, FnSecond);
}   

Date::Date(const Date& date)
{
    self = date;
}

Date::Date(Date&& date) noexcept
{
    self = date;
}

void Date::operator=(const Date& date)
{
    self.MsStr   = date.MsStr;
    self.MoTime  = date.MoTime;
}

void Date::operator=(Date&& date) noexcept
{
    self = date;
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
    // MsStr = new xstring(ToXString(std::asctime(std::localtime(&MoTime))));

    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&MoTime));
    MsStr = new xstring(buffer);
}

xstring& Date::GetStr()
{
    if (!MsStr)
        CreateStr();

    return *MsStr;
}

std::tm& Date::GetTM()
{
    std::tm& LoTime = *std::localtime(&MoTime);
    //MnYears     = LoTime.tm_year;
    //MnMonths    = LoTime.tm_mon;
    //MnDays      = LoTime.tm_mday;
    //MnHours     = LoTime.tm_hour;
    //MnSeconds   = LoTime.tm_sec;

    return LoTime;
}

void Date::SetDateTime(int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond)
{
    std::tm LoTm{};

    LoTm.tm_year = FnYear - 1900;
    LoTm.tm_mon = FnMonth - 1; // -1 due ot indexing
    LoTm.tm_mday = FnDay;
    LoTm.tm_hour = FnHour - 1;
    LoTm.tm_min = FnMin;
    LoTm.tm_sec = FnSecond;
    LoTm.tm_isdst = 0; // Not daylight saving

    MoTime = std::mktime(&LoTm);
}

std::ostream& operator<<(std::ostream& out, Date& obj)
{
    out << obj.GetStr();
    return out;
}