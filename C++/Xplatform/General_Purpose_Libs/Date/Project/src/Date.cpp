#include "Date.h"
#include <ctime>


Date::Date(uint FnEpochTime)
{
    MoTime =  std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
}


Date::Date(int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin)
{
    std::tm LoTm{};

    LoTm.tm_year = FnYear;
    LoTm.tm_mon  = FnMonth;
    LoTm.tm_mday = FnDay;
    LoTm.tm_hour = FnHour;
    LoTm.tm_min  = FnMin;
    LoTm.tm_isdst = 0; // Not daylight saving

    MoTime = std::mktime(&LoTm);
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
    MsStr = new xstring(ToXString(std::asctime(std::localtime(&MoTime))));
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

std::ostream& operator<<(std::ostream& out, Date& obj)
{
    out << obj.GetStr();
    return out;
}