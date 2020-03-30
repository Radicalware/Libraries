#include "Date.h"


const xvector<int> Date::StandardMonthDays
{
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
};

const xvector<int> Date::LeapMonthDays
{
        31, 29, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
};


void Date::update_total_days()
{
    switch (mYear)
    {
    case 0:
        mTotal_days = 0;
    case 1:
        mTotal_days = 365;
    default:
        mTotal_days = static_cast<int>((static_cast<double>(mYear) - 1) * 365.2425 + 1);
    }

    mTotal_days += mDay;

    auto month_day_vec = self.month_days();
    for (xvector<int>::const_iterator it = month_day_vec.begin(); it != month_day_vec.begin() + mMonth - 1; it++)
        mTotal_days += *it;
    mTotal_days++;
}

Date::Date()
{
}

Date::Date(size_t days)
{
    mTotal_days = days;
}

Date::Date(const xstring& str)
{
    xvector<int> ints = str.findall(R"((\d+))").render<int>([](const xstring& rend_str) { return rend_str.to_int(); });
    if (ints.size() < 3) 
    {
        xstring err("Three ints not found for date: ");
        err + str;
        throw std::runtime_error(err.c_str());
    }
    mMonth   = ints[0];
    mDay     = ints[1];
    mYear    = ints[2];

    self.update_total_days();
}

Date::Date(int month, int day, int year)
{
    mDay    = day;
    mMonth  = month;
    mYear   = year;

    self.update_total_days();
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
    self.mNeat = date.mNeat;
    self.mTotal_days = date.mTotal_days;
    self.mUpdated_ints = false;
    self.mUpdated_str  = false;
}

void Date::operator=(Date&& date) noexcept
{
    self = date;
}

Date::~Date()
{
    // used unique ptrs
}

Date& Date::update_ints()
{
    mUpdated_ints = true;

    mYear = static_cast<int>(static_cast<double>(mTotal_days) / 365.2425) + 1;

    int this_year_days = static_cast<int>(std::remainder(mTotal_days, 365.2425));

    xvector<int> month_day_vec = self.month_days();
    int month = 0;
    xvector<int>::const_iterator month_itr = month_day_vec.begin();
    for (;month_itr != month_day_vec.end() && this_year_days > 1; month_itr++)
    {
        month++;
        this_year_days -= *month_itr;
    };
    mMonth = month;

    mDay = this_year_days + *(month_itr - 1) - 1;

    return self;
}

Date& Date::update_str()
{
    // insert 0 locations 0M/0D/0YYY
    // 1.) idx = 0
    // 2.) idx = 3
    // 3.) idx = 6
    // pass if size = 10

    if (mUpdated_ints == false)
        self.update_ints();

    mUpdated_str = true;

    std::stringstream ostr;
    ostr << mMonth << '/' << mDay << '/' << mYear;
    mStr.clear();
    mStr = ostr.str();

    if (mNeat == false || mStr.size() == 10)
        return self;

    if (mStr[1] == '/')
        mStr.insert(mStr.begin(), '0');
    if (mStr.size() == 10)
        return self;
    mStr.insert(mStr.begin() + 3, '0');

    while (mStr.size() != 10)
        mStr.insert(mStr.begin() + 6, '0');

    return self;
}

bool Date::is_leap_year() const
{
    return ((mYear % 4) == 0);
}

const xvector<int>& Date::month_days() const
{
    if (self.is_leap_year())
        return Date::LeapMonthDays;
    else
        return Date::StandardMonthDays;
}

xstring Date::str()
{
    if (!mUpdated_str)
        self.update_str();

    return mStr;
}

size_t Date::total_days() const{
    return mTotal_days;
}

Date& Date::set_neat(bool neat)
{
    mNeat = neat;
    return self;
}

std::ostream& Date::operator<<(std::ostream& out)
{
    out << self.str();
    return out;
}

void Date::operator+=(int val){
    mTotal_days += val;
}

void Date::operator-=(int val){
    mTotal_days -= val;
}

Date Date::operator+(int val) const
{
    Date date = self;
    date += val;
    return date;
}

Date Date::operator-(int val) const
{
    Date date = self;
    date -= val;
    return date;
}

std::ostream& operator<<(std::ostream& out, Date& obj)
{
    out << obj.str() << " : " << obj.total_days();
    return out;
}
