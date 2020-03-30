#pragma once
#ifndef self
#define self (*this)
#endif

#include<sstream>
#include<math.h>
#include<string>
#include<memory>

#include "xstring.h"
#include "xvector.h"

// (737514 / 365.25) + 1

// 31 + 29 + 29 = 89 days


class Date
{

    bool mUpdated_str  = false;
    bool mUpdated_ints = false;

    xstring mStr;
    size_t mTotal_days = 0;

    int mDay = 0;
    int mMonth = 0;
    int mYear = 0;

    bool mNeat = false;

    void update_total_days();

public:
    const static xvector<int> StandardMonthDays;
    const static xvector<int> LeapMonthDays;

    Date();
    Date(size_t days);
    Date(const xstring& str);
    Date(int month, int day, int year);
    Date(const Date& date);
    Date(Date&& date) noexcept;
    void operator=(const Date& date);
    void operator=(Date&& date) noexcept;
    ~Date();

    Date& update_ints();
    Date& update_str();

    bool is_leap_year() const;
    const xvector<int>& month_days() const;

    xstring str();
    size_t total_days() const;
    Date& set_neat(bool neat);

    std::ostream& operator<<(std::ostream& out);

    void operator+=(int val);
    void operator-=(int val);

    Date operator+(int val) const;
    Date operator-(int val) const;
};

std::ostream& operator<<(std::ostream& out, Date& obj);