#pragma once

#include<sstream>
#include<math.h>
#include<string>
#include<memory>

#include "xstring.h"
#include "xvector.h"
#include "xmap.h"

// (737514 / 365.25) + 1

// 31 + 29 + 29 = 89 days

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    #ifdef DLL_EXPORT
       #define EXI __declspec(dllexport)
    #else
       #define EXI __declspec(dllimport)
    #endif
#else
    #define EXI
#endif


using uint = size_t;

namespace RA
{
    class EXI Date
    {
    public:
        using SteadyClock = std::chrono::steady_clock;
        using EpochTime = std::time_t;

        enum class Offset
        {
            None,   // Input is Default = UTC
            ToUTC,  // Input is Local but converted to UTC
            ToLocal // Input is UTC but converted to Local
        };

        enum class EMonth
        {
            None,
            January,
            February,
            March,
            April,
            May,
            June,
            July,
            August,
            September,
            October,
            November,
            December
        };

        // Exact Same Setup As "std::tm"
        struct Layout
        {
            int Sec;        // seconds after the minute - [0, 60] including leap second
            int Min;        // minutes after the hour - [0, 59]
            int Hour;       // hours since midnight - [0, 23]
            int Day;        // day of the month - [1, 31]
            int Month;      // months since January - [0, 11]
            int Year;       // years since 1900

            int WeekDay;    // days since Sunday - [0, 6]
            int YearDay;    // days since January 1 - [0, 365]

            int DaylightSavingsTimeFlag; // daylight savings time flag
        };

        void Clear();
        ~Date();
        Date(Offset FeOffset);
        Date(const Date& Other, Offset FeOffset);
        Date(uint FnEpochTime = 0, Offset FeOffset = Offset::None);

        // Format: "2021-06-23 20:00:00"
        Date(const xstring& FsDateTime, Offset FeOffset = Offset::None);
        Date(int FnYear, int FnMonth, int FnDay, int FnHour = 0, int FnMin = 0, int FnSecond = 0, Offset FeOffset = Offset::None);
        Date(int FnYear, int FnMonth, int FnDay, Offset FeOffset);

        Date(const Date& Other);
        Date(Date&& Other) noexcept;

        void operator=(const uint FnTime);
        void operator=(const Date& Other);
        void operator=(Date&& Other) noexcept;

        void            CreateStr();
        const xstring& GetStr();
        xstring   GetStr() const;
        Date::Layout& GetLayout();
        static int      GetDaysInMonth(const int FnYear, const int FnMonth);
        int             GetDaysInMonth();
        void            ClampDayToMonth();
        static bool     IsLeapYear(int FnYear);
        bool            IsLeapYear();
        bool            IsEvenHour();
        bool            IsEvenHour(const int FnRound);

        Date::EpochTime GetEpochTime() const;

        int             GetEpochTimeInt() const;
        pint            GetEpochTimePint() const;
        pint            GetEpochTimeMilliseconds() const;

        xstring         GetEpochTimeStr() const;
        xstring         GetNumericTimeStr();
        xstring         GetEpochTimeMillisecondsStr() const;

        static int      GetComputerOffset();
        static int      GetSecondsOffset();
        static int      GetHoursOffset();

        Date ToUTC() const;
        Date FromUTC() const;
        Date ToLocal() const;
        Date FromLocal() const;

        Date Year(int FnYear) const;
        Date Month(int FnMonth) const;
        Date Day(int FnDay) const;
        Date Hour(int FnHour) const;
        Date Min(int FnMin) const;
        Date Second(int FnSecond) const;

        void SetYear(int FnYear);
        void SetMonth(int FnMonth);
        void SetDay(int FnDay);
        void SetHour(int FnHour);
        void SetMin(int FnMin);
        void SetSecond(int FnSecond);

    private:
        istatic int  SnOffsetUTC = 0;
        istatic bool SbOffsetCalculated = false;

        static int  GetSecondsOffset(Offset FeOffset);

        void SetEpochTime(const Date::EpochTime FnEpochTime);
        void SetDateTime(int FnYear, int FnMonth, int FnDay, int FnHour = 0, int FnMin = 0, int FnSecond = 0);
        void SetDateTime(const Date::Layout& FnTime);

        Date::EpochTime MoEpochTime = 0;
        xstring* MsStr = nullptr;
        Date::Layout    MoTime;
        bool MbDateCurrent = false;

    public:
        bool operator==(const Date& Other) const;
        bool operator!=(const Date& Other) const;
        bool operator>=(const Date& Other) const;
        bool operator<=(const Date& Other) const;
        bool operator> (const Date& Other) const;
        bool operator< (const Date& Other) const;
    };
};

EXI std::ostream& operator<<(std::ostream& out, const RA::Date& obj);