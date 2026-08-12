#pragma once

#include<sstream>
#include<math.h>
#include<string>
#include<memory>
#include <chrono>
#include "re2/re2.h"

#include "xstring.h"
#include "xvector.h"
#include "xmap.h"
#include "Macros.h"

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


using xint = size_t;

namespace RA
{
    enum class ETimeZone
    {
        UTC,  // UTC Time Zone
        LOC   // Local Time Zone
    };
    static const xmap<ETimeZone, xstring> SmZoneToStr{
        { ETimeZone::UTC,  "UTC"  },
        { ETimeZone::LOC,  "LOC"  }
    };

    enum class ETimeFormat
    {
        None,
        Date,     // "2021-06-23"
        DateTime, // "2021-06-23 20:00:00"
        Weekday,  // "(2026)-(7|Jul)-(24|Fri) (13:04:16)"
        Epoch,    // "1624473600"
        Numeric,  // "20210623200000" >> 2021 06 23 20 00 00
        EpochEvenMilliseconds, // "1624473600000" rounded to nearest second

        DateNZ,     // "2021-06-23"
        DateTimeNZ, // "2021-06-23 20:00:00"
        WeekdayNZ,  // "(2026)-(7|Jul)-(24|Fri) (13:04:16)"
        EpochNZ,    // "1624473600"
        NumericNZ,  // "20210623200000" >> 2021 06 23 20 00 00
        EpochEvenMillisecondsNZ // "1624473600000" rounded to nearest second
    };


    enum class EMonth { None, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec };
    enum class EDay { None, Sun, Mon, Tue, Wed, Thr, Fri, Sat };

    extern const xvector<EMonth> SvMonths;
    extern const xvector<EDay>   SvDays;

    extern const xmap<EMonth, xstring> SmMonthToStr;
    extern const xmap<EMonth, xstring> SmLongMonthToStr;
    extern const xmap<EDay, xstring>   SmDayToStr;
    extern const xmap<EDay, xstring>   SmLongDayToStr;

    struct ESecondsIn
    {
        istatic CST xint Minutes = 60;
        istatic CST xint Hours = Minutes * 60;
        istatic CST xint Days = Hours * 24;
    };

    class EXI Date
    {
    public:
        using SteadyClock = std::chrono::steady_clock;
        using EpochTime = std::time_t;
        using ChronoMilliseconds = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>;
        using ChronoSeconds = std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;

        // Exact Same Setup As "std::tm" but easier to read vars
        class Format : private std::tm
        {
            friend class RA::Date;
            // NOTES: Format(2026, 1, 25) = Year 2026, January, 25th
            //       std::tm{2026, 1, 25} = Year 126, February, 26th
            void UpdateBeforeMakeTime();
            void UpdateAfterMakeTime();
        public:
            auto GetSec()     const { return tm_sec;  }
            auto GetMin()     const { return tm_min;  }
            auto GetHour()    const { return tm_hour; }
            auto GetDay()     const { return tm_mday; }
            auto GetMonth()   const { return tm_mon;  }
            auto GetYear()    const { return tm_year; }
            auto GetWeekDay() const { return tm_wday; }
            auto GetYearDay() const { return tm_yday; }

            auto GetEMonth() const { return static_cast<EMonth>(tm_mon); }
            auto GetEDay()   const { return static_cast<EDay>(tm_wday); }

            auto GetDaylightSavingsTimeFlag() const { return tm_isdst; }

            EXI friend std::ostream& operator<<(std::ostream& out, const Format& obj);
        };

        void Clear();
        Date& UpdateAll();
        Date& UpdateStr();
        Date& CreateStr();
        Date& CreateAll();
        void  AutoCreateAll();
        ~Date();
        Date() { MnEpochTime = 0; MeZone = ETimeZone::UTC; }
        bool IsNull() const { return MnEpochTime == 0; }
        bool operator!() const { return IsNull(); }
        explicit operator bool() const { return !IsNull(); }

        Date(ETimeZone FeEZone);
        template<typename B, EnableIfSameType(B, bool) = 0>
        Date(ETimeZone FeEZone, B SbCreateAll);
        Date(ETimeZone FeZone, xint FnEpochTime);
        Date(ETimeZone FeZone, ChronoMilliseconds FnChronoTime);
        Date(ETimeZone FeZone, ChronoSeconds FnChronoTime);

        // Format: "2021-06-23 20:00:00"
        Date(ETimeZone FeZone, const xstring& FsDateTime);
        Date(ETimeZone FeZone, const Format& FoFormat);
        Date(ETimeZone FeZone, int FnYear, int FnMonth, int FnDay, int FnHour, int FnMin, int FnSecond);
        Date(ETimeZone FeZone, int FnYear, int FnMonth, int FnDay);

        Date(ETimeZone FeZone, const Date& Other);
        Date(ETimeZone FeZone, Date&& Other) noexcept;
        Date(const Date& Other);
        Date(Date&& Other) noexcept;

        void operator=(const xint FnTime);
        void operator=(const Date::EpochTime FnTime);
        void operator=(const Date& Other);
        void operator=(Date&& Other) noexcept;

        inline Date Dup() const { return Date(MeZone, The); }

    private:
        IS bool SbAutoFormat = false;
        IS bool SbAutoCreateString = false;
        IS bool SbUseDST = false;
        IS RA::ETimeFormat SeDefaultStrFormat = RA::ETimeFormat::DateTime;
    public:
        static void SetAutoFormat(bool SbEnable);
        static void SetAutoCreateStrings(bool SbEnable);
        static void SetDefaultStrFormat(RA::ETimeFormat FeFormat);
        static void SetUseDST(bool SbEnable);

        IS bool BxAutoFormat() { return SbAutoFormat; }
        IS bool BxAutoCreateStrings() { return SbAutoCreateString; }
        IS bool BxUseDST() { return SbUseDST; }
    private:

        const xstring& RetWeekdayTimeStr();
        const xstring& RetDateTimeStr();
        const xstring& RetDateStr();
        const xstring& RetEpochTimeStr();
        const xstring& RetNumericTimeStr();
        const xstring& RetEpochTimeEvenMillisecondsStr();
    public:
        ETimeZone GetZone() const { return MeZone; }

        RIN auto&       GetStr()   const { if (!MsStrPtr) ThrowIt("String Not Created"); return *MsStrPtr; }
        const xstring&  RetStr(RA::ETimeFormat FeFormat = ETimeFormat::None);
              xstring   RetStr(RA::ETimeFormat FeFormat = ETimeFormat::None) const;
        const xstring&  NewStr(RA::ETimeFormat FeFormat = ETimeFormat::None);
              xstring   NewStr(RA::ETimeFormat FeFormat = ETimeFormat::None) const;

        
        RIN auto&       GetFormat() const { if (!MoFormatPtr) ThrowIt("Format Not Created"); return *MoFormatPtr; }
        Date::Format&   RetFormat();
        bool            BxFormatted() const { return MoFormatPtr != nullptr; }
        bool            BxHasString() const { return MsStrPtr != nullptr; }
        static int      GetDaysInMonth(const int FnYear, const int FnMonth);
        int             GetDaysInMonth();
        void            ClampDayToMonth();
        static bool     BxLeapYear(int FnYear);
        bool            BxLeapYear();
        bool            BxEvenHour();
        bool            BxEvenHour(const int FnRound);

        Date::EpochTime    GetEpochTime() const;
        ChronoMilliseconds GetChronoMilliseconds() const;
        ChronoSeconds      GetChronoSeconds() const;

        xint            GetEpochTimeInt() const;
        int             GetEpochTimeInt32() const;
        xint            GetEpochTimeEvenMilliseconds() const { return static_cast<xint>(MnEpochTime) * 1000; }

        static void     SetComputerEZone();
        static int      GetLocalOffsetFromUTC();
        static int      ConvertOffsetToHours(const ETimeZone FeEZone);
        static int      GetRelativeOffsetSeconds(const ETimeZone FeFrom, const ETimeZone FeTo);
        static int      GetRelativeOffsetHours(const ETimeZone FeFrom, const ETimeZone FeTo);
               int      GetSecondsOffset() const;
               int      GetHoursOffset() const;
               int      GetFlippedOffsetHours() const;
               int      GetFlippedOffsetSeconds() const;

        Date ToUTC() const;
        Date ToLocal() const;

        Date& SetYear(const int FnYear);
        Date& SetMonth(const int FnMonth);
        Date& SetDay(const int FnDay);
        Date& SetHour(const int FnHour);
        Date& SetMin(const int FnMin);
        Date& SetSecond(const int FnSecond);

        Date& AdjYear(const int FnYear);
        Date& AdjMonth(const int FnMonth);
        Date& AdjDay(const int FnDay);
        Date& AdjHour(const int FnHour);
        Date& AdjMin(const int FnMin);
        Date& AdjSecond(const int FnSecond);

        Date ToYear(const int FnYear) const;
        Date ToMonth(const int FnMonth) const;
        Date ToDay(const int FnDay) const;
        Date ToHour(const int FnHour) const;
        Date ToMin(const int FnMin) const;
        Date ToSecond(const int FnSecond) const;

        Date& AdjBusinessDay(const int FnDay);
        Date  ToBusinessDay(const int FnDay) const;

        Date& SetToNoon();
        Date& ZeroOutHourMinSec();

        ISC double SnDaysInMonth = 365.25 / 12;

    private:
        // Core Data Members that make up RA::Date
        IS int  SnLocalHoursOffset = 0;
        IS int  SnLocalSecondsOffset = 0;

        IS bool SbEZoneCalculated = false;
        void SetEpochTime(const Date::EpochTime FnEpochTime);
        void SetDateTime(int FnYear, int FnMonth, int FnDay, int FnHour = 0, int FnMin = 0, int FnSecond = 0);
        void SetDateTime();

        ETimeZone MeZone = ETimeZone::LOC;
        Date::EpochTime MnEpochTime = 0;
        ETimeFormat MeStrFormat = ETimeFormat::DateTime;
        xp<xstring> MsStrPtr = nullptr;
        xp<Format> MoFormatPtr = nullptr;

    public:
        bool BxLOC() const { return MeZone == ETimeZone::LOC; }
        bool BxUTC() const { return MeZone == ETimeZone::UTC; }
        bool operator==(const Date& Other) const;
        bool operator==(const xint  FnTime) const;
        bool operator==(const Date::EpochTime FnTime) const;

        bool operator!=(const Date& Other) const;
        bool operator>=(const Date& Other) const;
        bool operator<=(const Date& Other) const;
        bool operator> (const Date& Other) const;
        bool operator< (const Date& Other) const;

        Date  operator+(const xint FnSeconds) const;
        Date  operator-(const xint FnSeconds) const;
        Date  operator*(const xint FnFactor)  const;
        Date  operator/(const xint FnDivisor) const;
        xint  operator-(const Date& Other)    const;

        Date& operator+=(const xint FnSeconds);
        Date& operator-=(const xint FnSeconds);
        Date& operator*=(const xint FnFactor);
        Date& operator/=(const xint FnDivisor);

    private:
        istatic const re2::RE2 SnMatchDateTimeCaptureFull =
            re2::RE2(R"(^(\d\d\d\d)[^\d]+(\d\d)[^\d]+(\d\d)[^\d]+(\d\d)[^\d]+(\d\d)[^\d]+(\d\d).*$)");
        istatic const std::regex SnDateTimeCaptureFull =
            std::regex(R"(^(\d\d\d\d)[^\d]+(\d\d)[^\d]+(\d\d)[^\d]+(\d\d)[^\d]+(\d\d)[^\d]+(\d\d).*$)");
        istatic const std::regex SnDateTimeCaptureHalf =
            std::regex(R"(^(\d\d\d\d)[^\d]+(\d\d)[^\d]+(\d\d).*$)");

        // Global Operators
        EXI friend std::ostream& operator<<(std::ostream& out, const RA::Date& obj);
        EXI friend std::ostream& operator<<(std::ostream& out, const xp<RA::Date> ptr);
    };

    inline static const xp<Date> SoTodayLOCPtr = MKP<RA::Date>(RA::ETimeZone::LOC, true);
    inline static const xp<Date> SoTodayUTCPtr = MKP<RA::Date>(RA::ETimeZone::UTC, true);

    static const inline Date& GetTodayLOC() { return *RA::SoTodayLOCPtr; }
    static const inline Date& GetTodayUTC() { return *RA::SoTodayUTCPtr; }

    static const inline xp<Date>& GetTodayLOCPtr() { return RA::SoTodayLOCPtr; }
    static const inline xp<Date>& GetTodayUTCPtr() { return RA::SoTodayUTCPtr; }

    template<typename B, EnableIfSameType(B, bool) N>
    inline Date::Date(ETimeZone FeEZone, B SbCreateAll) : Date(FeEZone) {
        if (SbCreateAll)
            CreateAll();
    }
};

EXI bool operator==(const RA::Date& Left, const xint Right);
EXI bool operator!=(const RA::Date& Left, const xint Right);
EXI bool operator>=(const RA::Date& Left, const xint Right);
EXI bool operator<=(const RA::Date& Left, const xint Right);
EXI bool operator> (const RA::Date& Left, const xint Right);
EXI bool operator< (const RA::Date& Left, const xint Right);

EXI bool operator==(const xint& Left, const RA::Date& Right);
EXI bool operator!=(const xint& Left, const RA::Date& Right);
EXI bool operator>=(const xint& Left, const RA::Date& Right);
EXI bool operator<=(const xint& Left, const RA::Date& Right);
EXI bool operator> (const xint& Left, const RA::Date& Right);
EXI bool operator< (const xint& Left, const RA::Date& Right);

namespace std {
    template <>
    struct hash<RA::Date> {
        inline xint operator()(const RA::Date& FoDate) const {
            return std::hash<xint>{}(FoDate.GetEpochTimeEvenMilliseconds());
        }
    };
};
