#pragma once

#include<sstream>
#include<math.h>
#include<string>
#include<memory>

#include "xstring.h"
#include "xvector.h"

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

class EXI Date
{
public:
    enum class Offset
    {
        None,

        Local,
        UTC,

        ToUTC,
        ToLocal
    };

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

    void operator=(const Date& Other);
    void operator=(Date&& Other) noexcept;

    void CreateStr();
    const xstring& GetStr();
    std::tm GetTime();

    std::time_t GetEpochTime() const;
    xstring     GetEpochTimeStr() const;
    xstring     GetNumericTimeStr();

    static int GetSecondsOffset();
    static int GetHoursOffset();

    Date Year(int FnYear);
    Date Month(int FnMonth);
    Date Day(int FnDay);
    Date Hour(int FnHour);
    Date Min(int FnMin);
    Date Second(int FnSecond);

private:
    static bool SbAppliedLocalOffset;
    static int  SnLocalOffset;
    static int  GetSecondsOffset(Offset FeOffset);


    void SetDateTime(int FnYear, int FnMonth, int FnDay, int FnHour = 0, int FnMin = 0, int FnSecond = 0);
    void SetDateTime(const std::tm& FnTime);

    std::time_t MoEpochTime = 0;
    xstring* MsStr = nullptr;
    std::tm  MoTime;

public:
    bool operator==(const Date& Other) const;
    bool operator!=(const Date& Other) const;
    bool operator>=(const Date& Other) const;
    bool operator<=(const Date& Other) const;
    bool operator> (const Date& Other) const;
    bool operator< (const Date& Other) const;
};

EXI std::ostream& operator<<(std::ostream& out,       Date& obj);