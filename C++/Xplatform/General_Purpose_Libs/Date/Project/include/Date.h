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
    ~Date();
    Date(uint FnEpochTime = 0);
    Date(int FnYear, int FnMonth, int FnDay, int FnHour = 0, int FnMin = 0);

    Date(const Date& date);
    Date(Date&& date) noexcept;
    void operator=(const Date& date);
    void operator=(Date&& date) noexcept;

    void CreateStr();
    xstring& GetStr();
    std::tm& GetTM();

    std::time_t GetEpochTime() const { return MoTime; }

private:
    std::time_t MoTime = 0;
    xstring* MsStr = nullptr;

    //int MnYears = 0;
    //int MnMonths = 0;
    //int MnDays = 0;
    //int MnHours = 0;
    //int MnSeconds = 0;
};

EXI std::ostream& operator<<(std::ostream& out, Date& obj);