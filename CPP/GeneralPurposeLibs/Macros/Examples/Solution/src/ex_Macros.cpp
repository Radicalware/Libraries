
// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Nexus.h"
#include "Macros.h"

#include <sstream>
#include <assert.h>
//#include "vld.h"


void BadFuncitonWithSSS()
{
    Begin();
    xstring* PlayerNamePtr = nullptr;
    int Kills = 55;
    int Health = 98.555;
    NullThrow(PlayerNamePtr, SSS("Player is Null with Kills: ", Kills, " and Health: ", 98.555));
    Rescue();
}

void BadFunctionWithGET()
{
    Begin();
    xstring* NamePtr = nullptr;
    GET(Name);
    cout << Name << endl;
    delete NamePtr;
    RescueThrow();
}

struct Object
{
    static void TestPointers()
    {
        Begin();
        NewObj(Name, xstring("Riddick"));
        cout << Name << endl;
        DeleteObj(NamePtr);

        BadFunctionWithGET();
        RescueThrow();
    }

    xstring* TheNamePtr = nullptr;

    void FastCreateAndDestroy()
    {
        Begin();
        RenewObj(TheName, xstring("King"));
        cout << TheName << endl;
        DeleteObj(TheNamePtr);
        Rescue();
    }
};


void StlException()
{
    Begin();
    throw std::exception("We are throwing an STL Execption!");
    RescueThrow();
}



//class StaticClass
//{
//public:
//    inline static void PrintLocation()
//    {
//        cout << "__func__     = " << __func__ << endl;
//        cout << "__FUNCTION__ = " << __FUNCTION__ << endl;
//        cout << "__CLASS__    = " << __CLASS__ << endl;
//    }
//};

template<typename ... Args>
void Printer(Args ... args) {
    ((std::cout << args << ' '), ...);
    std::cout << '\n';
}

template<typename ... Args>
void Printer2(const std::string& Str, Args ... args)
{
    Printer(Str, ": ", std::forward<Args>(args)...);
}

void PrintDefault(const std::string& Str1, const std::string& Str2 = "") {
    Printer2(Str1, Str2);
}

#define LogPrint(...) PrintDefault("default text", __VA_ARGS__)

class StaticClass
{
public:
    inline static void PrintLocation()
    {
        std::cout << "test print" << std::endl;
    }
};

void PrintTest() {
    Printer2("first text", "second text");
    LogPrint();
    LogPrint("log text");
}

int main() 
{
    Nexus<>::Start(); 

    StaticClass::PrintLocation();


    Begin();
     BadFuncitonWithSSS();
    RescuePrint();

    Begin();
    Object::TestPointers();
    RescuePrint();

    Begin();
    Object Instance;
    Instance.FastCreateAndDestroy();
    RescuePrint();

    Begin();
    StlException();
    RescuePrint();

    Printer2(__CLASS__, "this is text");
    LogPrint();
    LogPrint("log text");

    cout << GREEN << "easy close" << WHITE << endl;
    return Nexus<>::Stop();
}
