#pragma once

#include "SharedPtr.h"
#include "xstring.h"
#include "xvector.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <exception>
#include <cctype>
#include <random>

typedef short int int8;
typedef unsigned short int uint8;
typedef unsigned long      uint32;
typedef unsigned long long uint64;
typedef unsigned long long uint;
typedef unsigned long long pint;

using std::set;
using std::cout;
using std::endl;
using std::wcout;


// -----------------------------------------------------------------------------------------------------------------------------
// Microsoft CPP Rest API

#define GetApiString(__STR__) \
    ItPtr->at(__STR__).as_string()

#define GetApiStr(__STR__) GetApiString(__STR__)


#define GetJsonDoubleFromString(__STR__) \
    std::stod(FoJSON.at(__STR__).as_string().c_str())

#define GetJsonDouble(__STR__) \
    FoJSON.at(__STR__).as_double()

#define GetJsonInt(__STR__) \
    FoJSON.at(__STR__).as_integer()

#define GetJsonString(__STR__) \
    xstring(FoJSON.at(__STR__).as_string().c_str())

#define GetJsonBool(__STR__) \
    FoJSON.at(__STR__).as_bool()

// --------------------------------------------------------
// using inter
#define GetApiDoubleFromString(__STR__) \
    std::stod(ItPtr->at(__STR__).as_string().c_str())
// using inter
#define GetApiDouble(__STR__) \
    ItPtr->at(__STR__).as_double()

// -----------------------------------------------------------------------------------------------------------------------------
// --- MULTI-ARGUMENT FUNCTION HANDLING
// Based on the work of "Rian Quinn" on Stack Overflow

#ifndef __MULTI_ARG_HANDLING
#define __MULTI_ARG_HANDLING
#define _my_BUGFX(x) x

#define _my_NARG2(...) _my_BUGFX(_my_NARG1(__VA_ARGS__,_my_RSEQN()))
#define _my_NARG1(...) _my_BUGFX(_my_ARGSN(__VA_ARGS__))
#define _my_ARGSN(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,N,...) N
#define _my_RSEQN() 9,8,7,6,5,4,3,2,1,0

#define _my_FUNC2(name,n) name ## n
#define _my_FUNC1(name,n)   _my_FUNC2(name,n)
#define GET_MACRO(func,...) _my_FUNC1(func,_my_BUGFX(_my_NARG2(__VA_ARGS__))) (__VA_ARGS__)
#endif // !__MULTI_ARG_HANDLING
// -----------------------------------------------------------------------------------------------------------------------------
// --- Template Alias Handling
// https://www.fluentcpp.com/2017/10/27/function-aliases-cpp/

#define ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF) \
template<typename... Args> \
inline auto highLevelF(Args&&... args) -> decltype(lowLevelF(std::forward<Args>(args)...)) \
{ \
    return lowLevelF(std::forward<Args>(args)...); \
}
// -----------------------------------------------------------------------------------------------------------------------------

#define SSS(...) RA::BindStr(__VA_ARGS__)
#define SSP(...) std::cout << RA::BindStr(__VA_ARGS__) << std::endl;

#define NullThrow(...) GET_MACRO(NullThrow, void(),__VA_ARGS__)
#define NullThrow1(VD, __PARAM__) if (!__PARAM__) { ThrowIt("Null Object Thrown >> \"", #__PARAM__,  "\""); }
#define NullThrow2(VD, __PARAM__,__ERROR_STR__) if (!__PARAM__) \
    { ThrowIt("Null Object Thrown >> \"", #__PARAM__, "\" >>> ", __ERROR_STR__); }

#define NullCheck(...) GET_MACRO(NullCheck, void(), __VA_ARGS__)
#define NullCheck1(VD, __PARAM__) if (!__PARAM__) { return; }
#define NullCheck2(VD, __PARAM__, __RET_TYPE__) if (!__PARAM__) { return __RET_TYPE__ ; }
#define NullCheck3(VD, __PARAM__, __RET_TYPE__, __ERROR_MESSAGE___) if (!__PARAM__) { PrintW(__ERROR_MESSAGE___); return __RET_TYPE__ ; }

// -----------------------------------------------------------------------------------------------------------------------------
// Holding Pointers

#define GetRef(...) GET_MACRO(GetRef, void(), __VA_ARGS__)
#define GetRef1(VD, __VARNAME__) \
    if(!__VARNAME__##Ptr) { \
        ThrowIt("Null Object Thrown >> \"" ## #__VARNAME__ "Ptr\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##Ptr;

#define GetRef2(VD,__VARNAME__,__PARAM__) \
    auto __VARNAME__##Ptr = __PARAM__; \
    if(!__VARNAME__##Ptr) { \
        ThrowIt("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##Ptr;
#define GET(...) GetRef(__VA_ARGS__);

#define SetRef(...) GET_MACRO(SetRef, void(), __VA_ARGS__)
#define SetRef2(VD,__VARNAME__,__PARAM__) \
    __VARNAME__##Ptr = __PARAM__; \
    if(!__VARNAME__##Ptr) { \
        ThrowIt("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##Ptr;
#define SetRef3(VD,__VARNAME__, __POINTER__, __PARAM__) \
    __POINTER__ = __PARAM__; \
    if(!__POINTER__) { \
        ThrowIt("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    auto& __VARNAME__ = *__POINTER__;


#define SET(...) SetRef(__VA_ARGS__);

// GET SMART/SHARED POINTER
#define GSS(...) GET_MACRO(GSS, void(), __VA_ARGS__)
#define GSS1(VD, __VARNAME__) \
    if(!__VARNAME__##Ptr.get()) { \
        ThrowIt("Null Object Thrown >> \"" ## #__VARNAME__ "Ptr\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##Ptr;
#define GSS2(VD,__VARNAME__,__PARAM__) \
    auto __VARNAME__##Ptr = __PARAM__; \
    if(!__VARNAME__##Ptr.get()) { \
        ThrowIt("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##Ptr;

// -----------------------------------------------------------------------------------------------------------------------------
// Exception Handling

#define Begin() \
    try {



#define __A_RESCUE_THROW__ \
    cout  << "\nException :>: " << __CLASS__ <<    " (" << __LINE__ <<  ")  "

#define __W_RESCUE_THROW__ \
    wcout << L"\nException :>: " << __CLASS__ <<  L" (" << __LINE__ << L")  "


#define RescueThrow(...) GET_MACRO(RescueThrow, void(), __VA_ARGS__)

#define RescueThrow0(VD) \
    }catch(const xstring& Err){ \
        __A_RESCUE_THROW__ << Err; \
        throw " "; \
    }catch(const char* Err){ \
        __A_RESCUE_THROW__ << Err; \
        throw " "; \
    }catch(const std::exception& Err){ \
        __A_RESCUE_THROW__ << Err.what(); \
        throw " "; \
    }catch(const std::string& Err){ \
        __A_RESCUE_THROW__ << Err; \
        throw " "; \
    }catch(const std::wstring& Err){ \
        __W_RESCUE_THROW__ << Err; \
        throw " "; \
    }catch (const wchar_t* Err){ \
        __W_RESCUE_THROW__ << Err; \
        throw " "; \
    }catch(...){\
        cout << "Exception Caught: \"Unknown Handler Type\""; \
        throw " "; \
    }

// auto Lambda = std::bind(<lambda>); 
// Rescue(Lambda);
#define RescueThrow1(VD, _Lambda_) \
    }catch(const xstring& Err){ \
        _Lambda_(); \
        __A_RESCUE_THROW__ << Err; \
        throw " "; \
    }catch(const char* Err){ \
        _Lambda_(); \
        __A_RESCUE_THROW__ << Err; \
        throw " "; \
    }catch(const std::exception& Err){ \
        _Lambda_(); \
        __A_RESCUE_THROW__ << Err.what(); \
        throw " "; \
    }catch(const std::string& Err){ \
        _Lambda_(); \
        __A_RESCUE_THROW__ << Err; \
        throw " "; \
    }catch(const std::wstring& Err){ \
        _Lambda_(); \
        __W_RESCUE_THROW__ << Err; \
        throw " "; \
    }catch (const wchar_t* Err){ \
        _Lambda_(); \
        __W_RESCUE_THROW__ << Err; \
        throw " "; \
    }catch(...){\
        _Lambda_(); \
        cout << "Exception Caught: \"Unknown Handler Type\""; \
        throw " "; \
    }

#define __A_RESCUE_PRINT__ \
    cout <<   "\nException :>: "<< __CLASS__ <<  " (" << __LINE__ <<  ")\n\n"

#define __W_RESCUE_PRINT__ \
    wcout << L"\nException :>: " << __CLASS__ << L" (" << __LINE__ << L")\n\n"


#define RescuePrint(...) GET_MACRO(RescuePrint, void(), __VA_ARGS__)

#define RescuePrint0(VD) \
    }catch(const xstring& Err){ \
        __A_RESCUE_PRINT__ << Err << '\n'; \
    }catch(const char* Err){ \
        __A_RESCUE_PRINT__ << Err << '\n'; \
    }catch(const std::exception& Err){ \
        __A_RESCUE_PRINT__ << Err.what() << '\n'; \
    }catch(const std::string& Err){ \
        __A_RESCUE_PRINT__ << Err << '\n'; \
    }catch(const std::wstring& Err){ \
        __W_RESCUE_PRINT__ << Err << L'\n'; \
    }catch (const wchar_t* Err){ \
        __W_RESCUE_PRINT__ << Err << L'\n'; \
    }catch(...){\
        cout << "Exception Caught: \"Unknown Handler Type\""; \
    }

#define RescuePrint1(VD, _Lambda_) \
    }catch(const xstring& Err){ \
        _Lambda_(); \
        __A_RESCUE_PRINT__ << Err << '\n'; \
    }catch(const char* Err){ \
        _Lambda_(); \
        __A_RESCUE_PRINT__ << Err << '\n'; \
    }catch(const std::exception& Err){ \
        _Lambda_(); \
        __A_RESCUE_PRINT__ << Err.what() << '\n'; \
    }catch(const std::string& Err){ \
        _Lambda_(); \
        __A_RESCUE_PRINT__ << Err << '\n'; \
    }catch(const std::wstring& Err){ \
        _Lambda_(); \
        __W_RESCUE_PRINT__ << Err << L'\n'; \
    }catch (const wchar_t* Err){ \
        _Lambda_(); \
        __W_RESCUE_PRINT__ << Err << L'\n'; \
    }catch(...){\
        _Lambda_(); \
        cout << "Exception Caught: \"Unknown Handler Type\""; \
    }


#define Rescue(...) GET_MACRO(Rescue, void(), __VA_ARGS__)
#define Rescue0(VD)           RescueThrow0(void);
#define Rescue1(VD, _Lambda_) RescueThrow1(void, _Lambda_);


#define FinalRescue(...) GET_MACRO(FinalRescue, void(), __VA_ARGS__)
#define FinalRescue0(VD)           RescuePrint0(void);
#define FinalRescue1(VD, _Lambda_) RescuePrint1(void, _Lambda_);

// -----------------------------------------------------------------------------------------------------------------------------
namespace RA
{
    template <typename FF>
    void _Support_JoinStr(std::ostringstream& SS, const FF& FnArg) {
        SS << FnArg;
    }

    template <typename FF, typename... RR>
    void _Support_JoinStr(std::ostringstream& SS, const FF& Frist, const RR&... Rest) {
        SS << Frist;
        RA::_Support_JoinStr(SS, Rest...);
    }

    template <typename FF, typename... RR>
    xstring BindStr(const FF& Frist, const RR&... Rest)
    {
        std::ostringstream SS;
        RA::_Support_JoinStr(SS, Frist, Rest...);
        return SS.str();
    }
};

#define ThrowIt(...) \
    throw RA::BindStr("\n  >>> ......... ", __CLASS__, " (", __LINE__, ")  " \
                       ,"\n  >>> ......... ", __VA_ARGS__)

// -----------------------------------------------------------------------------------------------------------------------------

namespace RA
{
    template<typename T = int>
    xvector<T> GetRandomSequence(const pint Length, const pint Floor, const pint Ceiling)
    {
        std::random_device os_seed;
        const uint_least32_t seed = os_seed();

        std::mt19937 generator(seed);
        std::uniform_int_distribution<T> distribute(Floor, Ceiling);

        xvector<T> Sequence;
        for (int repetition = 0; repetition < Length; ++repetition)
            Sequence << distribute(generator);

        return Sequence;
    }

    template<>
    xvector<char> GetRandomSequence<char>(const pint Length, const pint Floor, const pint Ceiling);

    xstring GetRandomStr(const pint Length, const pint Floor, const pint Ceiling);

}

// -----------------------------------------------------------------------------------------------------------------------------
