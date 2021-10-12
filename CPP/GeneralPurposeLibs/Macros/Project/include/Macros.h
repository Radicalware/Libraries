#pragma once

#include "xstring.h"
#include "xvector.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <exception>

using pint = size_t;
typedef short int int8;
typedef short unsigned int uint8;

using std::cout;
using std::endl;
using std::wcout;

#define The (*this)

#define SharedPtr std::shared_ptr
#define MakePtr   std::make_shared

#ifdef _WIN32
    #define __CLASS__  __FUNCTION__
#else
    #define __CLASS__  __PRETTY_FUNCTION__
#endif

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    #define WIN_BASE
#else
    #define NIX_BASE
#endif

#ifndef EXI
    #if (defined(WIN_BASE))
        #ifdef DLL_EXPORT
           #define EXI __declspec(dllexport)
        #else
           #define EXI __declspec(dllimport)
        #endif
    #else
        #define EXI
    #endif
#endif

// -----------------------------------------------------------------------------------------------------------------------------
// Enums can now use cout
template<typename T>
std::ostream& operator<<(typename std::enable_if<std::is_enum<T>::value, std::ostream>::type& stream, const T& e)
{
    return stream << static_cast<typename std::underlying_type<T>::type>(e);
}

// -----------------------------------------------------------------------------------------------------------------------------
// Microsoft CPP Rest API

#define GetApiString(__STR__) \
    ItPtr->at(__STR__).as_string()

#define GetApiStr(__STR__) GetApiString(__STR__)

#define GetApiDoubleFromString(__STR__) \
    std::stod(ItPtr->at(__STR__).as_string().c_str())

#define GetApiDouble(__STR__) \
    ItPtr->at(__STR__).as_double()

// -----------------------------------------------------------------------------------------------------------------------------
// --- MULTI-ARGUMENT FUNCTION HANDLING
// Based on the work of "Rian Quinn" on Stack Overflow

#define _my_BUGFX(x) x

#define _my_NARG2(...) _my_BUGFX(_my_NARG1(__VA_ARGS__,_my_RSEQN()))
#define _my_NARG1(...) _my_BUGFX(_my_ARGSN(__VA_ARGS__))
#define _my_ARGSN(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,N,...) N
#define _my_RSEQN() 9,8,7,6,5,4,3,2,1,0

#define _my_FUNC2(name,n) name ## n
#define _my_FUNC1(name,n)   _my_FUNC2(name,n)
#define GET_MACRO(func,...) _my_FUNC1(func,_my_BUGFX(_my_NARG2(__VA_ARGS__))) (__VA_ARGS__)

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

#define SSS(...) RA::JoinToStr(__VA_ARGS__)
#define SSP(...) std::cout << RA::JoinToStr(__VA_ARGS__) << std::endl;

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
    auto __VARNAME__##_PTR = __PARAM__; \
    if(!__VARNAME__##_PTR) { \
        ThrowIt("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##_PTR;

#define GET(...) GetRef(__VA_ARGS__);

// GET SMART/SHARED POINTER
#define GSS(...) GET_MACRO(GSS, void(), __VA_ARGS__)
#define GSS1(VD, __VARNAME__) \
    if(!__VARNAME__##Ptr.get()) { \
        ThrowIt("Null Object Thrown >> \"" ## #__VARNAME__ "Ptr\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##Ptr;

// -----------------------------------------------------------------------------------------------------------------------------
// Memory Managment: New, Renew and Delete

#define NewObject(...) GET_MACRO(NewObject, void(), __VA_ARGS__)
#define NewObject2(VD, __VARNAME__, __EXPR__) \
    auto* __VARNAME__##Ptr = new __EXPR__; \
    GetRef1(void(), __VARNAME__)

#define RenewObject(...) GET_MACRO(RenewObject, void(), __VA_ARGS__)
#define RenewObject2(VD, __VARNAME__, __EXPR__) \
    if(__VARNAME__##Ptr) \
        delete __VARNAME__##Ptr; \
    __VARNAME__##Ptr = new __EXPR__; \
    GetRef1(void(), __VARNAME__)

#ifndef DeleteObject
#define DeleteObject(__DVARNAME__) \
    if (__DVARNAME__) { \
        delete __DVARNAME__; \
        __DVARNAME__ = nullptr; \
    }
#endif

// -----------------------------------------------------------------------------------------------------------------------------
// Exception Handling

#define Begin() \
    try {



#define __A_RESCUE_THROW__ \
    cout  << "\nException :>: " << __CLASS__ <<    " (" << __LINE__ <<  ")  "

#define __W_RESCUE_THROW__ \
    wcout << L"\nException :>: " << __CLASS__ <<  L" (" << __LINE__ << L")  "

#define RescueThrow() \
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

#define __A_RESCUE_PRINT__ \
    cout <<   "\nException :>: "<< __CLASS__ <<  " (" << __LINE__ <<  ")\n\n"

#define __W_RESCUE_PRINT__ \
    wcout << L"\nException :>: " << __CLASS__ << L" (" << __LINE__ << L")\n\n"

#define RescuePrint() \
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

#define Rescue() RescueThrow()
#define FinalRescue() RescuePrint()

// -----------------------------------------------------------------------------------------------------------------------------
namespace RA
{
    template <typename FF>
    void _Support_JoinStr(std::stringstream& SS, const FF& FnArg) {
        SS << FnArg;
    }

    template <typename FF, typename... RR>
    void _Support_JoinStr(std::stringstream& SS, const FF& Frist, const RR&... Rest) {
        SS << Frist;
        RA::_Support_JoinStr(SS, Rest...);
    }

    template <typename FF, typename... RR>
    xstring JoinToStr(const FF& Frist, const RR&... Rest)
    {
        std::stringstream SS;
        RA::_Support_JoinStr(SS, Frist, Rest...);
        return SS.str();
    }
};

#define ThrowIt(...) \
    throw RA::JoinToStr("\n  >>> ......... ", __CLASS__, " (", __LINE__, ")  " \
                       ,"\n  >>> ......... ", __VA_ARGS__)