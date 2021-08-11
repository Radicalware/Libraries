#pragma once

#include "xstring.h"
#include "xvector.h"
#include <iostream>
#include <memory>
#include <sstream>

using uint = size_t;
typedef short int int8;
typedef short unsigned int uint8;

using std::cout;
using std::endl;
using std::wcout;

#define SharedPtr std::shared_ptr
#define MakePtr   std::make_shared

#ifdef _WIN32
#define __CLASS__  __FUNCTION__
#else
#define __CLASS__  __PRETTY_FUNCTION__
#endif

// ------------- MULTI-ARGUMENT FUNCTION HANDLING -----------------------------------------------------------------------------------------

#define _my_BUGFX(x) x

#define _my_NARG2(...) _my_BUGFX(_my_NARG1(__VA_ARGS__,_my_RSEQN()))
#define _my_NARG1(...) _my_BUGFX(_my_ARGSN(__VA_ARGS__))
#define _my_ARGSN(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,N,...) N
#define _my_RSEQN() 9,8,7,6,5,4,3,2,1,0

#define _my_FUNC2(name,n) name ## n
#define _my_FUNC1(name,n) _my_FUNC2(name,n)
#define GET_MACRO(func,...) _my_FUNC1(func,_my_BUGFX(_my_NARG2(__VA_ARGS__))) (__VA_ARGS__)

// ------------- DEBUGGING ----------------------------------------------------------------------------------------------------------------

#define SSS(...) RA::JoinToStr(__VA_ARGS__)

#define NullThrow(...) GET_MACRO(NullThrow, void(),__VA_ARGS__)
#define NullThrow1(VD, __PARAM__) if (!__PARAM__) { throw RetThrow("Null Object Thrown >> \"", #__PARAM__,  "\""); }
#define NullThrow2(VD, __PARAM__,__ERROR_STR__) if (!__PARAM__) \
    { throw RetThrow("Null Object Thrown >> \"", #__PARAM__, "\" >>> ", __ERROR_STR__); }

#define NullCheck(...) GET_MACRO(NullCheck, void(), __VA_ARGS__)
#define NullCheck1(VD, __PARAM__) if (!__PARAM__) { return; }
#define NullCheck2(VD, __PARAM__, __RET_TYPE__) if (!__PARAM__) { return __RET_TYPE__ ; }
#define NullCheck3(VD, __PARAM__, __RET_TYPE__, __ERROR_MESSAGE___) if (!__PARAM__) { PrintW(__ERROR_MESSAGE___); return __RET_TYPE__ ; }


#define GetRef(...) GET_MACRO(GetRef, void(), __VA_ARGS__)
#define GetRef1(VD, __VARNAME__) \
    if(!__VARNAME__##Ptr) { \
        throw RetThrow("Null Object Thrown >> \"" ## #__VARNAME__ "Ptr\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##Ptr;

#define GET(...) GetRef(__VA_ARGS__);


#define Begin() \
    try {
#define Rescue() \
    } \
    catch (const xstring& Err) { \
        Err.Print(); \
    }

namespace RA
{
    template <typename FF>
    void _Support_JoinStr(std::stringstream& SS, FF&& FnArg){
        SS << FnArg;
    }
    void _Support_JoinStr(std::stringstream& SS, const char* FnArg){
        SS << FnArg;
    }

    template <typename FF, typename... RR>
    void _Support_JoinStr(std::stringstream& SS, const FF& Frist, const RR&... Rest) {
        SS << Frist;
        RA::_Support_JoinStr(SS, Rest...);
    }
    //void _Support_JoinStr(std::stringstream& SS, const char* Frist, const char* Rest...) {
    //    SS << Frist;
    //    RA::_Support_JoinStr(SS, Rest...);
    //}

    template <typename FF, typename... RR>
    xstring JoinToStr(const FF& Frist, const RR&... Rest)
    {
        std::stringstream SS;
        RA::_Support_JoinStr(SS, Frist, Rest...);
        return SS.str();
    }

};

#define RetThrow(...) \
    RA::JoinToStr("Threw: \"", __CLASS__, "\" >> ", __VA_ARGS__)