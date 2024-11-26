#pragma once

/*
*|| Copyright[2023][Joel Leagues aka Scourge]
*|| Scourge /at\ protonmail /dot\ com
*|| www.Radicalware.net
*|| https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*||
*|| Licensed under the Apache License, Version 2.0 (the "License");
*|| you may not use this file except in compliance with the License.
*|| You may obtain a copy of the License at
*||
*|| http ://www.apache.org/licenses/LICENSE-2.0
*||
*|| Unless required by applicable law or agreed to in writing, software
*|| distributed under the License is distributed on an "AS IS" BASIS,
*|| WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*|| See the License for the specific language governing permissions and
*|| limitations under the License.
*/

#define __MACROS_H__

#pragma warning(disable:4251)
// allows you to have DLLs without needing a interface


#include "SharedPtr.h"
#include "SharedPtr/ReferencePtr.h"
#include "xstring.h"
#include "xvector.h"
#include "RawMapping.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <cctype>
#include <random>
#include <exception>
#include <assert.h>

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

#define NEW(...) GET_MACRO(NEW, void(), __VA_ARGS__)
#define NEW2(VD,__VARNAME__,__PARAM__) \
    auto __VARNAME__##Ptr = __PARAM__; \
    if(!__VARNAME__##Ptr) { \
        ThrowIt("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##Ptr;
#define NEW3(VD,__BASE_TYPE__,__VARNAME__,__PARAM__) \
    RA::SharedPtr<__BASE_TYPE__> __VARNAME__##Ptr = __PARAM__; \
    if(!__VARNAME__##Ptr) { \
        ThrowIt("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    __BASE_TYPE__& __VARNAME__ = *__VARNAME__##Ptr;

#define SET(...) GET_MACRO(SET, void(), __VA_ARGS__)
#define SET2(VD,__VARNAME__,__PARAM__) \
    __VARNAME__##Ptr = __PARAM__; \
    if(!__VARNAME__##Ptr) { \
        ThrowIt("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##Ptr;
#define SET3(VD,__VARNAME__, __POINTER__, __PARAM__) \
    __POINTER__ = __PARAM__; \
    if(!__POINTER__) { \
        ThrowIt("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    auto& __VARNAME__ = *__POINTER__;

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


#define RenewPtr(__VARNAME__, __EXP__) \
    if(!__VARNAME__##Ptr) { \
        HostDelete(__VARNAME__##Ptr); \
        __VARNAME__##Ptr = __EXP__; \
    } else \
        __VARNAME__##Ptr = __EXP__; \
    auto& __VARNAME__ = *__VARNAME__##Ptr;

// -----------------------------------------------------------------------------------------------------------------------------
// Exception Handling

#ifndef Begin
#define Begin() \
    try {
#endif


#ifndef __A_RESCUE_THROW__
#define __A_RESCUE_THROW__ \
    cout  << "\nException :>: " << __CLASS__ <<    " (" << __LINE__ <<  ")  "
#endif

#ifndef __W_RESCUE_THROW__
#define __W_RESCUE_THROW__ \
    wcout << L"\nException :>: " << __CLASS__ <<  L" (" << __LINE__ << L")  "
#endif

#undef  RescueThrow
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

#ifndef __A_RESCUE_PRINT__
#define __A_RESCUE_PRINT__ \
    cout <<   "\nException :>: "<< __CLASS__ <<  " (" << __LINE__ <<  ")\n\n"
#endif 

#ifndef __W_RESCUE_PRINT__
#define __W_RESCUE_PRINT__ \
    wcout << L"\nException :>: " << __CLASS__ << L" (" << __LINE__ << L")\n\n"
#endif

#undef  RescuePrint
#define RescuePrint(...) GET_MACRO(RescuePrint, void(), __VA_ARGS__)
#define FinalRescue(...) RescuePrint(__VA_ARGS__)

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


#undef  Rescue
#define Rescue(...) GET_MACRO(Rescue, void(), __VA_ARGS__)
#define Rescue0(VD)           RescueThrow0(void);
#define Rescue1(VD, _Lambda_) RescueThrow1(void, _Lambda_);

#undef  FinalRescue
#define FinalRescue(...) GET_MACRO(FinalRescue, void(), __VA_ARGS__)
#define FinalRescue0(VD)           RescuePrint0(void);
#define FinalRescue1(VD, _Lambda_) RescuePrint1(void, _Lambda_);

#define WATCH(__EXPR__) Begin(); __EXPR__; Rescue();

//#if UsingNVCC
//
//#define NVBegin Begin
//#define NVRescue Rescue
//#define NVFinalRescue FinalRescue
//
//#undef Begin
//#undef Rescue
//#undef FinalRescue
//
//#define Begin()
//#define Rescue()
//#define FinalRescue()
//#endif

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


#define AssertEqualDbl(...) GET_MACRO(AssertEqualDbl, void(),__VA_ARGS__)
#define AssertEqualDbl2(VD,_VAL1_, _VAL2_) \
    if (!RA::Appx(_VAL1_, _VAL2_)){ \
        ThrowIt("(", #_VAL1_, " != ", #_VAL2_, ")(", _VAL1_, " != ", _VAL2_, ")"); \
    }
#define AssertEqualDbl3(VD,_VAL1_, _VAL2_,_Clearance_) \
    if (!RA::Appx(_VAL1_, _VAL2_,_Clearance_)){ \
        ThrowIt("(", #_VAL1_, " != ", #_VAL2_, ")(", _VAL1_, " != ", _VAL2_, ")"); \
    }
#define AssertEqualDbl4(VD,_VAL1_, _VAL2_,_Bound1_,_Bound2_) \
    if (!RA::Appx(_VAL1_, _VAL2_,_Bound1_,_Bound2_)){ \
        ThrowIt("(", #_VAL1_, " != ", #_VAL2_, ")(", _VAL1_, " != ", _VAL2_, ")"); \
    }

#define AssertEqual(_VAL1_, _VAL2_) \
    if (_VAL1_ != _VAL2_){\
        ThrowIt("(", #_VAL1_, " != ", #_VAL2_, ")(", _VAL1_, " != ", _VAL2_, ")"); \
    }

#define AssertDblRange(_Upper_, _Val_, _Lower_) \
    if (_Val_ > _Upper_ || _Lower_ > _Val_){ \
        ThrowIt("!(", _Upper_, " >= ", _Val_, " >= ", _Lower_, ")"); \
    }

#define ThrowFailingCompare(_Left_, _Op_, _Right_) \
    if(!(_Left_ _Op _Right_)) { \
        ThrowIt("<(", _Left_, ' ', #_Op_, ' ', _Right_, ")>|<("#_Left_,' ', #_Op_, ' ', #_Right_,")>"); \
    }


// -----------------------------------------------------------------------------------------------------------------------------

namespace RA
{
    struct Rand
    {
        istatic std::random_device SfSeed;
        istatic const uint_least32_t SoSeed = SfSeed();
        istatic std::mt19937 SoGenerator = std::mt19937(SoSeed);

        TTT istatic auto GetDistributor(const T FnFloor, const T FnCeiling) {
            return std::uniform_int_distribution<xint>(static_cast<xint>(FnFloor), static_cast<xint>(FnCeiling)); }

        TTT istatic T GetValue(std::uniform_int_distribution<T>& FoDistributor)
        {
            return FoDistributor(SoGenerator);
        }

        TTT istatic T GetValue(const T FnFloor = 0, const T FnCeiling = 100)
        {
            auto LoDistributor = GetDistributor<T>(FnFloor, FnCeiling);
            return LoDistributor(SoGenerator);
        }

        TTT istatic xvector<T> GetRandomSequence(const xint Length, const T FnFloor, const T FnCeiling)
        {
            auto LoDistributor = GetDistributor<xint>(FnFloor, FnCeiling);
            xvector<T> Sequence;
            for (xint repetition = 0; repetition < Length; repetition++)
                Sequence << static_cast<T>(LoDistributor(SoGenerator));

            return Sequence;
        }

        static xvector<char> GetRandomCharSequence(const xint Length, const xint Floor, const xint Ceiling);
        static xstring       GetRandomStr(const xint Length, const xint Floor, const xint Ceiling);
    };


    template<typename... A>
    xstring Cat(A... Args) // Concatenate
    {
        std::ostringstream OSS;
        (OSS << ... << Args);
        return OSS;
    }

    template<typename... A>
    void Print(A... Args) // Concatenate
    {
        std::ostringstream OSS;
        (OSS << ... << Args);
        printf("%s\n", OSS.str().c_str());
    }

    template<typename L, typename R>
    auto GetRealNum(const L FnLeft, const R FnRight)
    {
        Begin();
        if (BxIsRealNum(FnLeft))
            return FnLeft;
        else if (BxIsRealNum(FnRight))
            return FnRight;
        ThrowIt("No Real Number");
        Rescue();
    }

    template<typename L, typename R>
    auto GetRealNonZero(const L FnLeft, const R FnRight)
    {
        Begin();
        if (BxIsRealNum(FnLeft) && FnLeft != 0)
            return FnLeft;
        else if (BxIsRealNum(FnRight) && FnRight != 0)
            return FnRight;
        ThrowIt("No Real Number");
        Rescue();
    }

#ifdef UsingMSVC
#define TestRealNum(_Num_) \
    if (!BxIsRealNum(_Num_)) ThrowIt("Not a Number: ", #_Num_);
#else // UsingMSVC
#define TestRealNum(_Num_) \
    if (!BxIsRealNum(_Num_)) printf("Not a Number: %llf\n", _Num_);
#endif

}

// -----------------------------------------------------------------------------------------------------------------------------
