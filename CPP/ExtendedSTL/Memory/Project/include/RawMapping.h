#pragma once

// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#ifndef __THE__
#define __THE__
#define The  (*this)
#endif

// #include "vcruntime.h"

#include <iostream>
#include <type_traits>
#include <math.h>
#include <functional>
#include <vcruntime.h>
#include <typeinfo>


#if _HAS_CXX20
#define CXX20 1
#endif

using std::cout;
using std::wcout;
using std::endl;

#ifdef _WIN32
#define __CLASS__  __FUNCTION__
#else
#define __CLASS__  __PRETTY_FUNCTION__
#endif

#ifndef _xint_
#define _xint_
typedef          short int int8;
typedef unsigned short int uint8;
typedef unsigned long      uint32;
typedef unsigned long long uint64;
typedef size_t             xint;
#endif


#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    #define BxWindows 1
    #define WIN32_LEAN_AND_MEAN
    
    #ifdef DLL_EXPORT
       #define EXI __declspec(dllexport)
    #else
       #define EXI __declspec(dllimport)
    #endif
#else
    #define BxNix 1
    #define EXI
#endif

#ifdef _DEBUG
#define BxDebug 1
#define BxRelease 0
#else
#define BxRelease 1
#define BxDebug 0
#endif


#define MAX(_LEFT_, _RIGHT_) ((_LEFT_ > _RIGHT_) ? _LEFT_ :  _RIGHT_)
#define MIN(_LEFT_, _RIGHT_) ((_LEFT_ < _RIGHT_) ? _LEFT_ :  _RIGHT_)
#define BxBothPosOrNeg(_LEFT_, _RIGHT_)    ((_LEFT_ < 0 && _RIGHT_ < 0) || (!(_LEFT_ < 0)) && (!(_RIGHT_ < 0)))
#define BxBothAbsPosOrNeg(_LEFT_, _RIGHT_) ((_LEFT_ < 0 && _RIGHT_ < 0) ||   (_LEFT_ > 0   &&   _RIGHT_ > 0))
#define BxBothOverUnder1(_LEFT_, _RIGHT_)  ((_LEFT_ < 1 && _RIGHT_ < 1) || (!(_LEFT_ < 1)) && (!(_RIGHT_ < 1)))

#define istatic inline static
#define VIR virtual
// Runtime Inline
#define RIN inline
// Const INline
#define CIN inline constexpr
// ConST
#define CST const
// Const VARiable
#define cvar const auto
// Non-Const VARiable
#define nvar auto

#define BxIsNotZero(__NUM__) (bool(__NUM__ > 0 || __NUM__ < 0))
#define BxIsZero(__NUM__) (bool(!(BxIsNotZero(__NUM__))))
#define BxPos(_NUM_) (!(_NUM_ < 0))
#define BxIsRealNum(__NUM__) (!(isnan(__NUM__) || isinf(__NUM__)))
#define BxIsRealNumNotZero(__NUM__) (BxIsNotZero(__NUM__) && !(isnan(__NUM__) || isinf(__NUM__)))
#define GetRealNumOrZero(__NUM__) ((BxIsRealNum(__NUM__)) ? __NUM__ : 0)
#define GetRealNumOrOne(__NUM__)  ((BxIsRealNum(__NUM__)) ? __NUM__ : 1)
#define GetPosNumOrOne(__NUM__)   ((__NUM__ > 0) ? __NUM__ : 1)

#define III(__MEM_FUNC__, __RET__) __RET__ __MEM_FUNC__
#define TTT template<typename T>

#define DefaultConstruct(_ClassName_) \
    _ClassName_() = default; \
    _ClassName_(const _ClassName_&) = default; \
    _ClassName_(_ClassName_&&) = default; \
    _ClassName_& operator=(const _ClassName_&) = default; \
    _ClassName_& operator=(_ClassName_&&) = default;

#define NoCopyConstructors(_ClassName_) \
    _ClassName_(_ClassName_&&) = delete; \
    void operator=(_ClassName_&&) = delete; \
    _ClassName_(const _ClassName_&) = delete; \
    void operator=(const _ClassName_&) = delete;

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
// SFINE Helpers

// Mods
#define RemovePtr(_TempObj_)                typename std::remove_pointer<_TempObj_>::type
#define RemoveRef(_TempObj_)                typename std::remove_reference<_TempObj_>::type
#define RemoveExt(_TempObj_)                typename std::remove_extent<_TempObj_>::type
#define RemoveAllExts(_TempObj_)            typename std::remove_all_extents<_TempObj_>::type
#define RemoveConst(_TempObj_)              typename std::remove_const<_TempObj_>::type

// Misc
#define IsSame(_TempObj_, _Comparison_)     std::is_same<_TempObj_, _Comparison_>::value
#define IsType(_Var_, _Type_)               IsSame(RemoveConst(RemoveRef(RemoveAllExts(decltype(_Var_)))), _Type_)
#define IsMutexPtr(_TempObj_)               std::is_same<_TempObj_, RA::SharedPtr<RA::Mutex>>::value
#define IsSharedPtr(_TempObj_)              RA::IsTypeSharedPtr<_TempObj_>()

// Get Configs
#define GetRankCount(_TempObj_)             std::rank<_TempObj_>::value
#define GetVarSize(_TempObj_)               std::alignment_of<_TempObj_>::value


// Standard Types
#define IsClass(_TempObj_)                  (std::is_class<RemoveRef(_TempObj_)>::value)
#define IsObjectSTD(_TempObj_)              (std::is_object<RemoveRef(_TempObj_)>::value && !std::is_invocable<RemoveRef(_TempObj_)>::value)
#define IsPointer(_TempObj_)                std::is_pointer<_TempObj_>::value
#define IsReference(_TempObj_)              std::is_reference<_TempObj_>::value
#define IsFundamental(_TempObj_)            std::is_fundamental<_TempObj_>::value
#define IsArithmetic(_TempObj_)             std::is_arithmetic<_TempObj_>::value
#define IsSigned(_TempObj_)                 std::is_signed<_TempObj_>::value
#define IsUnsigned(_TempObj_)               std::is_unsigned<_TempObj_>::value
#define IsInvocableVA(_Function_, _Args_)   std::is_invocable<RemoveRef(_Function_), _Args_...>::value
#define IsInvocable(_Function_)             std::is_invocable<RemoveRef(_Function_)>::value

#define IsConstructible(_TempObj_)          std::is_constructible<_TempObj_>::value
#define IsDefaultConstructable(_TempObj_)   std::is_default_constructible<_TempObj_>::value
#define IsCopyAssignable(_TempObj_)         std::is_copy_assignable<RemoveRef(_TempObj_)>::value
#define IsTrivial(_TempObj_)                std::is_trivial<RemoveRef(_TempObj_)>::value
#define IsCompound(_TempObj_)               std::is_compound<_TempObj_>::value
#define IsArray(_TempObj_)                  std::is_array<_TempObj_>::value
#define IsUnboundedArray(_TempObj_)         std::is_unbounded_array<_TempObj_>::value
#define IsVolatile(_TempObj_)               std::is_volatile<_TempObj_>::value
#define IsStandardLayout(_TempObj_)         std::is_standard_layout<_TempObj_>::value
#define IsDestructable(_TempObj_)           std::is_destructible<_TempObj_>::value
#define IsScalar(_TempObj_)                 std::is_scalar<_TempObj_>::value
#define IsAggregate(_TempObj_)              std::is_aggregate<RemoveRef(_TempObj_)>::value


#define IsSimple(__TempObj__) \
       std::is_trivially_copyable_v<__TempObj__> \
    && std::is_copy_constructible_v<__TempObj__> \
    && std::is_move_constructible_v<__TempObj__> \
    && std::is_copy_assignable_v<__TempObj__> \
    && std::is_move_assignable_v<__TempObj__>

// Function
#define _InvocableReq_(_Function_) ((IsTrivial(_Function_) &&  IsCopyAssignable(_Function_) &&  IsConstructible(_Function_)) || \
                                   (!IsTrivial(_Function_) && !IsCopyAssignable(_Function_) && !IsConstructible(_Function_)) || \
                                   ( IsTrivial(_Function_) &&  IsCopyAssignable(_Function_) && !IsConstructible(_Function_)))

#define IsFunctionVA(_Function_, _Args_)    std::is_invocable<_Function_, _Args_...>::value
#define IsFunction(_Function_) \
    ((IsInvocable(_Function_) && _InvocableReq_(_Function_)) && !IsAggregate(_Function_) || _InvocableReq_(_Function_) && !IsAggregate(_Function_))

#define IsObject(_Object_) !IsFunction(_Object_)

#define IsBasicFunction(_TempObj_) std::is_function<_TempObj_>::value

// Types
#define TypeIfFunction(_RET_, _Function_)  typename std::enable_if<IsFunction(_Function_), _RET_>::type
#define TypeIfObject(_RET_, _Object_)      typename std::enable_if<IsObject(_Object_), _RET_>::type
#define TypeIfFundamental(_RET_, _Object_) typename std::enable_if<IsFundamental(_Object_), _RET_>::type
#define RetIfTrue(_RET_, _Condition_)      typename std::enable_if<_Condition_, _RET_>::type

#define UsingFunction(_RET_)    TypeIfFunction(_RET_, F)
#define UsingObject(_RET_)      TypeIfObject(_RET_, O)
#define UsingFundamental(_RET_) TypeIfFundamental(_RET_, N)

namespace RA
{
    template<typename O>
    void PrintAttributes(O&& Obj)
    {
        std::cout << "Is Function       : " << IsFunction(O) << std::endl;
        std::cout << "Is Invocable      : " << IsInvocable(O) << std::endl;
        std::cout << "Is Aggregate      : " << IsAggregate(O) << std::endl;
        std::cout << "Is Trivial        : " << IsTrivial(O) << std::endl;
        std::cout << "Is CopyAssignable : " << IsCopyAssignable(O) << std::endl;
        std::cout << "Is Constructible  : " << IsConstructible(O) << std::endl;
        std::cout << '\n';
    }
}

namespace RA
{
    namespace TestAttribute
    {
        template<typename F, typename ...A>
        typename std::enable_if<IsFunction(F), void>::type      Function1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<!IsFunction(F), void>::type     Function1(F&& Function, A&& ...Args) {}
        template<typename O, typename ...A>
        typename std::enable_if<IsObject(O), void>::type        Object1(O& FnObj, A&& ...Args) {}
        template<typename O, typename ...A>
        typename std::enable_if<!IsObject(O), void>::type       Object1(O& FnObj, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<IsInvocable(F), void>::type     Invocable1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<!IsInvocable(F), void>::type    Invocable1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<_InvocableReq_(F), void>::type  InvocReq1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<!_InvocableReq_(F), void>::type InvocReq1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<IsAggregate(F), void>::type     Aggrigate1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<!IsAggregate(F), void>::type    Aggrigate1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<IsTrivial(F), void>::type       Trivial1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<!IsTrivial(F), void>::type      Trivial1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<IsCopyAssignable(F), void>::type    CopyAssignable1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<!IsCopyAssignable(F), void>::type   CopyAssignable1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<IsConstructible(F), void>::type     Constructible1(F&& Function, A&& ...Args) {}
        template<typename F, typename ...A>
        typename std::enable_if<!IsConstructible(F), void>::type    Constructible1(F&& Function, A&& ...Args) {}

        template<typename O, typename F, typename ...A>
        typename std::enable_if<IsFunction(F), void>::type      Function2(O& FnObj, F&& Function, A&& ...Args) {}
        template<typename O, typename F, typename ...A>
        typename std::enable_if<!IsFunction(F), void>::type     Function2(O& FnObj, F&& Function, A&& ...Args) {}
        template<typename O, typename F, typename ...A>
        typename std::enable_if<IsObject(O), void>::type        Object2(O& FnObj, F&& Function, A&& ...Args) {}
        template<typename O, typename F, typename ...A>
        typename std::enable_if<!IsObject(O), void>::type       Object2(O& FnObj, F&& Function, A&& ...Args) {}
        template<typename O, typename F, typename ...A>
        typename std::enable_if<IsInvocable(F), void>::type     Invocable2(O& FnObj, F&& Function, A&& ...Args) {}
        template<typename O, typename F, typename ...A>
        typename std::enable_if<!IsInvocable(F), void>::type    Invocable2(O& FnObj, F&& Function, A&& ...Args) {}
        template<typename O, typename F, typename ...A>
        typename std::enable_if<_InvocableReq_(F), void>::type  InvocReq2(O& FnObj, F&& Function, A&& ...Args) {}
        template<typename O, typename F, typename ...A>
        typename std::enable_if<!_InvocableReq_(F), void>::type InvocReq2(O& FnObj, F&& Function, A&& ...Args) {}
        template<typename O, typename F, typename ...A>
        typename std::enable_if<IsAggregate(F), void>::type     Aggrigate2(O& FnObj, F&& Function, A&& ...Args) {}
        template<typename O, typename F, typename ...A>
        typename std::enable_if<!IsAggregate(F), void>::type    Aggrigate2(O& FnObj, F&& Function, A&& ...Args) {}
    }
}

// SFINE Helpers
// -----------------------------------------------------------------------------------------------------------------------------
// Memory Managment: New, Renew and Delete

#define NewObj(...) GET_MACRO(NewObj, void(), __VA_ARGS__)
#define NewObj2(VD, __VARNAME__, __EXPR__) \
    auto* __VARNAME__##Ptr = new __EXPR__; \
    auto& __VARNAME__ = *__VARNAME__##Ptr;

#define RenewObj(...) GET_MACRO(RenewObj, void(), __VA_ARGS__)
//#define RenewObj2(VD, __VARNAME__, __EXPR__) \
//    if(this->__VARNAME__##Ptr) \
//        delete this->__VARNAME__##Ptr; \
//    this->__VARNAME__##Ptr = new std::remove_pointer<decltype(this->__VARNAME__##Ptr)>::type; \
//    auto& __VARNAME__ = *this->__VARNAME__##Ptr;
#define RenewObj2(VD, __VARNAME__, __EXPR__) \
    if(this->__VARNAME__##Ptr) \
        delete this->__VARNAME__##Ptr; \
    this->__VARNAME__##Ptr = new __EXPR__; \
    auto& __VARNAME__ = *this->__VARNAME__##Ptr;

#define CheckRenewObj(...) GET_MACRO(CheckRenewObj, void(), __VA_ARGS__)
#define CheckRenewObj1(VD, __VARNAME__) \
    if(!this->__VARNAME__##Ptr) \
        this->__VARNAME__##Ptr = new typename std::remove_pointer<decltype(this->__VARNAME__##Ptr)>::type; \
    auto& __VARNAME__ = *this->__VARNAME__##Ptr;
#define CheckRenewObj2(VD, __VARNAME__, __EXPR__) \
    if(!this->__VARNAME__##Ptr) \
        this->__VARNAME__##Ptr = new __EXPR__; \
    auto& __VARNAME__ = *this->__VARNAME__##Ptr;

#ifndef DeleteObj
#define DeleteObj(__DVARNAME__) \
    {if (__DVARNAME__ != nullptr) \
        delete __DVARNAME__; \
    __DVARNAME__ = nullptr;}
#define DeleteArr(__DVARNAME__) \
    {if (__DVARNAME__ != nullptr) \
        delete[] __DVARNAME__; \
    __DVARNAME__ = nullptr;}
#define HostDelete(__DVARNAME__) \
    { \
        if constexpr ((IsArray(decltype(__DVARNAME__)) || IsUnboundedArray(decltype(__DVARNAME__)))) \
            DeleteArr(__DVARNAME__); \
        if constexpr (!(IsArray(decltype(__DVARNAME__)) || IsUnboundedArray(decltype(__DVARNAME__)))) \
            DeleteObj(__DVARNAME__); \
    }

// Simple SET (if you have Macros.h, then use SET)
#define SSET(...) GET_MACRO(SSET, void(), __VA_ARGS__)
#define SSET2(VD,__VARNAME__,__PARAM__) \
    __VARNAME__##Ptr = __PARAM__; \
    if(!__VARNAME__##Ptr) { \
        throw ("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    auto& __VARNAME__ = *__VARNAME__##Ptr;
#define SSET3(VD,__VARNAME__, __POINTER__, __PARAM__) \
    __POINTER__ = __PARAM__; \
    if(!__POINTER__) { \
        throw ("Null Object Thrown, \"" ## #__PARAM__ "\" In Class \"" __CLASS__ "\""); \
    } \
    auto& __VARNAME__ = *__POINTER__;

#if UsingMSVC
#define CudaDelete HostDelete
#else
#define CudaDelete(__DVARNAME__) \
    {if (__DVARNAME__) { \
        cudaFree(__DVARNAME__); \
        __DVARNAME__ = nullptr; \
    }}
#endif

#endif

#ifndef FreeObj
#define FreeObj(__DVARNAME__) \
    if (__DVARNAME__) { \
        free(__DVARNAME__); \
        __DVARNAME__ = nullptr; \
    }
#endif


// -----------------------------------------------------------------------------------------------------------------------------
// Math
#define GetAbs(_NUM_) ((_NUM_ < 0) ? _NUM_ * -1 : _NUM_)
// -----------------------------------------------------------------------------------------------------------------------------
// MISC

template<typename R, typename N>
UsingFundamental(R) Cast(const N& Num) 
{ 
    if constexpr (IsUnsigned(R))
        return static_cast<R>(round(Num));
    else
        return static_cast<R>(Num);
}

template<typename R, typename O>
UsingObject(R&)     Cast(const O& Obj) { return static_cast<R&>(Obj); }

namespace RA
{
    template<typename T1, typename T2>
        T1  Pow( const T1 FnBase, const T2 FnExp);
    TTT T   Sqrt(const T  FnBase);

    double TrimZeros(const double FnNum, const double FnAccuracy);
    float  TrimZeros(const float  FnNum, const float  FnAccuracy);
    bool   Appx(const double FnFirst, const double FnSecond, const double FnAcceptibleRange = 0.0001); // e-4 (cout appx limit)
}

template<typename T1, typename T2>
T1 RA::Pow(const T1 FnBase, const T2 FnExp)
{
    return static_cast<T1>(pow(static_cast<double>(FnBase), static_cast<double>(FnExp)));
}

TTT T RA::Sqrt(const T FnBase)
{
    return static_cast<T>(sqrt(FnBase));
}


// --- CUDA -----------------------------
#ifdef UsingMSVC
// Inline ___ Functions
#define IHF inline
#define IDF inline
#define IXF inline
#define IGF static

// Divergent ___ Functions
#define DDF
#define DHF
#define DXF
#endif

namespace RA
{
    enum class EHardware
    {
        Default,
        CPU,
        GPU
    };
}

// --- CUDA -----------------------------

// -----------------------------------------------------------------------------------------------------------------------------
// Enums can now use cout
template<typename T>
std::ostream& operator<<(typename std::enable_if<std::is_enum<T>::value, std::ostream>::type& Stream, const T& Enum)
{
    return Stream << static_cast<typename std::underlying_type<T>::type>(Enum);
}
// -----------------------------------------------------------------------------------------------------------------------------
// General

// Safe Ref
#define SRef(__VAR__) if(__VAR__) (*__VAR__)

template<typename CallableFunctionT>
using FunctionReturnType = typename decltype(std::function{ std::declval<CallableFunctionT>() })::result_type;
#define GetFunctionReturnType(_TYPE_) FunctionReturnType<decltype(_TYPE_)>

// -----------------------------------------------------------------------------------------------------------------------------
//// Example
//if  constexpr (IsArray(int[]))
//    cout << "yes" << endl; // this will trigger

#ifndef __MACROS_H__
#ifdef  UsingMSVC

#define __A_RESCUE_PRINT__ \
    cout <<   "\nException :>: "<< __CLASS__ <<  " (" << __LINE__ <<  ")\n\n"

#define __W_RESCUE_PRINT__ \
    wcout << L"\nException :>: " << __CLASS__ << L" (" << __LINE__ << L")\n\n"

#define __A_RESCUE_THROW__ \
    cout  << "\nException :>: " << __CLASS__ <<    " (" << __LINE__ <<  ")  "

#define __W_RESCUE_THROW__ \
    wcout << L"\nException :>: " << __CLASS__ <<  L" (" << __LINE__ << L")  "


#define Begin() \
    try {

#define Rescue() \
    }catch(const char* Err){ \
        __A_RESCUE_THROW__ << Err; \
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
    }catch(const std::exception& Err){ \
        __A_RESCUE_THROW__ << Err.what(); \
        throw " "; \
    }catch(...){\
        cout << "Exception Caught: \"Unknown Handler Type\""; \
        throw " "; \
    }

#endif
#endif
