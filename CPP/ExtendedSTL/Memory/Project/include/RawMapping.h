#pragma once

#ifndef __THE__
#define __THE__
#define The  (*this)
#endif

#ifdef _WIN32
    #define __CLASS__  __FUNCTION__
#else
    #define __CLASS__  __PRETTY_FUNCTION__
#endif

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
    #define WINDOWS
    #define WIN_BASE

#else
    #define NIX
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
// Memory Managment: New, Renew and Delete

#define NewObject(...) GET_MACRO(NewObject, void(), __VA_ARGS__)
#define NewObject2(VD, __VARNAME__, __EXPR__) \
    auto* __VARNAME__##Ptr = new __EXPR__; \
    auto& __VARNAME__ = *__VARNAME__##_PTR;

#define RenewObject(...) GET_MACRO(RenewObject, void(), __VA_ARGS__)
#define RenewObject1(VD, __VARNAME__, __EXPR__) \
    if(this->__VARNAME__##Ptr) \
        delete this->__VARNAME__##Ptr; \
    this->__VARNAME__##Ptr = new std::remove_pointer<decltype(this->__VARNAME__##Ptr)>::type; \
    auto& __VARNAME__ = *this->__VARNAME__##Ptr;
#define RenewObject2(VD, __VARNAME__, __EXPR__) \
    if(this->__VARNAME__##Ptr) \
        delete this->__VARNAME__##Ptr; \
    this->__VARNAME__##Ptr = new __EXPR__; \
    auto& __VARNAME__ = *this->__VARNAME__##Ptr;

#define CheckRenewObject(...) GET_MACRO(CheckRenewObject, void(), __VA_ARGS__)
#define CheckRenewObject1(VD, __VARNAME__) \
    if(!this->__VARNAME__##Ptr) \
        this->__VARNAME__##Ptr = new std::remove_pointer<decltype(this->__VARNAME__##Ptr)>::type; \
    auto& __VARNAME__ = *this->__VARNAME__##Ptr;
#define CheckRenewObject2(VD, __VARNAME__, __EXPR__) \
    if(!this->__VARNAME__##Ptr) \
        this->__VARNAME__##Ptr = new __EXPR__; \
    auto& __VARNAME__ = *this->__VARNAME__##Ptr;

#ifndef DeleteObject
#define Delete(__DVARNAME__) \
    if (__DVARNAME__) { \
        delete __DVARNAME__; \
        __DVARNAME__ = nullptr; \
    }
#endif

// -----------------------------------------------------------------------------------------------------------------------------



