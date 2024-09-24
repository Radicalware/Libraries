#pragma once

#include<vector>
#include<type_traits>
#include<initializer_list>
#include<string>
#include<regex>
#include<sstream>
#include<set>
#include<type_traits>
#include <algorithm>
#include <execution>

#include "re2/re2.h"
#include "Memory.h"
#include "Nexus.h"

namespace RXM {
    using namespace std::regex_constants;
    using Type = syntax_option_type;
}

template<typename T, typename enabler_t = void> class xvector;

template<typename T> class PtrXVector;
template<typename T> class SPtrXVector;
class xstring;

// Values (Object/Primitive)
#define ValObjXVectorAPI  xvector<T, typename std::enable_if_t<!IsFundamental(T) && !IsPointer(T) && NotSharedPtr(T)>>
#define ValPrimXVectorAPI xvector<T, typename std::enable_if_t< IsFundamental(T) && !IsPointer(T)>>
template<typename T> class ValObjXVectorAPI;
template<typename T> class ValPrimXVectorAPI;

// Pointers (Object/Primitive)
#define PtrObjXVectorAPI  xvector<T*, typename std::enable_if_t<!IsFundamental(RemovePtr(T*))>>
#define PtrPrimXVectorAPI xvector<T*, typename std::enable_if_t< IsFundamental(RemovePtr(T*))>>
template<typename T> class PtrObjXVectorAPI;
template<typename T> class PtrPrimXVectorAPI;

// Shared Pointers
#define SPtrObjXVectorAPI xvector<xp<T>, typename std::enable_if_t<!IsFundamental(T)>>
template<typename T> class SPtrObjXVectorAPI;
