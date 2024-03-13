#pragma once

/*
* Copyright[2024][Joel Leagues aka Scourge] under the Apache v2 Licence
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
*/

#include<iostream>
#include<unordered_map>
#include<map>

#ifndef The
#define The (*this)
#endif

// template for "Argument Key-Value-Pairs"
#ifndef AKVP
#define AKVP template<typename KK, typename VV>
#endif

// template for "Argument Key"
#ifndef AKK
#define AKK template<typename KK>
#endif

#ifndef __TypeXINT__
#define __TypeXINT__
using xint = size_t;
#endif

namespace RA
{
    //template<typename K, typename V = size_t, typename H = std::hash<K>, typename C = std::equal_to<K>, typename enabler_t = void> class Mirror;
    template<typename K, typename V = size_t, typename H = std::hash<K>, typename C = std::less<K>, typename enabler_t = void> class Mirror;
};

#define AutoMirror   Mirror<K, V, H, C, typename std::enable_if_t< std::is_same<V, size_t>::value>>
#define ManualMirror Mirror<K, V, H, C, typename std::enable_if_t<!std::is_same<V, size_t>::value>>

