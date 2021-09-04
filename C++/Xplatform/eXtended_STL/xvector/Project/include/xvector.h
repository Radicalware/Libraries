#pragma once
#pragma warning (disable : 26444) // allow anynomous objects

/*
* Copyright[2019][Joel Leagues aka Scourge]
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
* https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http ://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/


#if (defined(WIN64) || defined(_WIN64) || defined(WIN32) || defined(_WIN32))
    using size64_t = __int64;
#else
    #include <cstdint>
    using size64_t = int64_t;
#endif

#include<vector>
#include<type_traits>
#include<initializer_list>
#include<string>
#include<regex>
#include<sstream>
#include<set>
#include<type_traits>

#include "re2/re2.h"
#include "Nexus.h"

template<typename T> class val_xvector;
template<typename T> class ptr_xvector;

template<typename T, typename enabler_t = void> class xvector;
    
// Values (Object/Primitive)
template<typename T> class xvector<T, typename std::enable_if< std::is_class<T>::value && !std::is_pointer<T>::value>::type>; // val_obj_xvector
template<typename T> class xvector<T, typename std::enable_if<!std::is_class<T>::value && !std::is_pointer<T>::value>::type>; // val_prim_xvector

// Pointers (Object/Primitive)
template<typename T> class xvector<T*, typename std::enable_if< std::is_class<std::remove_pointer_t<T*>>::value>::type>; // ptr_obj_xvector
template<typename T> class xvector<T*, typename std::enable_if<!std::is_class<std::remove_pointer_t<T*>>::value>::type>; // ptr_prim_xvector

#include "base_val_xvector.h"
#include "val_obj_xvector.h"
#include "val_prim_xvector.h"

#include "base_ptr_xvector.h"
#include "ptr_obj_xvector.h"
#include "ptr_prim_xvector.h"
