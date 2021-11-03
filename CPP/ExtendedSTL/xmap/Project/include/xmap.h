#pragma once

/*
* Copyright[2019][Joel Leagues aka Scourge]
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
* https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*
* Licensed under the Apache License, const V*ersion 2.0 (the "License");
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

#include<map>
#include<unordered_map>
#include<initializer_list>
#include<utility>
#include<memory>

#include "Nexus.h"
#include "xvector.h"
#include "xstring.h"
#include "Macros.h"

//template<typename K, typename V> class xmap;
//template<typename K, typename V> class xmap<K*, V*>;
//template<typename K, typename V> class xmap<K*, V >;
//template<typename K, typename V> class xmap<K , V*>;

#include "BaseXMap.h"
#include "ValValXMap.h"
#include "PtrPtrXMap.h"
#include "PtrValXMap.h"
#include "ValPtrXMap.h"

