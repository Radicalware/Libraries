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

#include "base_val_xvector.h"

template<typename T> class val_xvector;
template<typename T, typename enabler_t> class xvector;

template<typename T>
class xvector<T, typename std::enable_if<!std::is_class<T>::value>::type> : public val_xvector<T>
{
private:
    using val_xvector<T>::val_xvector;
public:
    inline xvector<T> common(char const* item);
};


template<typename T>
inline xvector<T> xvector<T, typename std::enable_if<!std::is_class<T>::value>::type>::common(char const* item) 
{
 size_t size = strlen(item);
 xvector<T> c_vec(size);

 for (int i = 0; i < size; i++)
     c_vec << item[i];

 return this->common(c_vec);
}
