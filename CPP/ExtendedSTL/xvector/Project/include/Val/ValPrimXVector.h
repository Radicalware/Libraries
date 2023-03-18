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
#include "BaseXVector.h"
#include "BaseValXVector.h"
#include <type_traits>

template<typename T>
class ValPrimXVectorAPI : public ValXVector<T>
{
public:
    using ValXVector<T>::ValXVector;
    using ValXVector<T>::operator=;
    using E = typename std::remove_const<T>::type; // E for Erratic

    typedef T value_type;

    RIN T GetCommonItems(char const* item);
    template<typename S = std::string>
    RIN S Join(const S& str = "") const;
    RIN std::string Join(const char str) const;
    RIN std::string Join(const char* str) const;
};


template<typename T>
RIN T ValPrimXVectorAPI::GetCommonItems(char const* item)
{
    xint size = strlen(item);
    xvector<T> c_vec(size);

    for (int i = 0; i < size; i++)
        c_vec << item[i];

    return this->GetCommonItems(c_vec);
}

template<typename T>
template<typename S>
RIN S ValPrimXVectorAPI::Join(const S& str) const
{
    std::ostringstream ostr;
    for (const auto& i : *this) {
        ostr << i;
        ostr << str;
    }
    return S(ostr.str().substr(0, ostr.str().size() - str.size()));
}

template<typename T>
RIN std::string ValPrimXVectorAPI::Join(const char str) const
{
    std::string val;
    val.insert(val.begin(), str);
    return this->Join(val);
}

template<typename T>
RIN std::string ValPrimXVectorAPI::Join(const char* str) const
{
    return this->Join(std::string(str));
}

