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

#include "XvectorTypes.h"
#include "BaseValXVector.h"
#include <type_traits>

template<typename T>
class ValObjXVectorAPI : public ValXVector<T>
{
public:
    using ValXVector<T>::ValXVector;
    using ValXVector<T>::operator=;
    using E = typename std::remove_const<T>::type; // E for Erratic

    RIN T Expand() const;

    RIN T Join(const T& str = "") const;
    RIN T Join(const char str) const;
    RIN T Join(const char* str) const;
};



template<typename T>
RIN T ValObjXVectorAPI::Expand() const
{   // go from xvector<xvector<xstring>> to xvector<xstring>
    E expanded_vec;
    for (typename xvector<T>::const_iterator double_vec = this->begin(); double_vec != this->end(); double_vec++) {
        for (typename T::const_iterator single_vec = double_vec->begin(); single_vec != double_vec->end(); single_vec++)
            expanded_vec << *single_vec;
    }
    return expanded_vec;
}

template<typename T>
RIN T ValObjXVectorAPI::Join(const T& str) const
{
    E ret;
    for (typename xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += *it + str;

    xint Diff = ret.size() - str.size();
    if (Diff > 0)
        return ret.substr(0, Diff);
    return ret;
}

template<typename T>
RIN T ValObjXVectorAPI::Join(const char str) const
{
    E ret;
    for (typename xvector<E>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += *it + str;

    if (ret.size() > 1)
        return ret.substr(0, ret.size() - 1);
    return ret;
}

template<typename T>
RIN T ValObjXVectorAPI::Join(const char* str) const
{
    E ret;
    for (typename xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += *it + str;

    long long int Diff = ret.size() - strlen(str);
    if(Diff > 0)
        return ret.substr(0, Diff);
    return ret;
}

