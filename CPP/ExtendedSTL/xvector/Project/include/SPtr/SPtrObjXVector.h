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
#include "BaseSPtrXVector.h"
#include <type_traits>

template<typename T, typename enabler_t> class xvector;
template<typename T> class SPtrXVector;
class xstring;

#define ForAll(__VECTOR__) for(xint i = 0; i < __VECTOR__.Size(); i++)

template<typename T>
class SPtrObjXVectorAPI : public SPtrXVector<xp<T>>
{
public:
    using SPtrXVector<xp<T>>::SPtrXVector;
    using SPtrXVector<xp<T>>::operator=;
    using E = typename std::remove_const<xp<T>>::type; // E for Erratic

    RIN T Expand() const;

    RIN T Join(const T& str = "") const;
    RIN T Join(const char str) const;
    RIN T Join(const char* str) const;
};



template<typename T>
RIN T SPtrObjXVectorAPI::Expand() const
{   // go from xvector<xvector<xstring>> to xvector<xstring>
    T expanded_vec;
    for (typename xvector<xp<T>>::const_iterator LoDoubleVec = this->begin(); LoDoubleVec != this->end(); LoDoubleVec++) {
        for (typename T::const_iterator LoSingleVec = (*LoDoubleVec)->begin(); LoSingleVec != (*LoDoubleVec)->end(); LoSingleVec++)
            expanded_vec << *LoSingleVec;
    }
    return expanded_vec;
}

template<typename T>
RIN T SPtrObjXVectorAPI::Join(const T& str) const
{
    T ret;
    for (typename xvector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += (*it).Get() + str;

    xint Diff = ret.size() - str.size();
    if (Diff > 0)
        return ret.substr(0, Diff);
    return ret;
}

template<typename T>
RIN T SPtrObjXVectorAPI::Join(const char str) const
{
    T ret;
    for (typename xvector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += (*it).Get() + str;

    if (ret.size() > 1)
        return ret.substr(0, ret.size() - 1);
    return ret;
}

template<typename T>
RIN T SPtrObjXVectorAPI::Join(const char* str) const
{
    T ret;
    for (typename xvector<xp<T>>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += (*it).Get() + str;

    long long int Diff = ret.size() - strlen(str);
    if(Diff > 0)
        return ret.substr(0, Diff);
    return ret;
}

