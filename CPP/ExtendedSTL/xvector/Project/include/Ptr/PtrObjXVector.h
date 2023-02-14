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

#include "BasePtrXVector.h"
#include <type_traits>

template<typename T> class PtrXVector;
template<typename T, typename enabler_t> class xvector;

template<typename T>
class PtrObjXVectorAPI : public PtrXVector<T*>
{
public:
    using PtrXVector<T*>::PtrXVector;
    using PtrXVector<T*>::operator=;
    using E = typename std::remove_const<T>::type; // E for Erratic

    typedef T value_type;

    virtual ~xvector();

    void DeleteAll();

    T Join(const T& str = "") const;
    T Join(const char str) const;
    T Join(const char* str) const;
};

template<typename T>
inline PtrObjXVectorAPI::~xvector()
{
    if (The.MbDeleteAllOnExit)
        DeleteAll();
}

template<typename T>
inline void PtrObjXVectorAPI::DeleteAll()
{
    for (int i = 0; i < The.Size(); i++)
    {
        T* PtrVal = The.RawPtr(i);
        if (!PtrVal)
            continue;
        delete PtrVal;
        PtrVal = nullptr;
    }
}

template<typename T>
inline T PtrObjXVectorAPI::Join(const T& str) const
{
    E ret;
    for (typename xvector<T*>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += **it + str;

    size_t Diff = ret.size() - str.size();
    if (Diff > 0)
        return ret.substr(0, Diff);
    return ret;
}

template<typename T>
inline T PtrObjXVectorAPI::Join(const char str) const
{
    E ret;
    for (typename xvector<T*>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += **it + str;

    if (ret.size() > 1)
        return ret.substr(0, ret.size() - 1);
    return ret;
}

template<typename T>
inline T PtrObjXVectorAPI::Join(const char* str) const
{
    E ret;
    for (typename xvector<T*>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += **it + str;

    long long int Diff = ret.size() - strlen(str);
    if (Diff > 0)
        return ret.substr(0, Diff);
    return ret;
}
