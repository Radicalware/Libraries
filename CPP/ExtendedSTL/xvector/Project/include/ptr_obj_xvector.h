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

#include "base_ptr_xvector.h"
#include <type_traits>

template<typename T> class ptr_xvector;
template<typename T, typename enabler_t> class xvector;

template<typename T>
class xvector<T*, typename std::enable_if_t<std::is_class<std::remove_pointer_t<T*>>::value>> : public ptr_xvector<T*>
{
private:
    typedef typename std::remove_const<T>::type E;// E for Erratic

public:
    using ptr_xvector<T*>::ptr_xvector;
    using ptr_xvector<T*>::operator=;

    typedef T value_type;
    inline xvector() = default;
    inline xvector(std::initializer_list<T*> lst): ptr_xvector<T*>(std::move(lst)) { };
    inline xvector(const std::vector<T*>& vec) : ptr_xvector<T*>(vec) { };
    inline xvector(std::vector<T*>&& vec) noexcept : ptr_xvector<T*>(std::move(vec)) { };
    inline xvector(const xvector<T*>& vec) : ptr_xvector<T*>(vec) { };
    inline xvector(xvector<T*>&& vec) noexcept : ptr_xvector<T*>(std::move(vec)) { };

    inline void operator=(const xvector<T*>& vec) { ptr_xvector<T*>::operator=(vec); };
    inline void operator=(const std::vector<T*>& vec) { ptr_xvector<T*>::operator=(vec); };

    inline void operator=(xvector<T*>&& vec) { ptr_xvector<T*>::operator=(std::move(vec)); };
    inline void operator=(std::vector<T*>&& vec) { ptr_xvector<T*>::operator=(std::move(vec)); };

    // Go from xvector<xvector<N>*> to xvector<N*>
    //         xvector<T*>          to xvector<N*>
    template<typename N = typename E::value_type> // Nested Type
    inline xvector<N*> Expand() const;

    T Join(const T& str = "") const;
    T Join(const char str) const;
    T Join(const char* str) const;
};


template<typename T>
template<typename N /* = typename E::value_type*/>
inline xvector<N*> xvector<T*, typename std::enable_if_t<std::is_class<std::remove_pointer_t<T*>>::value>>::Expand() const
{
    // Go from xvector<xvector<N>*> to xvector<N*>
    xvector<N*> expanded_vec;
    for (typename xvector<E*>::const_iterator double_vec = this->begin(); double_vec != this->end(); double_vec++) {
        for (typename xvector<N>::iterator single_vec = (*double_vec)->begin(); single_vec != (*double_vec)->end(); single_vec++)
            expanded_vec.push_back(&*single_vec);
    }
    return expanded_vec;
}

template<typename T>
inline T xvector<T*, typename std::enable_if_t<std::is_class<std::remove_pointer_t<T*>>::value>>::Join(const T& str) const
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
inline T xvector<T*, typename std::enable_if_t<std::is_class<std::remove_pointer_t<T*>>::value>>::Join(const char str) const
{
    E ret;
    for (typename xvector<T*>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += **it + str;

    if (ret.size() > 1)
        return ret.substr(0, ret.size() - 1);
    return ret;
}

template<typename T>
inline T xvector<T*, typename std::enable_if_t<std::is_class<std::remove_pointer_t<T*>>::value>>::Join(const char* str) const
{
    E ret;
    for (typename xvector<T*>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += **it + str;

    long long int Diff = ret.size() - strlen(str);
    if (Diff > 0)
        return ret.substr(0, Diff);
    return ret;
}
