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
class xvector<T*, typename std::enable_if_t<!std::is_class<std::remove_pointer_t<T*>>::value>> : public ptr_xvector<T*>
{
private:
    typedef typename std::remove_const<T>::type E;// E for Erratic

public:
    using ptr_xvector<T*>::ptr_xvector;
    using ptr_xvector<T*>::operator=;

    typedef T value_type;
    inline xvector(std::initializer_list<T*> lst): ptr_xvector<T*>(std::move(lst)) { };
    inline xvector(const std::vector<T*>& vec) : ptr_xvector<T*>(vec) { };
    inline xvector(std::vector<T*>&& vec) noexcept : ptr_xvector<T*>(std::move(vec)) { };
    inline xvector(const xvector<T*>& vec) : ptr_xvector<T*>(vec) { };
    inline xvector(xvector<T*>&& vec) noexcept : ptr_xvector<T*>(std::move(vec)) { };

    inline void operator=(const xvector<T*>& vec) { ptr_xvector<T*>::operator=(vec); };
    inline void operator=(const std::vector<T*>& vec) { ptr_xvector<T*>::operator=(vec); };

    inline void operator=(xvector<T*>&& vec) { ptr_xvector<T*>::operator=(std::move(vec)); };
    inline void operator=(std::vector<T*>&& vec) { ptr_xvector<T*>::operator=(std::move(vec)); };

    template<typename S>
    inline auto Join(const S& str)->std::enable_if_t<!std::is_same_v<S, char>, S>;
    inline std::string Join(const char str) const;
    inline std::string Join(const char* str) const;
};


template<typename T>
template<typename S>
inline auto xvector<T*, typename std::enable_if_t<!std::is_class<std::remove_pointer_t<T*>>::value>>::Join(const S& str)
    ->std::enable_if_t<!std::is_same_v<S, char>, S>
{
    std::string ret;
    for (typename xvector<T*>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += std::to_string(**it) + str.c_str();
    return S(ret.substr(0, ret.size() - str.size()));
}

template<typename T>
inline std::string xvector<T*, typename std::enable_if_t<!std::is_class<std::remove_pointer_t<T*>>::value>>::Join(const char str) const
{
    std::ostringstream ostr;
    for (typename xvector<T*>::const_iterator it = this->begin(); it != this->end(); it++) 
    {
        ostr << *it;
        ostr << str;
    }
    std::string retstr(ostr.str().c_str());
    return retstr.substr(0, retstr.size() - 1);
}

template<typename T>
inline std::string xvector<T*, typename std::enable_if_t<!std::is_class<std::remove_pointer_t<T*>>::value>>::Join(const char* str) const
{
    std::string retstr;
    for (typename xvector<T*>::const_iterator it = this->begin(); it != this->end(); it++)
        retstr += std::to_string(**it) + str;

    return retstr.substr(0, retstr.size() - strlen(str));
}

