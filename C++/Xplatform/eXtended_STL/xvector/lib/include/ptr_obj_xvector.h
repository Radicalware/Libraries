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

template<typename T> class ptr_xvector;
template<typename T, typename enabler_t> class xvector;

template<typename T>
class xvector<T*, typename std::enable_if<std::is_class<T*>::value>::type> : public ptr_xvector<T*>
{
private:
    using ptr_xvector<T*>::ptr_xvector;
    
public:
    // Go from xvector<xvector<N>> to xvector<N>
    template<typename N> // Nested Type
    inline xvector<N> expand() const;
};


template<typename T>
template<typename N>
inline xvector<N> xvector<T*, typename std::enable_if<std::is_class<T*>::value>::type>::expand() const
{
    // Go from xvector<xvector<N>> to xvector<N>
    xvector<N> expanded_vec;
    for (typename xvector<xvector<N>*>::const_iterator outer = this->begin(); outer != this->end(); outer++) {
        for (typename xvector<N>::const_iterator inner = (*outer)->begin(); inner != (*outer)->end(); inner++)
            expanded_vec << *inner;
    }
    return expanded_vec;
}
