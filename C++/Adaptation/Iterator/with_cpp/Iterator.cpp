#include "Iterator.h"


/*
* Copyright[2018][Joel Leagues aka Scourge]
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

    // The Iterator is an adaptation from the following

    // cs.helsinki.fi/u/tpkarkka/alglib/k06/lectures/Iterators.html
    // and perfectly insane's structure


// -----------------------------------------------------------------
template<typename T>
typename Iterator<T>::iterator Iterator<T>::begin(){
    return iterator(*m_array);
}
template<typename T>
typename Iterator<T>::iterator Iterator<T>::end(){
    return iterator(*m_array + *m_size);
}
template<typename T>
typename Iterator<T>::const_iterator Iterator<T>::cbegin() const{
    return const_iterator(*m_array);
}
template<typename T>
typename Iterator<T>::const_iterator Iterator<T>::cend() const{
    return const_iterator(*m_array + *m_size);
}
template<typename T>
typename Iterator<T>::const_iterator Iterator<T>::begin() const{
    return const_iterator(*m_array);
}
template<typename T>
typename Iterator<T>::const_iterator Iterator<T>::end() const{
    return const_iterator(*m_array + *m_size);
}
// -----------------------------------------------------------------
template<typename T>
typename Iterator<T>::reverse_iterator Iterator<T>::rbegin(){
    return reverse_iterator(*m_array + *m_size-1);
}
template<typename T>
typename Iterator<T>::reverse_iterator Iterator<T>::rend(){
    return reverse_iterator(*m_array-1);
}
template<typename T>
typename Iterator<T>::const_reverse_iterator Iterator<T>::crbegin() const{
    return const_reverse_iterator(*m_array + *m_size-1);
}
template<typename T>
typename Iterator<T>::const_reverse_iterator Iterator<T>::crend() const{
    return const_reverse_iterator(*m_array-1);
}
template<typename T>
typename Iterator<T>::const_reverse_iterator Iterator<T>::rbegin() const{
    return const_reverse_iterator(*m_array + *m_size-1);
}
template<typename T>
typename Iterator<T>::const_reverse_iterator Iterator<T>::rend() const{
    return const_reverse_iterator(*m_array-1);
}
// -----------------------------------------------------------------


