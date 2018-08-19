#ifndef _Iterator_H_
#define _Iterator_H_


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

    // The iterator is an adaptation from the following

    // cs.helsinki.fi/u/tpkarkka/alglib/k06/lectures/iterators.html
    // used by perfectly insane's structure that I then modified.


using size_t = unsigned long int;

template<typename T>
class Iterator
{
// ============================================================================================================

private:
    T**     m_array;      // the first pointer just de-references.
    size_t* m_size  = 0; 
public:

    class iterator{
        private:
            T* m_ptr;
        public:
            iterator(T* ptr) : m_ptr(ptr) { }
            iterator operator++() { iterator i = *this; m_ptr++; return i; }
            bool     operator==(const iterator& other) { return m_ptr == other.m_ptr; }
            bool     operator!=(const iterator& other) { return m_ptr != other.m_ptr; }
            T&       operator*()  { return *m_ptr; }
            T*       operator->() { return  m_ptr; }
    };

    class const_iterator{
        private:
            T* m_ptr;
        public:
            const_iterator(T* ptr) : m_ptr(ptr) { }
            const_iterator operator++() { const_iterator i = *this; m_ptr++; return i; }
            bool           operator==(const const_iterator& other) { return m_ptr == other.m_ptr; }
            bool           operator!=(const const_iterator& other) { return m_ptr != other.m_ptr; }
            const T&       operator*()  { return *m_ptr; }
            const T*       operator->() { return  m_ptr; }
    };

    class reverse_iterator{
        private:
            T* m_ptr;
        public:
            reverse_iterator(T* ptr) : m_ptr(ptr) { }
            reverse_iterator operator++() { reverse_iterator i = *this; m_ptr--; return i; }
            bool           operator==(const reverse_iterator& other) { return m_ptr == other.m_ptr; }
            bool           operator!=(const reverse_iterator& other) { return m_ptr != other.m_ptr; }
            const T&       operator*()  { return *m_ptr; }
            const T*       operator->() { return  m_ptr; }
    };

    class const_reverse_iterator{
        private:
            T* m_ptr;
        public:
            const_reverse_iterator(T* ptr) : m_ptr(ptr) { }
            const_reverse_iterator operator++() { const_reverse_iterator i = *this; m_ptr--; return i; }
            bool           operator==(const const_reverse_iterator& other) { return m_ptr == other.m_ptr; }
            bool           operator!=(const const_reverse_iterator& other) { return m_ptr != other.m_ptr; }
            const T&       operator*()  { return *m_ptr; }
            const T*       operator->() { return  m_ptr; }
    };

    // ---------------------------------------------
    iterator begin();
    iterator end();
    const_iterator cbegin() const;
    const_iterator cend() const;
    const_iterator begin() const;
    const_iterator end() const;
    // ---------------------------------------------
    reverse_iterator rbegin();
    reverse_iterator rend();
    const_reverse_iterator crbegin() const;
    const_reverse_iterator crend() const;
    const_reverse_iterator rbegin() const;
    const_reverse_iterator rend() const;
    // ---------------------------------------------

    void set_array(T** array){m_array = array;}
    void set_size(size_t* size){m_size = size;}

// ============================================================================================================
};


#include<string>
#include<vector>

#include<map>
#include<unordered_map>

#include<set>
#include<unordered_set>

#include<stack>
#include<queue>
#include<deque>


template class Iterator<char>;
template class Iterator<char*>;

template class Iterator<int>;
template class Iterator<short>;
template class Iterator<unsigned short>;
template class Iterator<long>;
template class Iterator<unsigned long int>;
template class Iterator<signed long long>;
template class Iterator<unsigned>;
template class Iterator<unsigned long long>;

template class Iterator<std::string>;
template class Iterator<std::vector<std::string>>;


template class Iterator<std::map<std::string,std::string>>;
template class Iterator<std::unordered_map<std::string,std::string>>;

template class Iterator<std::set<std::string>>;
template class Iterator<std::unordered_set<std::string>>;

template class Iterator<std::stack<std::string>>;
template class Iterator<std::queue<std::string>>;
template class Iterator<std::deque<std::string>>;


#endif


