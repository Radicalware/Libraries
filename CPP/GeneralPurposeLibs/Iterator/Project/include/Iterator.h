#pragma once

/*
*|| Copyright[2018][Joel Leagues aka Scourge]
*|| Scourge /at\ protonmail /dot\ com
*|| www.Radicalware.net
*|| https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*||
*|| Licensed under the Apache License, Version 2.0 (the "License");
*|| you may not use this file except in compliance with the License.
*|| You may obtain a copy of the License at
*||
*|| http ://www.apache.org/licenses/LICENSE-2.0
*||
*|| Unless required by applicable law or agreed to in writing, software
*|| distributed under the License is distributed on an "AS IS" BASIS,
*|| WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*|| See the License for the specific language governing permissions and
*|| limitations under the License.
*/

//| A reference for how an iterator works can be found at this link
//| http://www.cplusplus.com/reference/iterator/iterator
//| I added a lot of functionality such as
//| removing un-needed alias that were even within the brackets!!
//| Adding reverse iterators
//| Adding MANY operator overloaded functions including: 
//| --, -, +, -=, +=, >, <, >=, <=,
//| Mine is made in the usable for of a class object
//| I added the functions: distance(), cbegin(), cend(), rbegin(), rend()
//|                        crbegin(), crend(), set_array(), set_size()
//|

template<typename T>
class Iterator
{
    // ============================================================================================================
private:
    T**     m_array;
    size_t* m_size;

public:
    class iterator {
    private:
        T* m_ptr; // address of m_array
    public:
        iterator(T* ptr) : m_ptr(ptr) { }
        iterator operator+=(size_t num) { m_ptr += num; return *this; }
        iterator operator-=(size_t num) { m_ptr -= num; return *this; }
        T*       operator+ (size_t num) { return (m_ptr + num); }
        T*       operator- (size_t num) { return (m_ptr - num); }
        iterator operator++() { m_ptr++; return *this; }
        iterator operator--() { m_ptr--; return *this; }
        bool     operator==(const iterator& other) { return m_ptr == other.m_ptr; }
        bool     operator!=(const iterator& other) { return m_ptr != other.m_ptr; }
        bool     operator>(const iterator& other) { return m_ptr > other.m_ptr; }
        bool     operator<(const iterator& other) { return m_ptr < other.m_ptr; }
        bool     operator>=(const iterator& other) { return m_ptr >= other.m_ptr; }
        bool     operator<=(const iterator& other) { return m_ptr <= other.m_ptr; }
        T&       operator*() { return *m_ptr; }
        T*       operator->() { return  m_ptr; }
    };

    class const_iterator {
    private:
        T* m_ptr;
    public:
        const_iterator(T* ptr) : m_ptr(ptr) { }
        const_iterator operator+=(size_t num) { m_ptr += num; return *this; }
        const_iterator operator-=(size_t num) { m_ptr -= num; return *this; }
        const T*       operator+ (size_t num) { return (m_ptr + num); }
        const T*       operator- (size_t num) { return (m_ptr - num); }
        const_iterator operator++() { m_ptr++; return *this; }
        const_iterator operator--() { m_ptr--; return *this; }
        bool           operator==(const const_iterator& other) { return m_ptr == other.m_ptr; }
        bool           operator!=(const const_iterator& other) { return m_ptr != other.m_ptr; }
        bool           operator> (const const_iterator& other) { return m_ptr > other.m_ptr; }
        bool           operator< (const const_iterator& other) { return m_ptr < other.m_ptr; }
        bool           operator>=(const const_iterator& other) { return m_ptr >= other.m_ptr; }
        bool           operator<=(const const_iterator& other) { return m_ptr <= other.m_ptr; }
        const T&       operator*() { return *m_ptr; }
        const T*       operator->() { return  m_ptr; }
    };

    class reverse_iterator {
    private:
        T* m_ptr;
    public:
        reverse_iterator(T* ptr) : m_ptr(ptr) { }
        reverse_iterator operator+=(size_t num) { m_ptr -= num; return *this; }
        reverse_iterator operator-=(size_t num) { m_ptr += num; return *this; }
        T*               operator+ (size_t num) { return (m_ptr - num); }
        T*               operator- (size_t num) { return (m_ptr + num); }
        reverse_iterator operator++() { *this; m_ptr--; return *this; }
        reverse_iterator operator--() { *this; m_ptr++; return *this; }
        bool             operator==(const reverse_iterator& other) { return m_ptr == other.m_ptr; }
        bool             operator!=(const reverse_iterator& other) { return m_ptr != other.m_ptr; }
        bool             operator> (const reverse_iterator& other) { return m_ptr < other.m_ptr; }
        bool             operator< (const reverse_iterator& other) { return m_ptr > other.m_ptr; }
        bool             operator>=(const reverse_iterator& other) { return m_ptr <= other.m_ptr; }
        bool             operator<=(const reverse_iterator& other) { return m_ptr >= other.m_ptr; }
        T&               operator*() { return *m_ptr; }
        T*               operator->() { return  m_ptr; }
    };

    class const_reverse_iterator {
    private:
        T* m_ptr;
    public:
        const_reverse_iterator(T* ptr) : m_ptr(ptr) { }
        const_reverse_iterator operator+=(size_t num) { m_ptr -= num; return *this; }
        const_reverse_iterator operator-=(size_t num) { m_ptr += num; return *this; }
        const T*               operator+ (size_t num) { return (m_ptr - num); }
        const T*               operator- (size_t num) { return (m_ptr + num); }
        const_reverse_iterator operator++() { m_ptr--; return *this; }
        const_reverse_iterator operator--() { m_ptr++; return *this; }
        bool           operator==(const const_reverse_iterator& other) { return m_ptr == other.m_ptr; }
        bool           operator!=(const const_reverse_iterator& other) { return m_ptr != other.m_ptr; }
        bool           operator> (const const_reverse_iterator& other) { return m_ptr < other.m_ptr; }
        bool           operator< (const const_reverse_iterator& other) { return m_ptr > other.m_ptr; }
        bool           operator>=(const const_reverse_iterator& other) { return m_ptr <= other.m_ptr; }
        bool           operator<=(const const_reverse_iterator& other) { return m_ptr >= other.m_ptr; }
        const T&       operator*() { return *m_ptr; }
        const T*       operator->() { return  m_ptr; }
    };

    // --------------------------------------------------------------------------------------------
    iterator begin() { return iterator(*m_array); }
    iterator end() { return iterator(*m_array + *m_size); }
    const_iterator cbegin() const { return const_iterator(*m_array); }
    const_iterator cend()   const { return const_iterator(*m_array + *m_size); }
    const_iterator begin()  const { return const_iterator(*m_array); }
    const_iterator end()    const { return const_iterator(*m_array + *m_size); }
    // --------------------------------------------------------------------------------------------
    reverse_iterator rbegin() { return reverse_iterator(*m_array + *m_size - 1); }
    reverse_iterator rend() { return reverse_iterator(*m_array - 1); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(*m_array + *m_size - 1); }
    const_reverse_iterator crend()   const { return const_reverse_iterator(*m_array - 1); }
    const_reverse_iterator rbegin()  const { return const_reverse_iterator(*m_array + *m_size - 1); }
    const_reverse_iterator rend()    const { return const_reverse_iterator(*m_array - 1); }
    // --------------------------------------------------------------------------------------------
    void set_array(T** array) { m_array = array; }
    void set_size(size_t* size) { m_size = size; }
    // --------------------------------------------------------------------------------------------
    size_t distance(iterator start, iterator end) {
        size_t size = 0;
        for (; start < end; ++start)
            ++size;
        return size;
    }
    // ============================================================================================================
};



