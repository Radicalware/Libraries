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

    // The iterator is an adaptation from the following content

    // cs.helsinki.fi/u/tpkarkka/alglib/k06/lectures/iterators.html
    // dreamincode.net/forums/topic/58468-making-your-own-iterators


using size_t = unsigned long int;

template<typename T>
class Iterator
{
// ============================================================================================================

private:
    T**     m_array;      
    size_t* m_size  = 0; 
public:

    class iterator{
        private:
            T* m_ptr;
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
            T&       operator*()  { return *m_ptr; }
            T*       operator->() { return  m_ptr; }
    };

    class const_iterator{
        private:
            T* m_ptr;
        public:
            const_iterator(T* ptr) : m_ptr(ptr) { }
            const_iterator operator+=(size_t num) { m_ptr += num; return *this; }
            const_iterator operator-=(size_t num) { m_ptr -= num; return *this; }
            const T*       operator+ (size_t num) { return (m_ptr + num); }
            const T*       operator- (size_t num) { return (m_ptr - num); }
            const_iterator operator++(){ m_ptr++; return *this; }
            const_iterator operator--(){ m_ptr--; return *this; }
            bool           operator==(const const_iterator& other) { return m_ptr == other.m_ptr; }
            bool           operator!=(const const_iterator& other) { return m_ptr != other.m_ptr; }
            const T&       operator*() { return *m_ptr; }
            const T*       operator->(){ return  m_ptr; }
    };

    class reverse_iterator{
        private:
            T* m_ptr;
        public:
            reverse_iterator(T* ptr) : m_ptr(ptr) { }
            reverse_iterator operator+=(size_t num) { m_ptr -= num; return *this; }
            reverse_iterator operator-=(size_t num) { m_ptr += num; return *this; }
            T*               operator+ (size_t num) { return (m_ptr - num); }
            T*               operator- (size_t num) { return (m_ptr + num); }
            reverse_iterator operator++(){ *this; m_ptr--; return *this; }
            reverse_iterator operator--(){ *this; m_ptr++; return *this; }
            bool             operator==(const reverse_iterator& other) { return m_ptr == other.m_ptr; }
            bool             operator!=(const reverse_iterator& other) { return m_ptr != other.m_ptr; }
            T&               operator*() { return *m_ptr; }
            T*               operator->(){ return  m_ptr; }
    };

    class const_reverse_iterator{
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
            const T&       operator*() { return *m_ptr; }
            const T*       operator->(){ return  m_ptr; }
    };

    // --------------------------------------------------------------------------------------------
    iterator begin(){ return iterator(*m_array); }
    iterator end()  { return iterator(*m_array + *m_size); }
    const_iterator cbegin() const{ return const_iterator(*m_array); }
    const_iterator cend()   const{ return const_iterator(*m_array + *m_size); }
    const_iterator begin()  const{ return const_iterator(*m_array); }
    const_iterator end()    const{ return const_iterator(*m_array + *m_size); }
    // --------------------------------------------------------------------------------------------
    reverse_iterator rbegin(){ return reverse_iterator(*m_array + *m_size-1); }
    reverse_iterator rend()  { return reverse_iterator(*m_array-1); }
    const_reverse_iterator crbegin() const{ return const_reverse_iterator(*m_array + *m_size-1); }
    const_reverse_iterator crend()   const{ return const_reverse_iterator(*m_array-1);}
    const_reverse_iterator rbegin()  const{ return const_reverse_iterator(*m_array + *m_size-1); }
    const_reverse_iterator rend()    const{ return const_reverse_iterator(*m_array-1); }
    // --------------------------------------------------------------------------------------------
    void set_array(T** array)  { m_array = array; }
    void set_size(size_t* size){ m_size  = size;  }
    // --------------------------------------------------------------------------------------------

// ============================================================================================================
};

#endif


