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

#ifndef The
#define The (*this)
#endif // !The

#ifndef _xint_
#define _xint_
using xint = size_t;
// in ASM EAX =    Extended Register 32bit
//        RAX = Re-Extended Register 64bit
// xint = 32bit/64bit (depending on the compiler)
#endif

namespace RA
{
    template<typename T>
    class Iterator
    {
    private:
        T* m_array = nullptr;
        size_t* m_size = nullptr;

        class BaseData
        {
        public:
            BaseData(T* ptr) : m_ptr(ptr) { }
            T* m_ptr = nullptr;
        };

        class BaseComparisons : virtual public BaseData
        {
        public:
            BaseComparisons(T* ptr) : BaseData(ptr) { }

            template<typename I> bool operator==(const I& other) const { return The.m_ptr == other.m_ptr; }
            template<typename I> bool operator!=(const I& other) const { return The.m_ptr != other.m_ptr; }
            template<typename I> bool operator> (const I& other) const { return The.m_ptr >  other.m_ptr; }
            template<typename I> bool operator< (const I& other) const { return The.m_ptr <  other.m_ptr; }
            template<typename I> bool operator>=(const I& other) const { return The.m_ptr >= other.m_ptr; }
            template<typename I> bool operator<=(const I& other) const { return The.m_ptr <= other.m_ptr; }
        };

        class ConstGetters : virtual public BaseData
        {
        public:
            ConstGetters(T* ptr) : BaseData(ptr) { }
            const T& Get() const { return *The.m_ptr; }

            const T* operator+ (size_t num) const { return (The.m_ptr + num); }
            const T* operator- (size_t num) const { return (The.m_ptr - num); }
            const T& operator*()  const { return *The.m_ptr; }
            const T* operator->() const { return  The.m_ptr; }
        };

        class NonConstGetters: virtual public BaseData
        {
        public:
            NonConstGetters(T* ptr) : BaseData(ptr) { }
            T& Get() { return *The.m_ptr; }

            T* operator+ (size_t num) { return (The.m_ptr + num); }
            T* operator- (size_t num) { return (The.m_ptr - num); }
            T& operator*()  { return *The.m_ptr; }
            T* operator->() { return  The.m_ptr; }
        };

    public:
        class iterator : virtual public BaseComparisons, virtual public NonConstGetters
        {
        public:
            iterator(T* ptr) : BaseComparisons(ptr), NonConstGetters(ptr), BaseData(ptr) { }
            T& Get() { return *The.m_ptr; }
            iterator& operator+=(size_t num) { The.m_ptr += num; return The; }
            iterator& operator-=(size_t num) { The.m_ptr -= num; return The; }
            iterator& operator++() { The.m_ptr++; return The; }
            iterator& operator--() { The.m_ptr--; return The; }
            iterator& operator++(int) { decltype(The) New(The.m_ptr); The.m_ptr++;  return New; }
            iterator& operator--(int) { decltype(The) New(The.m_ptr); The.m_ptr--;  return New; }
        };

        class const_iterator : virtual public BaseComparisons, virtual public ConstGetters
        {
        public:
            const_iterator(T* ptr) : BaseComparisons(ptr), ConstGetters(ptr), BaseData(ptr) {}
            const_iterator& operator+=(size_t num) { The.m_ptr += num; return The; }
            const_iterator& operator-=(size_t num) { The.m_ptr -= num; return The; }
            const_iterator& operator++() { The.m_ptr++; return The; }
            const_iterator& operator--() { The.m_ptr--; return The; }
            const_iterator& operator++(int) { decltype(The) New(The.m_ptr); The.m_ptr++;  return New; }
            const_iterator& operator--(int) { decltype(The) New(The.m_ptr); The.m_ptr--;  return New; }
        };

        class reverse_iterator : virtual public BaseComparisons, virtual public NonConstGetters
        {
        public:
            reverse_iterator(T* ptr) : BaseComparisons(ptr), NonConstGetters(ptr), BaseData(ptr) { }
            T& Get() { return *The.m_ptr; }
            reverse_iterator& operator+=(size_t num) { The.m_ptr -= num; return The; }
            reverse_iterator& operator-=(size_t num) { The.m_ptr += num; return The; }
            reverse_iterator& operator++() { The.m_ptr--; return The; }
            reverse_iterator& operator--() { The.m_ptr++; return The; }
            reverse_iterator& operator++(int) { decltype(The) New(The.m_ptr); The.m_ptr++;  return New; }
            reverse_iterator& operator--(int) { decltype(The) New(The.m_ptr); The.m_ptr--;  return New; }
        };

        class const_reverse_iterator : virtual public BaseComparisons, virtual public ConstGetters
        {
        public:
            const_reverse_iterator(T* ptr) : BaseComparisons(ptr), ConstGetters(ptr), BaseData(ptr) {}
            const_reverse_iterator& operator+=(size_t num) { The.m_ptr -= num; return The; }
            const_reverse_iterator& operator-=(size_t num) { The.m_ptr += num; return The; }
            const_reverse_iterator& operator++() { The.m_ptr--; return The; }
            const_reverse_iterator& operator--() { The.m_ptr++; return The; }
            const_reverse_iterator& operator++(int) { decltype(The) New(The.m_ptr); The.m_ptr++;  return New; }
            const_reverse_iterator& operator--(int) { decltype(The) New(The.m_ptr); The.m_ptr--;  return New; }
        };

        // --------------------------------------------------------------------------------------------
        iterator begin() { return iterator(The.m_array); }
        iterator end()   { return iterator(The.m_array + *m_size); }
        const_iterator cbegin() const { return const_iterator(The.m_array); }
        const_iterator cend()   const { return const_iterator(The.m_array + *The.m_size); }
        const_iterator begin()  const { return const_iterator(The.m_array); }
        const_iterator end()    const { return const_iterator(The.m_array + *The.m_size); }
        // --------------------------------------------------------------------------------------------
        reverse_iterator rbegin() { return reverse_iterator(The.m_array + *The.m_size - 1); }
        reverse_iterator rend() { return reverse_iterator(The.m_array - 1); }
        const_reverse_iterator crbegin() const { return const_reverse_iterator(The.m_array + *The.m_size - 1); }
        const_reverse_iterator crend()   const { return const_reverse_iterator(The.m_array - 1); }
        const_reverse_iterator rbegin()  const { return const_reverse_iterator(The.m_array + *The.m_size - 1); }
        const_reverse_iterator rend()    const { return const_reverse_iterator(The.m_array - 1); }
        // --------------------------------------------------------------------------------------------
    public:
        Iterator() {}
        Iterator(T* FvArray, size_t* FnSize) { SetIterator(FvArray, FnSize); }
        void SetIterator(T* FvArray, size_t* FnSize) { SetIteratorArray(FvArray); SetIteratorSize(FnSize); }
        void SetIteratorArray(T* FvArray) { m_array = FvArray; }
        void SetIteratorSize(size_t* FnSize) { m_size = FnSize; }
        // --------------------------------------------------------------------------------------------
        size_t distance(iterator start, iterator end) {
            size_t size = 0;
            for (; start < end; ++start)
                ++size;
            return size;
        }
        // ============================================================================================================
    };
};