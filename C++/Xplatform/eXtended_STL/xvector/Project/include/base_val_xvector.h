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

#if (defined(WIN64) || defined(_WIN64) || defined(WIN32) || defined(_WIN32))
    using size64_t = __int64;
#else
    #include <cstdint>
    using size64_t = int64_t;
#endif

#include<vector>
#include<type_traits>
#include<initializer_list>
#include<string>
#include<regex>
#include<sstream>
#include<set>
#include<type_traits>

#include "Nexus.h"
#include "xvector.h"

template<typename T>
class val_xvector : public std::vector<T>
{    
public:
    typedef typename std::remove_const<T>::type E; // E for Erratic
    typedef typename std::remove_const<T>::type EVEC_T;
private:
    Nexus<E>* td = nullptr;
public:
    using std::vector<T, std::allocator<T>>::vector;

    inline val_xvector() {};
    inline ~val_xvector();
    inline val_xvector(std::initializer_list<T> lst): std::vector<T>(std::move(lst)) { };
    inline val_xvector(const std::vector<T>& vec) : std::vector<T>(vec) { };
    inline val_xvector(std::vector<T>&& vec) noexcept : std::vector<T>(std::move(vec)) { };
    inline val_xvector(const xvector<T>& vec) : std::vector<T>(vec) { };
    inline val_xvector(xvector<T>&& vec) noexcept : std::vector<T>(std::move(vec)) { };

    inline void operator=(const xvector<T>& vec);
    inline void operator=(const std::vector<T>& vec);

    inline void operator=(xvector<T>&& vec);
    inline void operator=(std::vector<T>&& vec);

    inline T& At(const size_t Idx);
    inline const T& At(const size_t Idx) const;

    template<typename P = T> 
    inline bool Has(const P* item) const;
    inline bool Has(const T& item) const;
    inline bool Has(T&& item) const;
    inline bool Has(char const* item) const;

    inline bool Lacks(const T& item) const;
    inline bool Lacks(T&& item) const;
    inline bool Lacks(char const* item) const;

    template<typename L = xvector<T>>
    inline xvector<T> GetCommonItems(L& item);
    template<typename L = xvector<T>>
    inline xvector<T> GetCommonItems(L&& item);

    inline void operator<<(const T& item);
    inline void operator<<(const T&& item);

    inline void operator*=(const size_t count);

    inline void Add();
    inline void Add(const T& val); 
    template <typename First, typename... Rest>
    inline void Add(const First& first, const Rest& ... rest);

    inline void AddCharStrings(int strC, char** strV);

    template<typename O>
    inline bool operator>(const O& other) const;
    template<typename O>
    inline bool operator<(const O& other) const;
    template<typename O>
    inline bool operator==(const O& other) const;
    template<typename O>
    inline bool operator!=(const O& other) const;

    inline bool operator>(const size_t value) const;
    inline bool operator<(const size_t value) const;
    inline bool operator==(const size_t value) const;
    inline bool operator!=(const size_t value) const;

    T& Back(size_t value = 0);
    T* GetBackPtr(size_t value = 0);

    inline std::pair<T, T> GetPair() const;

    template<typename C>
    inline xvector<C> Convert() const;

    template<typename C, typename F>
    inline xvector<C> Convert(F function) const;

    template<typename N = unsigned int>
    inline xvector<xvector<T>> Split(N count = 1) const;

    inline void operator+=(const xvector<T>& other);
    inline xvector<T> operator+(const xvector<T>& other) const;

    size_t Size() const;

    inline void Organize();
    inline void RemoveDups();

    template<typename F>
    inline xvector<T> Sort(F func);

    inline xvector<T*> GetPtrs();

    template<typename F, typename... A>
    inline void Proc(F&& function, A&& ...Args);
    template<typename F, typename... A>
    inline void ThreadProc(F&& function, A&& ...Args);

    template<typename N = E, typename F, typename ...A>
    inline xvector<N> ForEach(F&& function, A&& ...Args);
    template<typename K, typename V, typename F, typename ...A>
    inline std::unordered_map<K, V> ForEach(F&& function, A&& ...Args);


    template<typename N = E, typename F, typename... A>
    inline xvector<N> ForEachThread(F&& function, A&& ...Args);
    template<typename K, typename V, typename F, typename ...A>
    inline std::unordered_map<K, V> ForEachThread(F&& function, A&& ...Args);

    template<typename N = E, typename F, typename... A>
    inline void StartTasks(F&& function, A&& ...Args);

    template<typename N = E>
    inline typename std::enable_if<!std::is_same<N, void>::value, xvector<N>>::type GetTasks() const;

    inline bool TasksCompleted() const;

    // =================================== DESIGNED FOR NUMERIC BASED VECTORS ===================================

    inline T Sum(size_t FnSkipIdx = 0) const;
    inline T Mul(size_t FnSkipIdx = 0) const;

    // =================================== DESIGNED FOR STRING  BASED VECTORS ===================================

    inline T Join(const T& str = "") const;
    inline T Join(const char str) const;
    inline T Join(const char* str) const;

    inline bool MatchOne(const T& in_pattern) const;
    inline bool MatchOne(T&& in_pattern) const;
    inline bool MatchOne(char const* in_pattern) const;

    inline bool MatchAll(const T& in_pattern) const;
    inline bool MatchAll(T&& in_pattern) const;
    inline bool MatchAll(char const* in_pattern) const;

    inline bool ScanOne(const T& in_pattern) const;
    inline bool ScanOne(T&& in_pattern) const;
    inline bool ScanOne(char const* in_pattern) const;

    inline bool Scanall(const T& in_pattern) const;
    inline bool Scanall(T&& in_pattern) const;
    inline bool Scanall(char const* in_pattern) const;
    
    inline xvector<T> Take(const T& in_pattern) const;
    inline xvector<T> Take(T&& in_pattern) const;
    inline xvector<T> Take(char const* in_pattern) const;

    inline xvector<T> Remove(const T& in_pattern) const;
    inline xvector<T> Remove(T&& in_pattern) const;
    inline xvector<T> Remove(char const* in_pattern) const;

    inline xvector<T> SubAll(const T& in_pattern, const T& replacement) const;
    inline xvector<T> SubAll(T&& in_pattern, T&& replacement) const;
    inline xvector<T> SubAll(char const* in_pattern, char const* replacement) const;
    
    // double was chose to hold long signed and unsigned values
    inline xvector<T> operator()(long double x = 0, long double y = 0, long double z = 0, const char removal_method = 's') const;
    // s = slice perserves values you land on 
    // d = dice  removes values you land on
    // s/d only makes a difference if you modify the 'z' value

    // =================================== DESIGNED FOR STRING BASED VECTORS ===================================
};
// =============================================================================================================


template<typename T>
template<typename P>
inline bool val_xvector<T>::Has(const P* item) const
{
    for (typename val_xvector<T>::const_iterator it = this->begin(); it != this->end(); it++) {
        if (**it == *item)
            return true;
    }
    return false;
}

template<typename T>
inline val_xvector<T>::~val_xvector()
{
    DeleteNexusPtr(td);
}

template<typename T>
inline void val_xvector<T>::operator=(const xvector<T>& vec)
{
    this->clear();
    this->reserve(vec.size());
    this->insert(this->begin(), vec.begin(), vec.end());
}

template<typename T>
inline void val_xvector<T>::operator=(const std::vector<T>& vec)
{
    this->clear();
    this->reserve(vec.size());
    this->insert(this->begin(), vec.begin(), vec.end());
}

template<typename T>
inline void val_xvector<T>::operator=(xvector<T>&& vec)
{
    this->clear();
    this->reserve(vec.size());
    this->insert(this->begin(), std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end()));
}

template<typename T>
inline void val_xvector<T>::operator=(std::vector<T>&& vec)
{
    this->clear();
    this->reserve(vec.size());
    this->insert(this->begin(), std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end()));
}

template<typename T>
inline T& val_xvector<T>::At(const size_t Idx)
{
    if (Size() >= Idx)
        throw "Index Out Of Range";
    return (*this)[Idx];
}

template<typename T>
inline const T& val_xvector<T>::At(const size_t Idx) const
{
    if (Size() >= Idx)
        throw "Index Out Of Range";
    return (*this)[Idx];
}

template<typename T>
bool val_xvector<T>::Has(const T& item) const {
    return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool val_xvector<T>::Has(T&& item) const {
    return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool val_xvector<T>::Has(char const* item) const {
    return (bool(std::find(this->begin(), this->end(), item) != this->end()));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
bool val_xvector<T>::Lacks(T&& item) const {
    return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool val_xvector<T>::Lacks(const T& item) const {
    return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

template<typename T>
bool val_xvector<T>::Lacks(char const* item) const {
    return !(bool(std::find(this->begin(), this->end(), item) != this->end()));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename L>
xvector<T> val_xvector<T>::GetCommonItems(L& item) 
{
    std::sort(this->begin(), this->end());
    std::sort(item.begin(), item.end());

    xvector<T> vret(this->size() + item.size());
    set_intersection(this->begin(), this->end(), item.begin(), item.end(), vret.begin());
    return vret;
}

template<typename T>
template<typename L>
xvector<T> val_xvector<T>::GetCommonItems(L&& item) {
    return this->GetCommonItems(item);
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline void val_xvector<T>::operator<<(const T& item){
    this->emplace_back(item);
}

template<typename T>
inline void val_xvector<T>::operator<<(const T&& item){
    this->emplace_back(std::move(item));
}

template<typename T>
inline void val_xvector<T>::operator*=(const size_t count)
{
    xvector<E>* tmp = new xvector<E>;
    tmp->reserve(this->size() + 1);
    (*tmp).insert(tmp->begin(), this->begin(), this->end());
    
    this->reserve(this->size() * count + 1);
    for (int i = 0; i < count - 1; i++)
        this->insert(this->end(), tmp->begin(), tmp->end());
    delete tmp;
}

template<typename T>
inline void val_xvector<T>::Add() {}
template<typename T>
inline void val_xvector<T>::Add(const T& val){
    *this << val;
}
template<typename T>
template<typename First, typename ...Rest>
inline void val_xvector<T>::Add(const First& first, const Rest& ...rest)
{
    *this << first;
    this->Add(rest...);
}

template<typename T>
inline void val_xvector<T>::AddCharStrings(int strC, char** strV)
{
    for (int i = 0; i < strC; i++)
        this->push_back(T(strV[i]));
}

// ------------------------------------------------------------------------------------------------

template<typename T>
template<typename O>
inline bool val_xvector<T>::operator>(const O& other) const {
    return this->size() > other.size();
}

template<typename T>
template<typename O>
inline bool val_xvector<T>::operator<(const O& other) const {
    return this->size() < other.size();
}

template<typename T>
template<typename O>
inline bool val_xvector<T>::operator==(const O& other) const
{
    for (T& it : other) {
        if (this->Lacks(it))
            return false;
    }
    return true;
}

template<typename T>
template<typename O>
inline bool val_xvector<T>::operator!=(const O& other) const
{
    for (T& it : other) {
        if (this->Lacks(it))
            return true;
    }
    return false;
}
// --------------------------------------------------------
template<typename T>
inline bool val_xvector<T>::operator>(const size_t value) const {
    return this->size() > value;
}

template<typename T>
inline bool val_xvector<T>::operator<(const size_t value) const {
    return this->size() < value;
}

template<typename T>
inline bool val_xvector<T>::operator==(const size_t value) const {
    return this->size() == value;
}

template<typename T>
inline bool val_xvector<T>::operator!=(const size_t value) const {
    return this->size() != value;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline T& val_xvector<T>::Back(size_t value) {
    return this->operator[](this->size() - value - 1);
}

template<typename T>
inline T* val_xvector<T>::GetBackPtr(size_t value) {
    return &this->operator[](this->size() - value - 1);
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline std::pair<T, T> val_xvector<T>::GetPair() const
{
    return std::pair<E, E>(this->at(0), this->at(1));
}

template<typename T>
template<typename C>
inline xvector<C> val_xvector<T>::Convert() const
{
    xvector<C> ret;
    for (typename val_xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret << C(*it);
    return ret;
}


template<typename T>
template<typename C, typename F>
inline xvector<C> val_xvector<T>::Convert(F function) const
{
    xvector<C> ret;
    for (typename val_xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret << function(*it);
    return ret;
}

template<typename T>
template<typename N>
inline xvector<xvector<T>> val_xvector<T>::Split(N count) const
{

    xvector<xvector<T>> ret_vec;
    if (count < 2) {
        if (count == 1 && this->size() == 1) {
            ret_vec[0].reserve(this->size());
            for (typename val_xvector<T>::const_iterator it = this->begin(); it != this->end(); it++) {
                ret_vec[0].push_back(*it);
            }
        }
        else
            return ret_vec;
    }

    ret_vec.reserve(static_cast<size_t>(count) + 1);
    if (!this->size())
        return ret_vec;

    N reset = count;
    count = 0;
    const N new_size = static_cast<N>(this->size()) / reset;
    for (typename val_xvector<T>::const_iterator it = this->begin(); it != this->end(); it++) {
        if (count == 0) {
            count = reset;
            ret_vec.push_back(xvector<T>({ *it })); // create new xvec and add first el
            ret_vec[ret_vec.size() - 1].reserve(static_cast<size64_t>(new_size));
        }
        else {
            ret_vec[ret_vec.size() - 1] << *it;
        }
        count--;
    }
    return ret_vec;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
void val_xvector<T>::operator+=(const xvector<T>& other)
{
    this->insert(this->end(), other.begin(), other.end());
}

template<typename T>
xvector<T> val_xvector<T>::operator+(const xvector<T>& other) const 
{
    xvector<T> vret;
    vret.reserve(this->size());
    vret.insert(vret.end(),  this->begin(), this->end());
    vret += other;
    return vret;
}

// ------------------------------------------------------------------------------------------------

template<typename T>
inline size_t val_xvector<T>::Size() const
{
    return this->size();
}

template<typename T>
inline void val_xvector<T>::Organize()
{
    std::multiset<T> set_arr;
    for (typename val_xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        set_arr.insert(*it);

    this->clear();
    this->reserve(set_arr.size());

    for (typename std::multiset<T>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        this->push_back(*it);
}

template<typename T>
inline void val_xvector<T>::RemoveDups()
{
    std::set<T> set_arr;
    for (typename val_xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        set_arr.insert(*it);

    this->clear();
    this->reserve(set_arr.size());

    for (typename std::set<T>::const_iterator it = set_arr.begin(); it != set_arr.end(); it++)
        this->push_back(*it);
}

// -----------------------------------------------------------------------------------------------

template<typename T>
template<typename F>
inline xvector<T> val_xvector<T>::Sort(F func)
{
    std::sort(this->begin(), this->end(), func);
    return *this;
}

template<typename T>
inline xvector<T*> val_xvector<T>::GetPtrs()
{
    xvector<T*> ret_vec;
    for (T& item : *this)
        ret_vec << &item;
    
    return ret_vec;
}

template<typename T>
template<typename F, typename... A>
inline void val_xvector<T>::Proc(F&& function, A&& ...Args)
{
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++) {
        if (function(*it, Args...))
            break;
    }
}

template<typename T>
template<typename F, typename... A>
inline void val_xvector<T>::ThreadProc(F&& function, A&& ...Args)
{
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        Nexus<>::AddJobVal(function, *it, Args...);
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> val_xvector<T>::ForEach(F&& function, A&& ...Args)
{
    xvector<N> vret;
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        vret.push_back(function(*it, Args...));
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> val_xvector<T>::ForEach(F&& function, A&& ...Args)
{
    std::unordered_map<K, V> rmap;
    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        rmap.insert(function(*it, Args...));
    return rmap;
}

template<typename T>
template<typename N, typename F, typename ...A>
inline xvector<N> val_xvector<T>::ForEachThread(F&& function, A&& ...Args)
{
    Nexus<N>* trd = new Nexus<N>;

    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        trd->AddJobVal(function, *it, Args...);

    xvector<N> vret;
    trd->WaitAll();
    vret.reserve(trd->Size());

    for (size_t i = 0; i < trd->Size(); i++)
        vret << trd->GetWithoutProtection(i).Move();

    DeleteNexusPtr(trd);
    return vret;
}

template<typename T>
template<typename K, typename V, typename F, typename ...A>
inline std::unordered_map<K, V> val_xvector<T>::ForEachThread(F&& function, A&& ...Args)
{
    Nexus<std::pair<K,V>>* trd = new Nexus<std::pair<K, V>>;

    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        trd->AddJobVal(function, *it, Args...);

    std::unordered_map<K, V> rmap;
    trd->WaitAll();
    rmap.reserve(trd->size());

    for (size_t i = 0; i < trd->size(); i++)
        rmap.insert(trd->GetWithoutProtection(i).move());

    DeleteNexusPtr(trd);
    return rmap;
}

template<typename T>
template <typename N, typename F, typename ...A>
inline void val_xvector<T>::StartTasks(F&& function, A&& ...Args)
{
    DeleteNexusPtr(td);
    td = new Nexus<N>;

    for (typename xvector<E>::iterator it = this->begin(); it != this->end(); it++)
        td->AddJobVal(function, *it, Args...);
}

template<typename T>
template<typename N>
inline typename std::enable_if<!std::is_same<N, void>::value, xvector<N>>::type val_xvector<T>::GetTasks() const
{
    xvector<N> vret;
    td->WaitAll();
    for (size_t i = 0; i < td->size(); i++)
        vret << td->GetWithoutProtection(i).move();

    DeleteNexusPtr(td);
    return vret;
}

template<typename T>
inline bool val_xvector<T>::TasksCompleted() const
{
    if (!td)
        return true;
    return td->TasksCompleted();
}

// =============================================================================================================


template<typename T>
inline T val_xvector<T>::Sum(size_t FnSkipIdx) const
{
    if (!Size())
        return 0;

    T LnModSize = 0;
    if (FnSkipIdx && Size() > FnSkipIdx)
        LnModSize = Size() - FnSkipIdx;

    T num = 0;
    for (typename val_xvector<T>::const_iterator it = this->begin() + LnModSize; it != this->end(); it++) {
        num += *it;
    }
    return num;
}

template<typename T>
inline T val_xvector<T>::Mul(size_t FnSkipIdx) const
{
    if (!Size())
        return 0;

    if (Size() == 1)
        return (*this)[0];

    T LnModSize = 0;
    if (Size() > FnSkipIdx)
        LnModSize = Size() - FnSkipIdx;

    T num = 1;
    for (typename val_xvector<T>::const_iterator it = this->begin() + LnModSize; it != this->end(); it++) {
        num *= (*it);
    }
    return num;
}

// =============================================================================================================

template<typename T>
inline T val_xvector<T>::Join(const T& str) const
{
    T ret;
    for (typename val_xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += *it + str;

    return ret.substr(0, ret.length() - str.size());
}

template<typename T>
T val_xvector<T>::Join(const char str) const
{
    T ret;
    for (typename val_xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += *it + str;

    return ret.substr(0, ret.length() - 1);
}

template<typename T>
T val_xvector<T>::Join(const char* str) const
{
    T ret;
    for (typename val_xvector<T>::const_iterator it = this->begin(); it != this->end(); it++)
        ret += *it + str;

    return ret.substr(0, ret.length() - strlen(str));
}

template<typename T>
bool val_xvector<T>::MatchOne(const T& in_pattern) const {
    std::regex pattern(in_pattern.c_str());
    for (typename val_xvector<T>::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (std::regex_match(*iter, pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
bool val_xvector<T>::MatchOne(T&& in_pattern) const {
    return this->MatchOne(in_pattern);
}

template<typename T>
bool val_xvector<T>::MatchOne(char const* in_pattern) const {
    return this->MatchOne(T(in_pattern));
}

// =============================================================================================================

template<typename T>
bool val_xvector<T>::MatchAll(const T& in_pattern) const {
    std::regex pattern(in_pattern);
    for (typename T::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (!std::regex_match(*iter, pattern)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool val_xvector<T>::MatchAll(T&& in_pattern) const {
    return this->MatchAll(in_pattern);
}

template<typename T>
bool val_xvector<T>::MatchAll(char const* in_pattern) const {
    return this->MatchAll(T(in_pattern));
}

// =============================================================================================================


template<typename T>
bool val_xvector<T>::ScanOne(const T& in_pattern) const {
    std::regex pattern(in_pattern);
    for (typename val_xvector<T>::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (std::regex_search(*iter, pattern)) {
            return true;
        }
    }
    return false;
}

template<typename T>
bool val_xvector<T>::ScanOne(T&& in_pattern) const {
    return this->ScanOne(in_pattern);
}

template<typename T>
bool val_xvector<T>::ScanOne(char const* in_pattern) const {
    return this->ScanOne(T(in_pattern));
}

// =============================================================================================================

template<typename T>
bool val_xvector<T>::Scanall(const T& in_pattern) const {
    std::regex pattern(in_pattern);
    for (typename T::const_iterator iter = this->begin(); iter != this->end(); iter++) {
        if (!std::regex_search(*iter, pattern)) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool val_xvector<T>::Scanall(T&& in_pattern) const {
    return this->Scanall(in_pattern);
}

template<typename T>
bool val_xvector<T>::Scanall(char const* in_pattern) const {
    return this->Scanall(T(in_pattern));
}
// =============================================================================================================

template<typename T>
inline xvector<T> val_xvector<T>::Take(const T& in_pattern) const
{
    xvector<T> ret_vec;
    std::regex pattern(in_pattern);
    ret_vec.reserve(this->size()+1);
    for (size_t i = 0; i < this->size(); i++) {
        if ((std::regex_search((*this)[i], pattern)))
            ret_vec.push_back((*this)[i]);
    }
    return ret_vec;
}

template<typename T>
inline xvector<T> val_xvector<T>::Take(T&& in_pattern) const
{
    return this->Take(in_pattern);
}

template<typename T>
inline xvector<T> val_xvector<T>::Take(char const* in_pattern) const
{
    return this->Take(T(in_pattern));
}


template<typename T>
inline xvector<T> val_xvector<T>::Remove(const T& in_pattern) const
{
    xvector<T> ret_vec;
    ret_vec.reserve(this->size()+1);
    std::regex pattern(in_pattern.c_str());
    for (size_t i = 0; i < this->size(); i++) {
        if(!(std::regex_search((*this)[i].c_str(), pattern)))
            ret_vec.push_back((*this)[i]);
    }
    return ret_vec;
}

template<typename T>
inline xvector<T> val_xvector<T>::Remove(T&& in_pattern) const
{
    return this->Remove(in_pattern);
}

template<typename T>
inline xvector<T> val_xvector<T>::Remove(char const* in_pattern) const
{
    return this->Remove(T(in_pattern));
}

// =============================================================================================================

template<typename T>
inline xvector<T> val_xvector<T>::SubAll(const T& in_pattern, const T& replacement) const
{
    xvector<T> ret_vec;
    std::regex pattern(in_pattern.c_str());
    ret_vec.reserve(this->size()+1);
    for (typename val_xvector<T>::const_iterator iter = this->begin(); iter != this->end(); iter++) 
        ret_vec << T(std::regex_replace(*iter, pattern, replacement));
    return ret_vec;
}

template<typename T>
inline xvector<T> val_xvector<T>::SubAll(T&& in_pattern, T&& replacement) const
{
    return this->SubAll(in_pattern, replacement);
}

template<typename T>
inline xvector<T> val_xvector<T>::SubAll(char const* in_pattern, char const* replacement) const
{
    return this->SubAll(T(in_pattern), T(replacement));
}

// =============================================================================================================

template<typename T>
xvector<T> val_xvector<T>::operator()(long double x, long double y, long double z, const char removal_method) const {

    size_t m_size = this->size();
    xvector<T> n_arr;
    n_arr.reserve(m_size + 4);

    double n_arr_size = static_cast<double>(m_size) - 1;

    if (z >= 0) {

        if (x < 0) { x += n_arr_size; }

        if (!y) { y = n_arr_size; }
        else if (y < 0) { y += n_arr_size; }
        ++y;

        if (x > y) { return n_arr; }

        typename val_xvector<T>::const_iterator iter = this->begin();
        typename val_xvector<T>::const_iterator stop = this->begin() + static_cast<size_t>(y);

        if (z == 0) { // forward direction with no skipping
            for (iter += static_cast<size_t>(x); iter != stop; ++iter)
                n_arr.push_back(*iter);
        }
        else if (removal_method == 's') { // forward direction with skipping
            double iter_insert = 0;
            --z;
            for (iter += static_cast<size_t>(x); iter != stop; ++iter) {
                if (!iter_insert) {
                    n_arr.push_back(*iter);
                    iter_insert = z;
                }
                else {
                    --iter_insert;
                }
            }
        }
        else {
            double iter_insert = 0;
            --z;
            for (iter += static_cast<size_t>(x); iter != stop; ++iter) {
                if (!iter_insert) {
                    iter_insert = z;
                }
                else {
                    n_arr.push_back(*iter);
                    --iter_insert;
                }
            }
        }
    }
    else { // reverse direction
        z = z * -1 - 1;
        if (!x) { x = n_arr_size; }
        else if (x < 0) { x += n_arr_size; }

        if (!y) { y = 0; }
        else if (y < 0) { y += n_arr_size; }

        if (y > x) { return n_arr; }
        
        typename val_xvector<T>::const_reverse_iterator iter = this->rend() - static_cast<size_t>(x) - 1;
        typename val_xvector<T>::const_reverse_iterator stop = this->rend() - static_cast<size_t>(y);

        size_t iter_insert = 0;

        if (z == 0) {
            for (; iter != stop; ++iter) {
                if (!iter_insert)
                    n_arr.push_back(*iter);
            }
        }
        else if (removal_method == 's') {
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    n_arr.push_back(*iter);
                    iter_insert = static_cast<size_t>(z);
                }
                else {
                    --iter_insert;
                }
            }
        }
        else {
            for (; iter != stop; ++iter) {
                if (!iter_insert) {
                    iter_insert = static_cast<size_t>(z);
                }
                else {
                    n_arr.push_back(*iter);
                    --iter_insert;
                }
            }
        }
    }
    return n_arr;
}
