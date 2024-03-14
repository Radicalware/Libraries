#pragma once

#include <climits>
#include <set>
#include <sstream>
#include <memory>

#ifndef The
#define The (*this)
#endif

#ifndef TTT
#define TTT template<typename T>
#endif

#ifndef _TypeXP_
#define _TypeXP_
TTT
using sxp = std::shared_ptr<T>;
#endif

#ifndef __TypeXINT__
#define __TypeXINT__
using xint = size_t;
#endif


template<typename T = int>
struct Key
{
    T MnLower = 0;
    T MnUpper = 0;

    Key() = default;
    Key(T FnFirst, T FnSecond) :
        MnLower((FnFirst < FnSecond) ? FnFirst : FnSecond),
        MnUpper((FnFirst > FnSecond) ? FnFirst : FnSecond)
    {
    }
    Key(std::initializer_list<T>&& Args);

    Key(const Key& Other) { MnLower = Other.MnLower; MnUpper = Other.MnUpper; }
    Key(Key&& Other) { MnLower = Other.MnLower; MnUpper = Other.MnUpper; }
    Key& operator=(const Key& Other) { MnLower = Other.MnLower; MnUpper = Other.MnUpper; }
    Key& operator=(Key&& Other) { MnLower = Other.MnLower; MnUpper = Other.MnUpper; }

    bool operator==(const     T   FnVal) const { return (MnLower <= FnVal && FnVal <= MnUpper); }
    bool operator==(const Key<T>& Other) const { return (MnLower == Other.MnLower && MnUpper == Other.MnUpper); }

    bool operator!=(const     T   FnVal) const { return !(The == FnVal); }
    bool operator!=(const Key<T>& Other) const { return !(The == Other); }

    bool operator<(const Key<T>& Other) const { 
        if (The == Other) return false; 
        return MnLower < Other.MnLower; 
    }
    bool operator<(const     T& Other)  const { 
        if (The == Other) return false; 
        return MnLower < Other; 
    }

    xint GetHash() const
    {
        xint h1 = std::hash<T>{}(MnLower);
        xint h2 = std::hash<T>{}(MnUpper);
        //return h1 ^ (h2 << 1);
        return h1;
    }
    std::string ToString() const;

    struct Equals {
        using is_transparent = std::true_type;
        bool operator()(const Key<T>& lhs, const T& rhs) const {
            return lhs == rhs;
        }
        bool operator()(const Key<T>& lhs, const Key<T>& rhs) const {
            return lhs == rhs;
        }
    };
    struct Compare {
        using is_transparent = std::true_type;
        bool operator()(const Key<T>& Obj, const T& Num) const {
            if (Obj == Num)
                return false;
            return Obj.MnLower < Num;
        }
        bool operator()(const T& Num, const Key<T>& Obj) const {
            if (Obj == Num)
                return false;
            return Num < Obj.MnLower;
        }
        bool operator()(const Key<T>& lhs, const Key<T>& rhs) const {
            return lhs < rhs;
        }
    };
    struct Hash {
        using is_transparent = std::true_type;
        size_t operator()(const T& Obj) const {
            return std::hash<T>{}(Obj);
        }
        size_t operator()(const Key<T>& Obj) const {
            return (size_t)Obj.GetHash();
        }
    };
};

template<typename T>
inline Key<T>::Key(std::initializer_list<T>&& Args) 
{
    auto LnCount = 0;
    for (const auto& Val : Args) {
        if (LnCount == 0)
            MnLower = Val;
        else if (LnCount == 1)
            MnUpper = Val;
        ++LnCount;
    }
    if (LnCount == 1)
        MnUpper = MnLower;
}

template<typename T>
inline std::string Key<T>::ToString() const
{
    std::stringstream OSS;
    OSS << '[' << MnLower << ", " << MnUpper << ']';
    return OSS.str();
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Key<T>& FoKey)
{
    out << FoKey.ToString();
    return out;
}


namespace std
{
    template<typename T>
    struct BaseHashCircle
    {
        _CXX17_DEPRECATE_ADAPTOR_TYPEDEFS typedef Key<T> argument_type;
        _CXX17_DEPRECATE_ADAPTOR_TYPEDEFS typedef xint result_type;
    };

    template<typename T>
    struct hash<Key<T>> : public BaseHashCircle<T>
    {
        xint operator()(const Key<T>& FoKey) const noexcept {
            return FoKey.GetHash();
        }
    };
    template<typename T>
    struct hash<Key<T>*> : public BaseHashCircle<T>
    {
        xint operator()(const Key<T>* FoKey) const noexcept {
            return FoKey->GetHash();
        };
    };
    template<typename T>
    struct hash<const Key<T>*> : public BaseHashCircle<T>
    {
        xint operator()(const Key<T>* FoKey) const noexcept {
            return FoKey->GetHash();
        };
    };
};


template<typename T>
bool operator==(const std::shared_ptr<Key<T>>& Left, const std::shared_ptr<Key<T>>& Right) {
    return *Left == *Right;
}
