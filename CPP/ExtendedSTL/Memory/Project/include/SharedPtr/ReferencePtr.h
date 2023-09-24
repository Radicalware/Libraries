#pragma once

// Copyright[2023][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "RawMapping.h"
#include "SharedPtr/MakeShared.h"
#include "SharedPtr/SharedPtrObj.h"

namespace RA
{
    template<typename T>
    class ReferencePtr
    {
    public:
        CIN ReferencePtr() { MoObjPtr = MKP<RA::SharedPtr<T>>(); }
        CIN ReferencePtr(const RA::SharedPtr<T>& FoPtr);
        CIN ReferencePtr(const RA::ReferencePtr<T>&  FoPtr);
        CIN ReferencePtr(      RA::ReferencePtr<T>&& FoPtr) = delete;

        CIN bool operator!(void) const;
        CIN bool IsNull()        const;

        CIN void operator=(const RA::SharedPtr<T>&  FoPtr);
        CIN void operator=(      RA::SharedPtr<T>&& FoPtr) = delete;

        CIN void operator=(const RA::ReferencePtr<T>& FoRPtr);

        CIN auto& Get()    { return *(*MoObjPtr); }
        CIN auto& GetPtr() { return   *MoObjPtr;  }

        CIN const auto& Get()    const { return *(*MoObjPtr); }
        CIN const auto& GetPtr() const { return   *MoObjPtr;  }

        CIN       auto& operator*()       { return *(*MoObjPtr); }
        CIN const auto& operator*() const { return *(*MoObjPtr); }

        CIN       auto* operator->()       { return (*MoObjPtr).get(); }
        CIN const auto* operator->() const { return (*MoObjPtr).get(); }

    private:
        RA::SharedPtr<RA::SharedPtr<T>> MoObjPtr = nullptr;
    };
};


template<typename T> 
CIN RA::ReferencePtr<T>::ReferencePtr(const RA::SharedPtr<T>& FoPtr)
{
    MoObjPtr = MKP<RA::SharedPtr<T>>();
    *MoObjPtr = FoPtr;
}

template<typename T>
CIN RA::ReferencePtr<T>::ReferencePtr(const RA::ReferencePtr<T>& FoRPtr)
{
    MoObjPtr = MKP<RA::SharedPtr<T>>();
    MoObjPtr = FoRPtr.MoObjPtr;
}

template<typename T>
CIN bool RA::ReferencePtr<T>::operator!(void) const
{
    return (MoObjPtr.get() == nullptr || MoObjPtr.get()->get() == nullptr);
}

template<typename T>
CIN bool RA::ReferencePtr<T>::IsNull() const
{
    return (MoObjPtr.get() == nullptr || MoObjPtr.get()->get() == nullptr);
}

template<typename T>
CIN void RA::ReferencePtr<T>::operator=(const RA::SharedPtr<T>& FoPtr)
{
    if (!MoObjPtr)
        MoObjPtr = RA::MakeShared<RA::SharedPtr<T>>();
    *MoObjPtr = FoPtr;
}

template<typename T>
CIN void RA::ReferencePtr<T>::operator=(const RA::ReferencePtr<T>& FoRPtr)
{
    if (!MoObjPtr)
        MoObjPtr = RA::MakeShared<RA::SharedPtr<T>>();
    MoObjPtr = FoRPtr.MoObjPtr;
}

template<typename T>
using rp = RA::ReferencePtr<T>;
