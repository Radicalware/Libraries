#pragma once

// Copyright[2023][Joel Leagues aka Scourge] under the Apache V2 Licence

#define CreateConcept() \
    static_assert(!std::is_const_v<T>, "Shouldn't be const here"); \
    Model() = default; \
    virtual ~Model() = default; \
    Model(const T& v) : MxData(v) {} \
    Model(T&& v) : MxData(std::move(v)) {} \
    T MxData

// CreateConceptFunction
#define AbsCreateFunctionStep1(__ACTION__) \
    virtual void Execute##__ACTION__() const = 0;

// CreateModelFunction
#define AbsCreateFunctionStep2(__ACTION__) \
    void Execute##__ACTION__() const override { AbstractFunction::__ACTION__(MxData); }

#define CreateAbstractFunction(__ACTION__) \
    void __ACTION__() const { Obj->Execute##__ACTION__(); }


#define CreateAbstractConstructors(__ClassName__) \
    template<typename T> \
          T& Cast()       { return dynamic_cast<__ClassName__::Model<T>*>(&*this->Obj.get())->MxData; }  \
    template<typename T> \
    const T& Cast() const { return dynamic_cast<__ClassName__::Model<T>*>(&*this->Obj.get())->MxData; }  \
    \
    __ClassName__() = default; \
    \
    template<typename T> \
    __ClassName__(T&& Other){ \
        Obj = std::unique_ptr<Concept>(new Model<std::decay_t<T>>(std::forward<T>(Other))); \
    } \
    \
    template<typename T> \
    __ClassName__& operator=(T&& Other) \
    { \
        Obj = std::unique_ptr<Concept>(new Model<std::decay_t<T>>(std::forward<T>(Other))); \
        return *this; \
    } \
    template<typename T> \
    __ClassName__(const T& Other){ \
        __ClassName__(T(Other)); \
    } \
    \
    template<typename T> \
    __ClassName__& operator=(const T& Other) \
    { \
        return __ClassName__(T(Other)); \
    } \

// ----------------------------------------------------------------------------

#define AbsStepOneCreateConcept \
    struct Concept \
    { \
        virtual ~Concept() = default;

#define AbsStepTwoCreateModule \
    }; \
    \
    template<typename T> \
    struct Model : public Concept  \
    {

#define AbsStepThreeCreateSmartPtr \
        CreateConcept(); \
    }; \
    \
    std::unique_ptr<Concept> Obj;

// ----------------------------------------------------------------------------

#define AbsConstruct(__BASE__, __DERIVED__) \
        __DERIVED__(         __BASE__&& Other)  { Copy(Other); } \
        __DERIVED__(   const __BASE__&  Other)  { Copy(Other); } \
        void operator=(      __BASE__&& Other)  { Copy(Other); } \
        void operator=(const __BASE__&  Other)  { Copy(Other); } 


// ----------------------------------------------------------------------------