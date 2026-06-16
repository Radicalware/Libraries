Keep all variables Pascal Case (not camel case)

Prefix >>>
```
    1st Char is the Scope and Upper Case:
        L = Local
        F = Function Parameter
        M = Module (of a class)
        S = Static
        G = Global (do not use)

    2nd Char is the Variable Type and Lower Case:
        o = object
        s = string
        n = numeric
        b = bool
        e = enum
        m = map
        v = vector/deque/list/array/etc
        f = function/lambda/bind/etc
        x = template object
```
Prefix Examples >>>
```
        LvDates = Locally scoped list of dates
        MsDates = Class module that is a string of dates (probably CSV string)
        FbDate  = A function parameter that is a boolean to use a date
        SmDates = A Static object of mapped dates
        LoDate  = A locally referenced date object
        FxDate  = A function param that is a template of Date types

      template<typename T> auto Function(const xvector<T>& FvDates);
```
Suffix >>>
```
    * If it is a pointer type, then end it with Ptr
    * Example = auto LoDate = RA::MakeShared<Date>();
    * Any time you have a pointer, always use the GET macro
    * GET(LoDate) // then use the generated reference.
    * The GET macro will throw an exception if the pointer is null
    * Always have functions wrapped in Begin()/Rescue() macros for handling
    * Return a ref (instead of ptr), if the Ptr is a part of the class
```
Suffix Examples >>>
```
    void Function(xp<RA::Date> FoDatePtr)
    {
        Begin();
        GET(FoDate);
        // -- do something useful -- //
        Rescue();
    }
```
