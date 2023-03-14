#pragma once

#include<functional>
#include<string>
#include<any>

template<typename T>
class Task
{
protected:
    std::string*        MsName = nullptr; // set as pointer because it isn't always used
    std::function<T()>  MfMethod;
    xint                MnMutexID = 0;
    xint                MnID = 0;

public:
    Task(      Task&& task) = delete; // Used Shared Pointer
    Task(const Task&  task) = delete; // Used Shared Pointer
    Task(const xint FnID,       std::function<T()>&&    FfMethod);
    Task(const xint FnID, const std::function<T()>&     FfMethod);
    Task(const xint FnID,       std::function<T()>&&    FfMethod,       std::string&& FsName);
    Task(const xint FnID, const std::function<T()>&     FfMethod, const std::string&  FsName);
    Task(const xint FnID,       std::function<T()>&&    FfMethod, const xint FnMutexID);
    Task(const xint FnID, const std::function<T()>&     FfMethod, const xint FnMutexID);
    ~Task();

    // Use Shared Pointers
    void operator=(const Task& task) = delete;
    void operator=(Task&& task)      = delete;

    void AddMethod(const std::function<T()>& FfMethod);

    constexpr bool HasName() const { return MsName != nullptr; }
    const std::string* GetNamePtr() const;
          std::string  GetName() const;
          xint         GetMutexID() const;
    T RunTask();
    xint GetID() const { return MnID; }
};

// ----------------------------------------------------------------------------------------------------
template<typename T>
inline Task<T>::Task(const xint FnID, std::function<T()>&& FfMethod) : 
    MnID(FnID), MfMethod(std::move(FfMethod))
{
}
template<typename T>
inline Task<T>::Task(const xint FnID, const std::function<T()>& FfMethod) : 
    MnID(FnID), MfMethod(FfMethod)
{
}
// ----------------------------------------------------------------------------------------------------
template<typename T>
inline Task<T>::Task(const xint FnID, std::function<T()>&& FfMethod, std::string&& FsName) : 
    MnID(FnID), MfMethod(std::move(FfMethod))
{
    MsName = new std::string(std::move(FsName));
}
template<typename T>
inline Task<T>::Task(const xint FnID, const std::function<T()>& FfMethod, const std::string& FsName) : 
    MnID(FnID), MfMethod(FfMethod), MsName(new std::string(FsName))
{
}

template<typename T>
inline Task<T>::Task(const xint FnID, std::function<T()>&& FfMethod, xint FnMutexID) : 
    MnID(FnID), MfMethod(std::move(FfMethod)), MnMutexID(FnMutexID)
{
}
template<typename T>
inline Task<T>::Task(const xint FnID, const std::function<T()>& FfMethod, const xint FnMutexID) : 
    MnID(FnID), MfMethod(FfMethod), MnMutexID(FnMutexID)
{
}

// ----------------------------------------------------------------------------------------------------

template<typename T>
inline Task<T>::~Task()
{
    if(MsName != nullptr)
        delete MsName;
}

template<typename T>
inline void Task<T>::AddMethod(const std::function<T()>& FfMethod)
{
    MfMethod = FfMethod;
}

template<typename T>
inline const std::string* Task<T>::GetNamePtr() const
{
    return MsName;
}

template<typename T>
inline std::string Task<T>::GetName() const
{
    return *MsName;
}

template<typename T>
inline xint Task<T>::GetMutexID() const
{
    return MnMutexID;
}

template<typename T>
inline T Task<T>::RunTask()
{
    return MfMethod();
}
