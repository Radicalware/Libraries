#pragma once

#include<functional>
#include<string>
#include<any>

template<typename T>
class Task
{
protected:
    std::string*        m_name = nullptr; // set as pointer because it isn't always used
    std::function<T()>  m_method;
    size_t              MnMutexID = 0;

public:
    Task(      Task&& task) = delete; // Used Shared Pointer
    Task(const Task&  task) = delete; // Used Shared Pointer
    Task(      std::function<T()>&&    i_method);
    Task(const std::function<T()>&     i_method);
    Task(      std::function<T()>&&    i_method,       std::string&& i_name);
    Task(const std::function<T()>&     i_method, const std::string&  i_name);
    Task(      std::function<T()>&&    i_method, const size_t i_mutex_id);
    Task(const std::function<T()>&     i_method, const size_t i_mutex_id);
    ~Task();

    // Use Shared Pointers
    void operator=(const Task& task) = delete;
    void operator=(Task&& task)      = delete;

    void AddMethod(const std::function<T()>& i_method);

    bool HasName() const;
    const std::string* GetNamePtr() const;
          std::string  GetName() const;
          size_t       GetMutexID() const;
    T RunTask();
};

// ----------------------------------------------------------------------------------------------------
template<typename T>
inline Task<T>::Task(std::function<T()>&& i_method) : m_method(std::move(i_method))
{   }
template<typename T>
inline Task<T>::Task(const std::function<T()>& i_method) : m_method(i_method)
{   }
// ----------------------------------------------------------------------------------------------------
template<typename T>
inline Task<T>::Task(std::function<T()>&& i_method, std::string&& i_name) : m_method(std::move(i_method))
{
    m_name = new std::string(std::move(i_name));
}
template<typename T>
inline Task<T>::Task(const std::function<T()>& i_method, const std::string& i_name) : m_method(i_method)
{
    m_name = new std::string(i_name);
}

template<typename T>
inline Task<T>::Task(std::function<T()>&& i_method, size_t i_mutex_id) : m_method(std::move(i_method)), MnMutexID(i_mutex_id)
{
}
template<typename T>
inline Task<T>::Task(const std::function<T()>& i_method, const size_t i_mutex_id) : m_method(i_method), MnMutexID(i_mutex_id)
{
}

// ----------------------------------------------------------------------------------------------------

template<typename T>
inline Task<T>::~Task()
{
    if(m_name != nullptr)
        delete m_name;
}

template<typename T>
inline void Task<T>::AddMethod(const std::function<T()>& i_method)
{
    m_method = i_method;
}

template<typename T>
inline bool Task<T>::HasName() const
{
    return (m_name != nullptr) ? true : false;
}

template<typename T>
inline const std::string* Task<T>::GetNamePtr() const
{
    return m_name;
}

template<typename T>
inline std::string Task<T>::GetName() const
{
    return *m_name;
}

template<typename T>
inline size_t Task<T>::GetMutexID() const
{
    return MnMutexID;
}

template<typename T>
inline T Task<T>::RunTask()
{
    return m_method();
}
