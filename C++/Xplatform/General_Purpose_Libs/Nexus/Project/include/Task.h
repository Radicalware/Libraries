#pragma once

#include<functional>
#include<string>

template<typename T>
class Task
{
    std::string* m_name = nullptr; // set as pointer because it isn't always used
    std::function<T()> m_method;
    bool m_blank = false;

public:
    Task();
    Task(      Task&& task) noexcept;
    Task(const Task&  task);
    Task(      std::function<T()>&& i_method);
    Task(const std::function<T()>&  i_method);
    Task(      std::function<T()>&& i_method,       std::string&& i_name);
    Task(const std::function<T()>&  i_method, const std::string&  i_name);
    ~Task();

    void operator=(const Task& task);
    void operator=(Task&& task) noexcept;

    void add_method(const std::function<T()>& i_method);

    bool blank() const;
    bool has_name() const;
    const std::string* name_ptr() const;
    const std::string name() const;
    T operator()();
};


template<typename T>
inline Task<T>::Task()
{
    m_blank = true;
}
// ----------------------------------------------------------------------------------------------------

template<typename T>
inline Task<T>::Task(Task&& task) noexcept{
    this->operator=(std::move(task));
}

template<typename T>
inline Task<T>::Task(const Task& task){
    this->operator=(task);
}
// ----------------------------------------------------------------------------------------------------
template<typename T>
inline Task<T>::Task(std::function<T()>&& i_method) : m_method(std::move(i_method))
{   }
template<typename T>
inline Task<T>::Task(const std::function<T()>& i_method) : m_method(i_method)
{   }
// ----------------------------------------------------------------------------------------------------
template<typename T>
inline Task<T>::Task(std::function<T()>&& i_method, std::string&& i_name): m_method(std::move(i_method))
{
    m_name = new std::string(std::move(i_name));
}
template<typename T>
inline Task<T>::Task(const std::function<T()>& i_method, const std::string& i_name): m_method(i_method)
{
    m_name = new std::string(i_name);
}

template<typename T>
inline void Task<T>::operator=(const Task& task)
{
    if (task.m_name != nullptr) {
        if(m_name != nullptr)
            delete m_name;
        m_name = new std::string(*task.m_name);
    }
    m_method = task.m_method;
    m_blank = false;
}

template<typename T>
inline void Task<T>::operator=(Task&& task) noexcept
{
    if (task.m_name != nullptr) {
        if (m_name != nullptr)
            delete m_name;
        m_name = new std::string(std::move(*task.m_name));
    }
    m_method = std::move(task.m_method);
    m_blank = false;
}

// ----------------------------------------------------------------------------------------------------

template<typename T>
inline Task<T>::~Task()
{
    if(m_name != nullptr)
        delete m_name;
}

template<typename T>
inline void Task<T>::add_method(const std::function<T()>& i_method)
{
    m_method = i_method;
    m_blank = false;
}

template<typename T>
inline bool Task<T>::blank() const
{
    return m_blank;
}

template<typename T>
inline bool Task<T>::has_name() const
{
    return (m_name != nullptr) ? true : false;
}

template<typename T>
inline const std::string* Task<T>::name_ptr() const
{
    return m_name;
}

template<typename T>
inline const std::string Task<T>::name() const
{
    return *m_name;
}

template<typename T>
inline T Task<T>::operator()()
{
    return m_method();
}
