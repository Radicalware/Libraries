
#include "Atomic/ClassAtomic.h"



//template<typename T>
//
//const std::atomic<std::shared_ptr<T>>& 
//Atomic<T, typename std::enable_if_t<std::is_class<T>::value>>::Base() const
// { 
//    return *reinterpret_cast<std::atomic<std::shared_ptr<T>>>(this); 
//}