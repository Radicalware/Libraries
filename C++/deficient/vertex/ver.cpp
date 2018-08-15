#include "Timer.h" # a learncpp lib found on github.com/radicalware
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>   
#include <cstdint>
#include <cstring>




using std::string;
using std::vector;
using std::cout;
using std::endl;

template<typename T>
class vertex
{

private:
	T* m_array;

	T* m_storage_1;
	T* m_storage_2;

	bool m_init = false;
	bool m_using1;

	size_t m_size = 0;      // slots used
	size_t m_capacity = 10; // slots avalible 

public:
	vertex(){	};

	~vertex(){
		if (m_init){
			if(m_using1){
				delete [] m_storage_1;
			}else{
				delete [] m_storage_2;
			}
		}
		// delete [] m_storage_1;
		// delete [] m_storage_2;	
	}

	void append(T value){
		if(m_init){
			if (!(m_size < m_capacity)){ // copying
				m_capacity *= 2;

				if(m_using1){
					m_using1 = false;
					m_storage_2 = new T[m_capacity]; 

					// for(size_t i = 0; i < m_size; i++)
					// 	m_storage_2[i] = m_storage_1[i];

    				std::memcpy(m_storage_2, &m_storage_1[0], (m_size * sizeof(T)));

					delete [] m_storage_1;

					m_size += 1;
					m_storage_2[m_size] = value;
					m_array = m_storage_2;

				}else{ 
					m_using1 = true;
					m_storage_1 = new T[m_capacity];

					// for(size_t i = 0; i < m_size; i++)
					// 	m_storage_1[i] = m_storage_2[i];
    				
    				std::memcpy(m_storage_1, &m_storage_2[0], (m_size * sizeof(T)));

					delete [] m_storage_2;

					m_size += 1;
					m_storage_1[m_size] = value;
					m_array = m_storage_1;
				}
			}else{ // adding without copying
				m_size += 1;
				if (m_using1){
					m_storage_1[m_size] = value;
				}else{
					m_storage_2[m_size] = value;
				}
			}
		}else{ cout << "init " << endl;
			m_init = true;
			m_using1 = true;

			m_storage_1 = new T[m_capacity]; 
			m_storage_1[m_size] = value;
			m_array = m_storage_1;
		}
	}

	size_t size(){return m_size;}
	size_t capacity(){return m_capacity;}

	T operator[](size_t loc){return m_array[loc];}
};



void bench_vector(int bench_size){
	std::string word = "word";
	int integer = 50;
	Timer t;

	//std::vector<std::string> vec;
	std::vector<int> vec;

	for (int i = 0; i < bench_size; i++){
		vec.push_back(integer);
	}

	cout << "vector speed = "<< t.elapsed() << endl;
	// t.reset();

	// for (int i = 0; i < 99999; i++){
	// 	vec.insert(vec.begin()+i, i*2);
	// }

	//cout << "insert speed = "<< t.elapsed() << endl;
}

void bench_vertex(int bench_size){
	std::string word = "word";
	int integer = 50;
	Timer t;

	vertex<int> ver;
	for(int i = 0; i < bench_size; i++){
		ver.append(integer);
	}
	cout << "vertex speed = " << t.elapsed() << endl;
}

void test_function(int bench_size){
	std::string word = "word";
	int integer = 50;
	Timer t;

	vertex<int> ver;
	//vertex<std::string> ver;

	for(int i = 5; i < bench_size; i++){
		ver.append(i);
	}
	cout << "size = " << ver.size() << endl;
}

int main(int argc, char *argv[])
{

	int bench_size = 55;
	bench_vector(bench_size);
	bench_vertex(bench_size);

	//test_function(bench_size);


    return 0;
}
