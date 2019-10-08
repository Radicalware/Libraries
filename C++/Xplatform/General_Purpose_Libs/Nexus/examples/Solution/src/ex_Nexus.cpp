#include<iostream>
#include<exception>

#include "Nexus.h"


using std::cout;
using std::endl;
using std::bind;

using ull = unsigned long long;

ull prime_number(size_t placement) {
	if (placement == 1)
		return 1;
	else if (placement < 1)
		throw "A value less than 1 is not allowed"; // char throw
	else if (placement == 99)
		throw std::string().at(1); // std throw

	ull prime = 1;
	int current = 1;
	int counter = 0;
	do {
		current++;
		bool zeroed = false;
		
		for (ull i = 2; i <= current / 2; i++) {
			if (current % i == 0) {
				zeroed = true;
				break;
			}
		}
		if (!zeroed) {
			prime = current;
			counter++;
		}
	} while (counter < placement || current > 9223372036854775806);

	return prime;
}

template<typename T>
void test_exception(xstring input, Nexus<T>& nex) {
	try {
		cout << nex.get(input) << endl;
	}
	catch (const std::exception& exc) {
		cout << "Exception by: \"" << input << "\"\n" << exc.what() << "\n\n";
	}
	catch (const char* err_str) {
		cout << "Exception by: \"" << input << "\"\n" << err_str << "\n\n";
	}
}


template<typename T>
void test_exception(size_t input, Nexus<T>& nex) {
	try {
		cout << nex.get(input) << endl;
	}
	catch (const std::exception& exc) {
		cout << "Exception by: \"" << input << "\"\n" << exc.what() << "\n\n";
	}
	catch (const char* err_str) {
		cout << "Exception by: \"" << input << "\"\n" << err_str << "\n\n";
	}
}

int main(){

	// =============================================================================================

	Nexus<ull> nex;
	nex.add_job("ten",    prime_number, 10);
	nex.add_job("eleven", prime_number, 11);
	nex.add_job("twelve", prime_number, 12);
	nex.add_job(prime_number, 13);             // no key for 13, you must reference by int
	nex.add_job("fourteen", prime_number, 14);
	nex.add_job("fifteen",  prime_number, 15);

	cout << "thread zero  : " << nex.get(0) << endl;
	cout << "thread one   : " << nex.get(1) << endl;
	cout << "thread two   : " << nex.get(2) << endl;
	cout << "thread three : " << nex.get(3) << endl;
	cout << "thread four  : " << nex.get(4) << endl;
	cout << "thread five  : " << nex.get(5) << endl;
	                      test_exception(20, nex); // 20 tasks were not created.

	cout << "thread zero  : " << nex.get("ten") << endl;
	cout << "thread one   : " << nex.get("eleven") << endl;
	cout << "thread two   : " << nex.get("twelve") << endl;
	                      test_exception("thirteen", nex); // key thirteen was not given.
	cout << "thread four  : " << nex.get("fourteen") << endl;
	cout << "thread five  : " << nex.get("fifteen")  << "\n\n";

	// =============================================================================================

	nex.add_job( "Failure 0", prime_number,  0);
	nex.add_job("Failure 99", prime_number, 99);

	test_exception("Failure 0", nex);
	test_exception("Failure 99", nex);
	cout << "\n";

	// =============================================================================================

	Nexus<xstring> nexx;
	auto prime_34_arg0 = []() -> xstring {
		return to_xstring(prime_number(34));
	};
	auto prime_35_arg1 = [](int val) -> xstring {
		return to_xstring(prime_number(val));
	};
	auto prime_46_arg2 = [](int val1, const char* val2) -> xstring {
		return to_xstring(prime_number(val1+std::atoi(val2)));
	};

	nexx.add_job(prime_34_arg0);
	nexx.add_job(prime_35_arg1, 35);
	nexx.add_job("two args", prime_46_arg2, 36, "10");
	cout << nexx.get(0) << endl;
	cout << nexx.get(1) << endl;
	cout << nexx.get("two args") << endl;
	
	test_exception(3, nexx);

	// =============================================================================================

	nex.wait_all();
	const size_t start_loc = nex.size();
	cout << "Jobs in progress should not exceed: " << nex.thread_count() << endl;
	cout << "Jobs In Progress: " << nex.threads_used() << endl;
	cout << "Starting Loop\n";
	for (int i = 0; i < nex.thread_count() * 2; i++) {
		nex.add_job(prime_number, 10000);
		nex.sleep(10);
		cout << "Jobs Running: " << nex.threads_used() << endl;
	}
	cout << "waiting for jobs to finish\n";
	nex.wait_all();

	cout << "total: " << nex.size() << endl;
	for (int i = start_loc; i < nex.size(); i++) {
		try {
			cout << "result " << i << ": " << nex.get(i) << endl;
		}
		catch (...) {
			test_exception(i, nex);
		}
	}
	nex.wait_all();
	cout << "Jobs In Progress: " << nex.threads_used() << endl;

	return 0;
}
