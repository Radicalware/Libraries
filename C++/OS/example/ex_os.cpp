#include<iostream>
#include<vector>
#include<string>
#include<ctime>

#include "./os.h" // Found on "https://github.com/Radicalware"

using std::cout;
using std::endl;
using std::string;
using std::vector;



int main(int argc, char** argv){

	OS* os = new OS();

	std::string data = "test write 1\ntest write line 2\n";

	std::string test_read  = "test_read.txt";
	std::string test_write = "test_write.txt";
	std::string blank      = "";

	cout << "----------------------------------------------\n";
	cout << test_read << endl;
	cout << os->open(test_read).read() << endl;

	cout << "----------------------------------------------\n";
	cout << "** current file context\n\n" << os->open(test_write).read();
	os->open(test_write).write(data);
	cout << "** new file context\n\n";
	cout << os->open(test_write).read();
	os->open(test_write).write(blank);
	cout << os->open(test_write).read() << endl;
	cout << "----------------------------------------------\n";
	std::vector<string> dirs;
	os->dir("./","-r","-n",dirs);
	for(auto&i : dirs) cout << i << endl;
	cout << "----------------------------------------------\n";

	char command_char[] {"ls -la"};
	std::string command_str {"ls -la"};
	std::string console_log_char = os->console(command_char);
	std::string console_log_str  = os->console(command_str);

	cout << console_log_str << endl;
	cout << "----------------------------------------------\n";

	os->set_args(argc, argv);

	cout << os->args_str() << endl;
	cout << os->base_args_str() << endl;
	cout << os->sub_args_str() << endl;

	cout << os->if_key("-first") << endl;
	cout << os->if_key("-no_arg") << endl;

	cout << os->if_value("-first","argF2") << endl;
	cout << os->if_value("-first","no_value") << endl;

	cout << os->value("-second",2) << endl;

	
	return 0;
}