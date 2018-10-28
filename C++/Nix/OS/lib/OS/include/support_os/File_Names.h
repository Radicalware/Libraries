#pragma once

#include<string>

class File_Names
{
private:
	std::string m_old;
	std::string m_target;

public:
	File_Names() {};
	~File_Names() {};

	File_Names(std::string i_old, std::string i_target);
	File_Names(std::string i_target);


	void check_dir_start(std::string& item);
	std::string fix_slash(std::string& item);


	void set_old(std::string item);
	void set_target(std::string item);


	void assert_folder_syntax(const std::string& folder1, const std::string& folder2 = "");

	std::string old();
	std::string target();
};