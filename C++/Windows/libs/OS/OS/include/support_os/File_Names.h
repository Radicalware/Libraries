#pragma once

#include<string>

class File_Names
{
private:
	std::string m_old;
	std::string m_target;
	bool m_rexit;
	// Checking for file input consumes a lot of time
	// Set it true for when users input data.
	// Set it false for any other time.
	// ex: os.set_file_regex(true);
	// to turn it on.

public:
	File_Names(bool rexit);
	~File_Names() {};

	File_Names(bool rexit, std::string i_old, std::string i_target);
	File_Names(bool rexit, std::string i_target);


	void check_dir_start(std::string& item);
	std::string fix_slash(std::string& item);


	void set_old(std::string item);
	void set_target(std::string item);


	void assert_folder_syntax(const std::string& folder1, const std::string& folder2 = "");

	std::string old();
	std::string target();
};