#pragma once

#include "Dir_Type.h"
#include<string>

class File_Names : public Dir_Type
{
private:
	std::string m_traverse_target;

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

	void set_old();
	void set_target();
	void set_old(std::string& m_old);
	void set_target(std::string& m_target);

	std::string fix_slash(std::string& item);
	void imaginary_path();

	void assert_folder_syntax(const std::string& folder1, const std::string& folder2 = "");

	std::string old();
	std::string target();
	std::string traverse_target();
};
