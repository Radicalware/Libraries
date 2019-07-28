#pragma once

#include "xstring.h"

#include "Dir_Type.h"
#include<string>

class File_Names : public Dir_Type
{
private:
	xstring m_traverse_target;

	xstring m_old;
	xstring m_target;

	bool m_rexit;
	// Checking for file input consumes a lot of time
	// Set it true for when users input data.
	// Set it false for any other time.
	// ex: os.set_file_regex(true);
	// to turn it on.

public:
	File_Names(bool rexit);
	~File_Names() {};

	File_Names(bool rexit, xstring i_old, xstring i_target);
	File_Names(bool rexit, xstring i_target);

	void set_old();
	void set_target();
	void set_old(xstring& m_old);
	void set_target(xstring& m_target);

	xstring fix_slash(xstring& item);
	void imaginary_path();

	void assert_folder_syntax(const xstring& folder1, const xstring& folder2 = "");

	xstring old();
	xstring target();
	xstring traverse_target();
};
