

#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
	#include "..\include\support_os\File_Names.h"
#else
	#include "../../include/support_os/File_Names.h"
#endif

#include<string>
#include "re.h"

File_Names::File_Names(bool rexit): m_rexit(rexit){}

File_Names::File_Names(bool rexit, std::string i_old, std::string i_target) :
	m_rexit(rexit), m_old(i_old), m_target(i_target) {
	this->set_old(m_old);
	this->set_target(m_target);
}

File_Names::File_Names(bool rexit, std::string i_target) :
	m_rexit(rexit), m_target(i_target) {
	this->set_target(m_target);
}

void File_Names::check_dir_start(std::string& item) {

	if (!(
		(item[0] == '/' || item[0] == '\\') || \

		(item[0] == '.' && item[1] == '/') || \
		(item[0] == '.' && item[1] == '\\') || \

		(item[0] == '.' && item[1] == '.' && item[2] == '/') || \
		(item[0] == '.' && item[1] == '.' && item[2] == '\\')
		))
	{
		item = "./" + item;
	}
}

std::string File_Names::fix_slash(std::string& item) {
	this->check_dir_start(item);
#if defined(WIN_BASE)
	item = re::sub("/", "\\\\", item);
#elif defined(NIX_BASE)
	item = re::sub("\\\\", "/", item);
#endif		
	return item;
}

void File_Names::assert_folder_syntax(const std::string& folder1, const std::string& folder2) {

	auto asserts = [](const std::string& folder) -> void {

		if (!re::match(std::string(R"(^([\./\\]+?)[\-\d\w\.\\/]+$)"), folder)) {
			throw std::runtime_error("Failed Dir Syntax = "
				R"(^([\./\\]+?)[\-\d\w\.\\/]+$)"
				"\n  what():  Dir Item: " + folder + "\n");
		}

		if (re::scan(R"([^\\]\s)", folder)) {
			throw std::runtime_error("You can't have a space in a dir item\n" \
				"  what():  without an escape char\n");
		}
	};
	asserts(folder1);
	if (folder2.size()) {
		asserts(folder2);
	}
}

void File_Names::set_old(std::string item) {
	m_old = this->fix_slash(item);
	if(m_rexit)
		this->assert_folder_syntax(m_old);
}

void File_Names::set_target(std::string item) {
	m_target = this->fix_slash(item);
	if (m_rexit)
		this->assert_folder_syntax(m_target);
}



std::string File_Names::old() { return m_old; }
std::string File_Names::target() { return m_target;  }