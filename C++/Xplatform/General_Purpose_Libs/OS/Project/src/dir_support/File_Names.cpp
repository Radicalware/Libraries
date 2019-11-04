
#include "dir_support/File_Names.h"


File_Names::File_Names(bool rexit) : m_rexit(rexit) 
{
    full[0] = '\0';
}


File_Names::File_Names(bool rexit, const xstring& i_target) :
    m_rexit(rexit), m_target(i_target)
{
    this->fix_slash(m_target);
}

File_Names::File_Names(bool rexit, const xstring& i_old, const xstring& i_target) :
    m_rexit(rexit), m_old(i_old), m_target(i_target) 
{
    this->set_old();
    this->set_target();
}


// syntax to differentiate folder from file is that folders end with a '/' or a '\\'

void File_Names::fix_slash(xstring& item) {
#if defined(WIN_BASE)
    char* unused = _fullpath(full, item.sub("/", "\\\\").c_str(), _MAX_PATH);
#elif defined(NIX_BASE)
    char* val = realpath(item.sub("\\\\", "/").c_str(), full);
#endif
    item = full; // TODO: fix, this causes an issue in linux when global OS is set in the h file
    // FIX: Just don't set the extern OS in the header, instead set it in the main.cpp
}

void File_Names::assert_folder_syntax(const xstring& folder1, const xstring& folder2) {

    auto asserts = [](const xstring& folder) -> void {

        if (!folder.match(R"(^([\./\\]+?)[\-\d\w\.\\/]+$)")) {
            throw std::runtime_error("Failed Dir Syntax = "
                R"(^([\./\\]+?)[\-\d\w\.\\/]+$)"
                "\n  what():  Dir Item: " + folder + "\n");
        }

        if (folder.scan(R"([^\\]\s)")) {
            throw std::runtime_error("You can't have a space in a dir item\n" \
                "  what():  without an escape char\n");
        }
    };
    asserts(folder1);
    if (folder2.size()) {
        asserts(folder2);
    }
}

void File_Names::set_old() {
    if (m_rexit)
        this->assert_folder_syntax(m_old);
    this->fix_slash(m_old);
}

void File_Names::set_target() {
    if (m_rexit)
        this->assert_folder_syntax(m_target);
    this->fix_slash(m_target);
}

void File_Names::set_old(xstring& i_old) {
    m_old = i_old;
    if (m_rexit)
        this->assert_folder_syntax(m_old);
    this->fix_slash(m_old);
}

void File_Names::set_target(xstring& i_target) {
    m_target = i_target;
    m_traverse_target = i_target;
    if (m_rexit)
        this->assert_folder_syntax(m_target);
    this->fix_slash(m_target);
}


const xstring File_Names::old() const { return m_old; }
const xstring File_Names::target() const { return m_target; }
const xstring File_Names::traverse_target() const { return m_traverse_target; }
