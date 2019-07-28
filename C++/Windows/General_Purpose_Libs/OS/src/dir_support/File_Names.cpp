
#include<string>
#include<limits.h>
#include<stdlib.h>
#include "../../include/dir_support/File_Names.h"
#include "../../include/dir_support/Dir_Type.h"


File_Names::File_Names(bool rexit) : m_rexit(rexit) {}


File_Names::File_Names(bool rexit, xstring i_target) :
    m_rexit(rexit), m_traverse_target(i_target) {
    m_target = m_traverse_target;
    this->set_target();
}

File_Names::File_Names(bool rexit, xstring i_old, xstring i_target) :
    m_rexit(rexit), m_old(i_old), m_traverse_target(i_target) {
    m_target = m_traverse_target;
    this->set_old();
    this->set_target();
}


// syntax to differentiate folder from file is that folders end with a '/' or a '\\'

xstring File_Names::fix_slash(xstring& item) {
#if defined(WIN_BASE)
    item = item.sub("/", "\\\\");
    char full[_MAX_PATH];
    _fullpath(full, item.c_str(), _MAX_PATH);
    item = full;

#elif defined(NIX_BASE)
    item = item.sub("\\\\", "/");

    char resolved_path[PATH_MAX];
    realpath(item.c_str(), resolved_path);
    item = resolved_path;
#endif
    return item;
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
    m_old = this->fix_slash(m_old);
}

void File_Names::set_target() {
    if (m_rexit)
        this->assert_folder_syntax(m_target);
    m_target = this->fix_slash(m_target);
}

void File_Names::set_old(xstring& i_old) {
    m_old = i_old;
    if (m_rexit)
        this->assert_folder_syntax(m_old);
    m_old = this->fix_slash(m_old);
}

void File_Names::set_target(xstring& i_target) {
    m_target = i_target;
    m_traverse_target = i_target;
    if (m_rexit)
        this->assert_folder_syntax(m_target);
    m_target = this->fix_slash(m_target);
}

void File_Names::imaginary_path(){
    if(m_traverse_target[0] == '.'){
#if defined(NIX_BASE)
        m_target = this->pwd() + '/' + m_traverse_target.sub(R"(^[\.\\/]+)", "");
#elif defined(WIN_BASE)
        m_target = this->pwd() + '\\' + m_traverse_target.sub(R"(^[\.\\/]+)", "");
#endif
    }else{
        m_target = m_traverse_target;
    }
}

xstring File_Names::old() { return m_old; }
xstring File_Names::target() { return m_target; }
xstring File_Names::traverse_target() { return m_traverse_target; }
