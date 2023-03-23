#include "handlers/File.h"


RE2 RA::OS_O::File::s_get_file(R"(^.*[\\/])");
RE2 RA::OS_O::File::s_forwardslash(R"(/)");

RA::OS_O::File::File()
{
}

void RA::OS_O::File::SetFile(const xstring& iname) {
    m_name = File::FullPath(iname);
}


RA::OS_O::File::File(const File& file)
{
    *this = file;
}

RA::OS_O::File::File(const xstring& iname)
{
    m_name = iname;
}

void RA::OS_O::File::operator=(const File& file)
{
    this->Close();
    m_name = file.m_name;
    m_data = file.m_data;
}

xstring RA::OS_O::File::GetName() const
{
    return m_name;
}

xstring RA::OS_O::File::GetData() const
{
    return m_data;
}

void RA::OS_O::File::SetRead()
{
    Begin();
    this->Close();
    m_handler = 'r';
    m_in_stream.open(m_name.c_str(), std::ios::binary | std::ios::in);
    Rescue();
}

void RA::OS_O::File::SetWrite()
{
    Begin();
    Close();
    m_handler = 'w';
    m_out_stream.open(
        m_name.c_str(),
        std::ios::binary | std::ios::in | std::ios::out | std::ofstream::trunc
    );
    Rescue();
}

void RA::OS_O::File::SetAppend()
{
    Begin();
    this->Close();
    m_handler = 'a';
    m_out_stream.open(
        m_name.c_str(),
        std::ios::binary | std::ios::in | std::ios::out | std::ios::ate
    );
    Rescue();
}

void RA::OS_O::File::Close()
{
    Begin();
    if (m_in_stream.is_open())
        m_in_stream.close();

    if (m_out_stream.is_open())
        m_out_stream.close();
    Rescue();
}

void RA::OS_O::File::Clear(){
    this->SetWrite();
}

void RA::OS_O::File::Remove(){
    this->RM();
}

void RA::OS_O::File::RM()
{
    this->Close();
    errno = 0;
    if(!OS_O::Dir_Type::HasFile(m_name))
        throw std::runtime_error(std::string("Error Filename: ") + m_name + " Does Not Exist!\n");
    try {
#if defined(BxWindows)
        DeleteFileA(m_name.c_str());
#elif defined(BxNix)
        remove(m_name.c_str());
#endif
        if (errno)
            throw;
    }
    catch (...) {
        xstring err = "Failed (" + RA::ToXString(errno) + "): Failed to delete file: '" + m_name + "'\n";
        throw std::runtime_error(err);
    }
}

void RA::OS_O::File::Copy(const xstring& location){
    this->CP(location);
}

void RA::OS_O::File::CP(const xstring& location)
{
    xstring new_path = FullPath(location);
    try {
        this->SetRead();
        std::ofstream out_stream(new_path, std::ios::out | std::ios::binary);
        out_stream << m_in_stream.rdbuf();
        if (out_stream.is_open()) out_stream.close();
    }
    catch (std::runtime_error & err) {
        throw std::runtime_error(xstring("OS::move_file Failed: ") + err.what());
    }
}

void RA::OS_O::File::Move(const xstring& location){
    this->MV(location);
}

void RA::OS_O::File::MV(const xstring& location)
{
    this->CP(location);
    this->RM();
}


void RA::OS_O::File::operator<<(const xstring& FnWriteData)
{
    m_out_stream << FnWriteData;
}