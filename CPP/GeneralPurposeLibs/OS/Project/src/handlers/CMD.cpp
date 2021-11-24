#include "handlers/CMD.h"

RA::OS_O::CMD::CMD()
{
}

RA::OS_O::CMD::CMD(const CMD& cmd)
{
    *this = cmd;
}

void RA::OS_O::CMD::operator=(const CMD& cmd)
{
    m_cmd = cmd.m_cmd;
    m_out = cmd.m_out;
    m_err = cmd.m_err;
    m_err_message = cmd.m_err_message;
}

xstring RA::OS_O::CMD::GetCommand() const
{
    return m_cmd;
}

xstring RA::OS_O::CMD::GetOutput() const
{
    return m_out;
}

xstring RA::OS_O::CMD::GetError() const
{
    return m_err;
}

xstring RA::OS_O::CMD::GetErrorMessage() const
{
    return m_err_message;
}
