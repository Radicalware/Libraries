#include "handlers/CMD.h"

OS_O::CMD::CMD()
{
}

OS_O::CMD::CMD(const CMD& cmd)
{
    *this = cmd;
}

void OS_O::CMD::operator=(const CMD& cmd)
{
    m_cmd = cmd.m_cmd;
    m_out = cmd.m_out;
    m_err = cmd.m_err;
    m_err_message = cmd.m_err_message;
}

xstring OS_O::CMD::cmd() const
{
    return m_cmd;
}

xstring OS_O::CMD::out() const
{
    return m_out;
}

xstring OS_O::CMD::err() const
{
    return m_err;
}

xstring OS_O::CMD::err_message() const
{
    return m_err_message;
}
