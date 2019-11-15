#include "xstring.h"

class OS;
namespace OS_O // OS Object
{
    class CMD
    {
        friend class ::OS;

        xstring m_cmd;
        xstring m_out;
        xstring m_err;
        xstring m_err_message;

    public:
        CMD();
        CMD(const CMD& cmd);
        void operator=(const CMD& cmd);
        xstring cmd() const;
        xstring out() const;
        xstring err() const;
        xstring err_message() const;
    };
};