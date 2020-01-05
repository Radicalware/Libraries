
#include "BaseRex.h"
#include <string_view>
#include "re2/re2.h"

int main()
{
    std::string original = "XXword1ZZ XXword2ZZ";
    std::string_view view = original;
    re2::StringPiece data = view.data();

    std::string str;

    RE2::FindAndConsume(&data, R"((?:XX)(\w+)(?:ZZ))", &str);
    cout << str << endl;

    RE2::FindAndConsume(&data, R"((?:XX)(\w+)(?:ZZ))", &str);
    cout << str << endl;

    cout << "orig: " << original << endl;
    cout << "view: " << view << endl;
    cout << "data: " << data << endl;



    return 0;
}
