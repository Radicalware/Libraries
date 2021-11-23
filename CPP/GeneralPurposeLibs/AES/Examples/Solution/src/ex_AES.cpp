#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/err.h>
#include <string.h>   

#include "Macros.h"
#include "AES.h"

using std::cout;
using std::endl;

int main(int arc, char *argv[])
{
    Begin();
    Nexus<>::Start();

    xstring Plaintext;
    Plaintext  = "This is my secret message, don't abuse it!";

    cout << Plaintext.size() << endl;

    {
        RA::AES Encryption;
        Encryption.SetPlainText(Plaintext);
        //Encryption.SetAllRandomValues();
        Encryption.SetAAD("1234567812345678111");
        Encryption.SetKey("12345678123456781234567812345678");
        Encryption.SetIV("123456781234335678");
        Encryption.Encrypt();
        Encryption.GetCipherText().ToByteString().Print("\n\n");
        Encryption.Decrypt().Print();
    }

    {
        RA::AES Encryption;
        xstring ByteStr = "\\x32\\x56\\x5D\\xBC\\x95\\xDC\\x44\\xEC\\x03\\x08\\x35\\x40\\xF1\\x33\\x36\\xC0\\x9E\\xE5\\x9E\\x47\\x76\\x4C\\xF3\\xAC\\x96\\xAE\\x3C\\xDF\\xCD\\x2F\\x91\\x74\\x4B\\x6F\\x73\\xD0\\xE1\\xB5\\xE1\\xB5\\x55\\xA0";
        Encryption.SetCipherText(ByteStr.FromByteStringToASCII());
        Encryption.SetAAD("1234567812345678111");
        Encryption.SetKey("12345678123456781234567812345678");
        Encryption.SetIV("123456781234335678");
        Encryption.SetTag("\xE5\xCD\x80\xF4\xA3");
        Encryption.Decrypt().Print();

    }

    RescuePrint();
    return Nexus<>::Stop();
}

