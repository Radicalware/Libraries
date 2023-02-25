#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/err.h>
#include <string.h>   

#include "Macros.h"
#include "AES.h"

using std::cout;
using std::endl;


RA::AES Encryption()
{
    Begin();
    xstring Plaintext = "The is my secret message, don't abuse it!";
    RA::AES EncryptedAES;
    EncryptedAES.SetPlainText(Plaintext);
    EncryptedAES.SetAllRandomValues();
    EncryptedAES.Encrypt();
    return EncryptedAES;
    Rescue();
}

RA::AES Decryption(const RA::AES& FoEncryptedAES)
{
    Begin();
    RA::AES Decryption;
    // note: this is FooBar, you could use operator=(Other)
    Decryption.SetCipherText(FoEncryptedAES.GetCipherText());
    Decryption.SetAAD(FoEncryptedAES.GetAAD());
    Decryption.SetKey(FoEncryptedAES.GetKey());
    Decryption.SetIV(FoEncryptedAES.GetIV());
    Decryption.SetTag(FoEncryptedAES.GetTag());
    Decryption.Decrypt();
    return Decryption;
    Rescue();
}

int main(int arc, char *argv[])
{
    Begin();
    Nexus<>::Start();

    auto EncryptedAES = Encryption();
    EncryptedAES.GetCipherText().ToByteCode().Print("\n\n");

    auto DecryptedAES = Decryption(EncryptedAES);
    DecryptedAES.GetPlainText().Print("\n\n");

    RescuePrint();
    return Nexus<>::Stop();
}

