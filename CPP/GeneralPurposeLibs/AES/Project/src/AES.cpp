
#include "AES.h"
#include "xstring.h"

#include <chrono>
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/err.h>

RA::AES::AES(const xint FnEncryptionSize)
{
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();
    MnEncryptionSize = FnEncryptionSize;
}

RA::AES::~AES()
{
}

RA::AES::AES(const RA::AES& Other)
{
    The = Other;
}

RA::AES::AES(RA::AES&& Other) noexcept
{
    The = Other;
}

void RA::AES::operator=(const RA::AES& Other)
{
    // MsTagPtr     = sxp<unsigned char*>(Other.MsTagPtr.get());
    MsTagPtr.Clone(Other.MsTagPtr);

    MsPlaintext  = Other.MsPlaintext;
    MsCipherText = Other.MsCipherText;
    MsAAD        = Other.MsAAD;
    MsKey        = Other.MsKey;
    MsIV         = Other.MsIV;
    MsTag        = Other.MsTag;

    MnEncryptionSize = Other.MnEncryptionSize;
    MnCipherTextSize = Other.MnCipherTextSize;
}

void RA::AES::operator=(RA::AES&& Other) noexcept
{
    MsTagPtr     = Other.MsTagPtr;

    MsPlaintext  = std::move(Other.MsPlaintext);
    MsCipherText = std::move(Other.MsCipherText);
    MsAAD        = std::move(Other.MsAAD);
    MsKey        = std::move(Other.MsKey);
    MsIV         = std::move(Other.MsIV);
    MsTag        = std::move(Other.MsTag);

    MnEncryptionSize = Other.MnEncryptionSize;
    MnCipherTextSize = Other.MnCipherTextSize;
}

RA::AES& RA::AES::Encrypt()
{
    Begin();
    // Loosely based on https://stackoverflow.com/users/608639/jww methods

    if (MsPlaintext.Size() > MnEncryptionSize)
        MnEncryptionSize = MsPlaintext.Size();

    if (32 > MnEncryptionSize)
        MnEncryptionSize = 32;

    if (MnEncryptionSize % 2 != 0)
        MnEncryptionSize++;

    if (!MsPlaintext.Size())
        ThrowIt("You have no text to encrypt!");

    if (!MsAAD.Size() || !MsKey.Size() || !MsIV.Size())
        ThrowIt("Your AES Encryption Is Missing Values");

    auto AADPtr         = MsAAD.ToUnsignedChar();       auto AAD = AADPtr.Raw();                auto AADLen = MsAAD.Size();
    auto KeyPtr         = MsKey.ToUnsignedChar();       auto Key = KeyPtr.Raw();                auto KeyLen = MsKey.Size();
    auto IVPtr          = MsIV.ToUnsignedChar();        auto IV  = IVPtr.Raw();                 auto IvLen  = MsIV.Size();
    auto PlaintextPtr   = MsPlaintext.ToUnsignedChar(); auto PlainText = PlaintextPtr.Raw();    auto PlainTextLen = MsPlaintext.Size();

    auto CipherTextPtr = RA::SharedPtr<unsigned char[]>(MnEncryptionSize * 2);
    auto CipherText = CipherTextPtr.Raw();

    const xint TagLen = MnEncryptionSize / 8;
    MsTagPtr = RA::SharedPtr<unsigned char[]>(TagLen);
    auto Tag    = MsTagPtr.Raw();

    EVP_CIPHER_CTX* Context = nullptr;
    int Len = 0;

    /* Create and initialise the context */
    if (!(Context = EVP_CIPHER_CTX_new())) 
        The.ThrowErrors();
    /* Initialise the encryption operation. */
    if (1 != EVP_EncryptInit_ex(Context, EVP_aes_256_gcm(), nullptr, nullptr, nullptr))
        The.ThrowErrors();
    /* Set IV length if default 12 bytes (96 bits) is not appropriate */
    if (1 != EVP_CIPHER_CTX_ctrl(Context, EVP_CTRL_GCM_SET_IVLEN, IvLen, nullptr))
        The.ThrowErrors();
    /* Initialise Key and IV */
    if (1 != EVP_EncryptInit_ex(Context, nullptr, nullptr, Key, IV)) 
        The.ThrowErrors();
    /* Provide any AAD data. The can be called zero or more times as required*/
    if (AAD && AADLen > 0)
    {
        if (1 != EVP_EncryptUpdate(Context, nullptr, &Len, AAD, AADLen))
            The.ThrowErrors();
    }
    /* Provide the message to be encrypted, and obtain the encrypted output.
     * EVP_EncryptUpdate can be called multiple times if necessary */
    if (PlainText)
    {
        if (1 != EVP_EncryptUpdate(Context, CipherText, &Len, PlainText, PlainTextLen))
            The.ThrowErrors();

        MnCipherTextSize = Len;
    }
    /* Finalise the encryption. Normally CipherText bytes may be written at
     * this stage, but this does not occur in GCM mode */
    if (1 != EVP_EncryptFinal_ex(Context, CipherText + Len, &Len)) 
        The.ThrowErrors();
    MnCipherTextSize += Len;

    /* Get the Tag */
    if (1 != EVP_CIPHER_CTX_ctrl(Context, EVP_CTRL_GCM_GET_TAG, TagLen, Tag))
        The.ThrowErrors();

    /* Clean up */
    EVP_CIPHER_CTX_free(Context);

    MsTag = xstring(MsTagPtr.Raw(), TagLen);
    MsCipherText = xstring(CipherTextPtr.Raw(), MnEncryptionSize);

    return The;
    Rescue();
}

xstring RA::AES::Decrypt()
{
    Begin();
    // Loosely based on https://stackoverflow.com/users/608639/jww methods

    auto AADPtr         = MsAAD.ToUnsignedChar();   auto AAD = AADPtr.Raw();    auto AADLen = MsAAD.Size();
    auto KeyPtr         = MsKey.ToUnsignedChar();   auto Key = KeyPtr.Raw();    auto KeyLen = MsKey.Size();
    auto IVPtr          = MsIV.ToUnsignedChar();    auto IV  = IVPtr.Raw(); auto IvLen  = MsIV.Size();

    auto CipherTextPtr = MsCipherText.ToUnsignedChar();
    auto CipherText = CipherTextPtr.Raw();


    EVP_CIPHER_CTX* Context = NULL;
    int Len = 0, PlainTextLen = 0, ret = 0;

    /* Create and initialise the context */
    if (!(Context = EVP_CIPHER_CTX_new())) 
        The.ThrowErrors();

    /* Initialise the decryption operation. */
    if (!EVP_DecryptInit_ex(Context, EVP_aes_256_gcm(), NULL, NULL, NULL))
        The.ThrowErrors();

    /* Set IV length. Not necessary if this is 12 bytes (96 bits) */
    if (!EVP_CIPHER_CTX_ctrl(Context, EVP_CTRL_GCM_SET_IVLEN, IvLen, NULL))
        The.ThrowErrors();

    /* Initialise Key and IV */
    if (!EVP_DecryptInit_ex(Context, NULL, NULL, Key, IV)) 
        The.ThrowErrors();

    /* Provide any AAD data. The can be called zero or more times as
     * required */
    if (AAD && AADLen > 0)
    {
        if (!EVP_DecryptUpdate(Context, NULL, &Len, AAD, AADLen))
            The.ThrowErrors();
    }

    /* Provide the message to be decrypted, and obtain the PlainText output.
     * EVP_DecryptUpdate can be called multiple times if necessary */
    if (!CipherText)
        ThrowIt("No Cipher Text");

    auto MsDecryptedTextPtr = RA::SharedPtr<unsigned char[]>(MnCipherTextSize * 2);
    if (!EVP_DecryptUpdate(Context, MsDecryptedTextPtr.Raw(), &Len, CipherText, MnCipherTextSize))
        The.ThrowErrors();
    PlainTextLen = Len;

    /* Set expected Tag value. Works in OpenSSL 1.0.1d and later */
    MsTagPtr = MsTag.ToUnsignedChar();
    if (!EVP_CIPHER_CTX_ctrl(Context, EVP_CTRL_GCM_SET_TAG, MsTag.Size(), MsTagPtr.Raw()))
        The.ThrowErrors();

    /* Finalise the decryption. A positive return value indicates success,
     * anything else is a failure - the PlainText is not trustworthy.*/
    ret = EVP_DecryptFinal_ex(Context, MsDecryptedTextPtr.Raw() + Len, &Len);

    /* Clean up */
    EVP_CIPHER_CTX_free(Context);

    if (ret > 0)
        PlainTextLen += Len;
    else
        ThrowIt("Failed to Decrypt");

    MsPlaintext = xstring(MsDecryptedTextPtr.Raw(), PlainTextLen);
    return MsPlaintext;

    Rescue();
}

RA::AES& RA::AES::SetPlainText(const xstring& FsPlainText)
{
    if (FsPlainText.Size() > 128)
        ThrowIt("The string exceeds the 128 Byte Limit");
    MsPlaintext        = FsPlainText;
    MnEncryptionSize   = MsPlaintext.Size();
    return The;
}

RA::AES& RA::AES::SetCipherText(const xstring& FsCipherText)
{
    MsCipherText     = FsCipherText;
    MnCipherTextSize = MsCipherText.Size();
    return The;
}

RA::AES& RA::AES::SetAllRandomValues()
{
    The.SetRandomAAD();
    The.SetRandomKey();
    The.SetRandomIV();
    return The;
}

void RA::AES::SetRandomAAD()
{
    Begin();
    if (!MnEncryptionSize)
        ThrowIt("Encryption Size is Zero");
    MsAAD = RA::GetRandomStr(16, 48, 57); // INT 48 - 57 = ASCII 0 - 9
    Rescue();
}

void RA::AES::SetRandomKey()
{
    Begin();
    if (!MnEncryptionSize)
        ThrowIt("Encryption Size is Zero");
    MsKey = RA::GetRandomStr(32, 48, 57);
    Rescue();
}

void RA::AES::SetRandomIV()
{
    Begin();
    if (!MnEncryptionSize)
        ThrowIt("Encryption Size is Zero");
    MsIV = RA::GetRandomStr(16, 48, 57);
    Rescue();
}

void RA::AES::ThrowErrors() const
{
    Begin();
    xstring ErrorStr;
    while (unsigned long errCode = ERR_get_error())
        ErrorStr += ERR_error_string(errCode, nullptr) + '\n';
    ThrowIt(ErrorStr);
    Rescue();
}
