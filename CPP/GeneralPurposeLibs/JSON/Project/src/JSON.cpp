#include "JSON.h"
#include "Macros.h"
#include <nlohmann/json.hpp>
#include <cpprest/json.h>

void RA::JSON::Print(const web::json::value& FoWebJson)
{
    Begin();
    if (FoWebJson.is_null())
    {
        cout << "{ null }\n";
        return;
    }
    std::cout << nlohmann::json::parse(WTXS(FoWebJson.serialize().c_str()).c_str()).dump(4, ' ', true).c_str() << std::endl;
    RescueThrow();
}

void RA::JSON::Print(const RA::JSON& FoWebJson)
{
    Print(SCast<web::json::value>(FoWebJson));
}

RA::JSON::~JSON()
{
}

RA::JSON::JSON(const xstring& FoString) : value(web::json::value::object())
{
    if (!FoString.Size())
        ThrowIt("No Input");
    Set(FoString);
}

RA::JSON::JSON(const BSON::Value& FoBSON) : value(web::json::value::object())
{
    Set(FoBSON);
}

RA::JSON::JSON(const web::json::value& FoJson) : value(web::json::value::object())
{
    Set(FoJson);
}

void RA::JSON::operator=(const RA::JSON& Other)
{
    if (Other.is_null())
        web::json::value::operator=(web::json::value::null());
    else if (Other.is_boolean())
        web::json::value::operator=(web::json::value::boolean(Other.as_bool()));
    else if (Other.is_double())
        web::json::value::operator=(web::json::value::number(Other.as_double()));
    else if (Other.is_integer())
        web::json::value::operator=(web::json::value::number(Other.as_integer()));
    else if (Other.is_string())
        web::json::value::operator=(web::json::value::string(Other.as_string()));
    else
        web::json::value::operator=(static_cast<const web::json::value&>(Other));
}

void RA::JSON::operator=(JSON&& Other) noexcept
{
    if (Other.is_null())
        web::json::value::operator=(web::json::value::null());
    else if (Other.is_boolean())
        web::json::value::operator=(web::json::value::boolean(Other.as_bool()));
    else if (Other.is_double())
        web::json::value::operator=(web::json::value::number(Other.as_double()));
    else if (Other.is_integer())
        web::json::value::operator=(web::json::value::number(Other.as_integer()));
    else if (Other.is_string())
        web::json::value::operator=(web::json::value::string(Other.as_string()));
    else
        web::json::value::operator=(static_cast<web::json::value&&>(Other));
}

RA::JSON& RA::JSON::operator[](const std::wstring& FsKey)
{
    return static_cast<RA::JSON&>(web::json::value::operator[](FsKey));
}

RA::JSON& RA::JSON::operator[](const wchar_t* FsKey)
{
    return static_cast<RA::JSON&>(web::json::value::operator[](std::wstring(FsKey)));
}

RA::JSON& RA::JSON::operator=(const std::wstring& FsValue)
{
    web::json::value::operator=(web::json::value::string(FsValue));
    return The;
}

RA::JSON& RA::JSON::operator=(const wchar_t* FsValue)
{
    return The = std::wstring(FsValue);
}

RA::BSON::Value RA::JSON::ToBSON(const xstring& FsObject)
{
    return bsoncxx::from_json(FsObject.c_str());
}

RA::BSON::Value RA::JSON::ToBSON() const
{
    return ToBSON(GetSingleLineJson());
}

void RA::JSON::Set(xstring&& FoString)
{
    Begin();
    if (FoString.Last() == ',')
    {
        FoString.insert(FoString.begin(), '[');
        FoString.Last() = ']';
    }
    The = web::json::value::parse(SCast<std::string>(FoString.InRemove('$')));
    RescueThrow([&FoString]() { RA::JSON(FoString).GetPrettyJson().Print(); });
}

void RA::JSON::Set(const xstring& FoString)
{
    Begin();
    auto LoString = FoString;
    if (LoString.Last() == ',')
    {
        LoString.insert(LoString.begin(), '[');
        LoString.Last() = ']';
    }
    The = web::json::value::parse(SCast<std::string>(LoString.InRemove('$')));
    RescueThrow([&FoString]() { RA::JSON(FoString).GetPrettyJson().Print(); });
}

void RA::JSON::Set(const BSON::Value& FoBSON)
{
    Begin();
    Set(FoBSON.view());
    RescueThrow([this]() { GetPrettyJson().Print(); });
}

void RA::JSON::Set(const BSON::View& FoBSON)
{
    Begin();
    The = BSONViewToJson(FoBSON);
    RescueThrow([this]() { GetPrettyJson().Print(); });
}

void RA::JSON::Set(const web::json::value& FoJson)
{
    Begin();
    if (FoJson.is_null())
        return;
    web::json::value::operator=(FoJson);
    RescueThrow([this]() { GetPrettyJson().Print(); });
}

template<typename T>
web::json::value RA::JSON::BSONElementToJson(const T& FoElem)
{
    switch (FoElem.type())
    {
    case bsoncxx::type::k_string:
        return web::json::value::string(
            utility::conversions::to_string_t(std::string(FoElem.get_string().value)));
    case bsoncxx::type::k_int32:
        return web::json::value::number(FoElem.get_int32().value);
    case bsoncxx::type::k_int64:
        return web::json::value::number(FoElem.get_int64().value);
    case bsoncxx::type::k_double:
        return web::json::value::number(FoElem.get_double().value);
    case bsoncxx::type::k_bool:
        return web::json::value::boolean(FoElem.get_bool().value);
    case bsoncxx::type::k_document:
        return BSONViewToJson(FoElem.get_document().value);
    case bsoncxx::type::k_array:
    {
        web::json::value LoArr = web::json::value::array();
        int LnIdx = 0;
        for (const auto& LoItem : FoElem.get_array().value)
            LoArr[LnIdx++] = BSONElementToJson(LoItem);
        return LoArr;
    }
    // ... handle k_oid, k_date, k_null, etc. as needed
    default:
        return web::json::value::null();
    }
}

web::json::value RA::JSON::BSONViewToJson(const BSON::View& FoBSON)
{
    web::json::value LoResult = web::json::value::object();
    for (const auto& LoElem : FoBSON)
    {
        auto LoKey = utility::conversions::to_string_t(std::string(LoElem.key()));
        LoResult[LoKey] = BSONElementToJson(LoElem);
    }
    return LoResult;
}

xstring RA::JSON::GetPrettyJson(const int FnSpaceCount, const char FcIndentChar, const bool FbEnsureAscii) const
{
    Begin();
    if (The.is_null())
        return "{ null }\n";
    return nlohmann::json::parse(WTXS(serialize().c_str()).c_str())
        .dump(FnSpaceCount, FcIndentChar, FbEnsureAscii).c_str();
    RescueThrow([this]() { GetPrettyJson().Print(); });
}

xstring RA::JSON::GetSingleLineJson() const
{
    Begin();
    if (The.is_null())
        return "{ null }\n";
    return nlohmann::json::parse(WTXS(serialize().c_str()).c_str()).dump(4, ' ', true).c_str();
    RescueThrow([this]() { GetPrettyJson().Print(); });
}

void RA::JSON::PrintJson() const
{
    Begin();
    GetPrettyJson().Print();
    RescueThrow();
}

RA::JSON& RA::JSON::At(const xint Idx, const bool FbSilentError)
{
    Begin();
    if (!is_array())
        ThrowIt("JSON is not an array");
    return SCast<JSON&>(at(Idx));
    RescueThrow([&]() { if (!FbSilentError) GetPrettyJson().Print(); });
}

const RA::JSON& RA::JSON::At(const xint Idx, const bool FbSilentError) const
{
    return const_cast<JSON*>(this)->At(Idx, FbSilentError);
}

RA::JSON& RA::JSON::KeyFC(const char* FsKey, const bool FbSilentError)
{
    Begin();
    size_t Size = strlen(FsKey) + 1;
    std::wstring LsWide(Size - 1, L'\0');
    for (size_t i = 0; i < Size - 1; ++i)
        LsWide[i] = static_cast<wchar_t>(static_cast<unsigned char>(FsKey[i]));
    Key(LsWide);
    return The;
    RescueThrow([&]() { if (!FbSilentError) GetPrettyJson().Print(); });
}

RA::JSON& RA::JSON::KeyFC(const xstring& FsKey, const bool FbSilentError)
{
    Begin();
    return Key(FsKey.ToStdWString());
    RescueThrow([&]() { if (!FbSilentError) GetPrettyJson().Print(); });
}

RA::JSON& RA::JSON::Key(const std::wstring& FsKey, const bool FbSilentError)
{
    Begin();
    
    if (is_array())
    {
        if (size() == 1) {
            if (at(0).is_object())
                return SCast<JSON&>(at(0).as_object()[FsKey]);
            ThrowIt("Index 0 has no object");
        }
        ThrowIt("JSON is an array, use At() instead of Key()");
    }
    else if (is_object())
    {
        if (has_field(FsKey))
            return SCast<JSON&>(as_object()[FsKey]);
        ThrowIt("JSON Has No Key: ", FsKey);
    }
    ThrowIt("JSON is not an object"); /// todo: enum map for The.type();

    RescueThrow([&]() { if (!FbSilentError) GetPrettyJson().Print(); });
}

const RA::JSON& RA::JSON::KeyFC(const char* FsKey, const bool FbSilentError) const
{
    return const_cast<JSON*>(this)->KeyFC(FsKey, FbSilentError);
}

const RA::JSON& RA::JSON::KeyFC(const xstring& FsKey, const bool FbSilentError) const
{
    return const_cast<JSON*>(this)->KeyFC(FsKey, FbSilentError);
}

const RA::JSON& RA::JSON::Key(const std::wstring& FsKey, const bool FbSilentError) const
{
    return const_cast<JSON*>(this)->Key(FsKey, FbSilentError);
}

bool RA::JSON::BxNullFC(const char* FsKey) const
{
    Begin();
    return KeyFC(FsKey).is_null();
    Rescue();
}

bool RA::JSON::BxNullFC(const xstring& FsKey) const
{
    Begin();
    return Key(FsKey.ToStdWString()).is_null();
    Rescue();
}

bool RA::JSON::BxNull() const
{
    Begin();
    return is_null();
    Rescue();
}

bool RA::JSON::BxNull(const std::wstring& FsKey) const
{
    Begin();
    return Key(FsKey).is_null();
    Rescue();
}

xstring RA::JSON::ToString() const
{
    if (is_string())
        return RA::WTXS(as_string().c_str());
    return RA::WTXS(to_string().c_str());
}

xstring RA::JSON::ToString(const web::json::value& FwObject)
{
    if (FwObject.is_string())
        return RA::WTXS(FwObject.as_string().c_str());
    return RA::WTXS(FwObject.serialize().c_str());
}
