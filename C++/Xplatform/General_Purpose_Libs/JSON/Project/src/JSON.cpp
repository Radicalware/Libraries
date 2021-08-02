#include "JSON.h"

#include <nlohmann/json.hpp>

#define ThrowNoJSON() \
    if (!MoFullJson.get()) throw "No JSON";

void RA::JSON::Clear()
{
    MoBsonValue.reset();
    MoFullJson.reset();
    MoZoomedJson.reset();
}

RA::JSON::~JSON()
{
    Clear();
}

RA::JSON::JSON(const xstring& FoString, Init FeInit)
{
    Set(FoString, FeInit);
}

RA::JSON::JSON(const BSON::Value& FoBSON, Init FeInit)
{
    Set(FoBSON, FeInit);
}

RA::JSON::JSON(const web::json::value& FoJson, Init FeInit)
{
    Set(FoJson, FeInit);
}

void RA::JSON::SetJSON(const xstring& FoString)
{
    utility::stringstream_t StringStream;
    xstring JsonStr = '[';
    JsonStr += FoString.Sub(R"((\$oid))", "oid");
    JsonStr += ']';
    StringStream << JsonStr.c_str();
    
    web::json::value JsonValue = web::json::value::parse(StringStream);
    if (!MoFullJson.get())
        MoFullJson = std::make_shared<web::json::value>(JsonValue);
    else
        *MoFullJson = JsonValue;
}

void RA::JSON::SetBSON(const BSON::Value& FoBSON)
{
    if (!MoBsonValue.get())
        MoBsonValue = std::make_shared<BSON::Value>(FoBSON);
    else
        *MoBsonValue = FoBSON;
}

void RA::JSON::Set(const xstring& FoString, Init FeInit)
{
    if(FeInit == Init::Both || FeInit == Init::JSON) // not exclusive to only JSON
        SetJSON(FoString);
    
    SetBSON(bsoncxx::from_json(FoString.c_str()));
}

void RA::JSON::Set(const BSON::Value& FoBSON, Init FeInit)
{
    if (FeInit == Init::Default)
        FeInit = Init::Both;

    if (FeInit == Init::Both || FeInit == Init::BSON)  // If not exclusivly only JSON
        SetBSON(FoBSON);

    if (FeInit == Init::Both || FeInit == Init::JSON)
        SetJSON(bsoncxx::to_json(FoBSON));
}

void RA::JSON::Set(const web::json::value& FoJson, Init FeInit)
{
    if (FeInit == Init::Default)
        FeInit = Init::Both;

    if (FeInit == Init::Both || FeInit == Init::JSON) // If not exclusivly only BSON
    {
        if (!MoFullJson.get())
            MoFullJson = std::make_shared<web::json::value>(FoJson);
        else
            *MoFullJson = FoJson;
    }

    if(FeInit == Init::Both || FeInit == Init::BSON)
        SetBSON(bsoncxx::from_json(GetSingleLineJson().c_str()));
}

xstring RA::JSON::GetPrettyJson(const int FnSpaceCount, const char FcIndentChar, const bool FbEnsureAscii) const
{
    ThrowNoJSON();
    const web::json::value& JsonTarget = (MoZoomedJson) ? *MoZoomedJson : *MoFullJson;
    return nlohmann::json::parse(WTXS(JsonTarget.serialize().c_str()).c_str()).dump(FnSpaceCount, FcIndentChar, FbEnsureAscii).c_str();
}

xstring RA::JSON::GetSingleLineJson() const
{
    ThrowNoJSON();
    return nlohmann::json::parse(WTXS(GetZoomedObject().serialize().c_str()).c_str()).dump(4, ' ', true).c_str();
}

void RA::JSON::PrintJson() const
{
    ThrowNoJSON();
    std::cout << nlohmann::json::parse(WTXS(GetZoomedObject().serialize().c_str()).c_str()).dump(4, ' ', true).c_str() << std::endl;
}

bool RA::JSON::Has(const xstring& FsObjectName) const
{
    if (!MoFullJson.get())
        throw "No JSON";
    return MoFullJson.get()->has_field(FsObjectName.ToStdWString());
}

bool RA::JSON::Has(const wchar_t* FsObjectName) const
{
    if (!MoFullJson.get())
        throw "No JSON";
    return MoFullJson.get()->has_field(FsObjectName);
}

RA::JSON& RA::JSON::Zoom(const char* FacObject)
{
    ThrowNoJSON();
    size_t Size = strlen(FacObject) + 1;
    wchar_t* LacWideChar = new wchar_t[Size];
    mbstowcs_s(NULL, LacWideChar, strlen(FacObject) + 1, FacObject, strlen(FacObject));
    LacWideChar[Size - 1] = L'\0';
    Zoom(LacWideChar);
    delete[] LacWideChar;
    return *this;
}

RA::JSON& RA::JSON::Zoom(const xstring& FsObject)
{
    ThrowNoJSON();
    return Zoom(FsObject.c_str());
}

RA::JSON& RA::JSON::Zoom(const wchar_t* FacObject)
{
    ThrowNoJSON();

    if (!MoZoomedJson.get())
    {
        MoZoomedJson = std::make_shared<web::json::value>();
        *MoZoomedJson = *MoFullJson;
    }
 
    try {
        *MoZoomedJson = MoZoomedJson->at(FacObject);
    }
    catch (...)
    {
        PrintJson();
        std::cout << "\n\n FAILED!! \n\n";
        exit(1);
    }
    return *this;
}

RA::JSON& RA::JSON::ZoomReset()
{
    ThrowNoJSON();
    MoZoomedJson.reset();
    return *this;
}

const web::json::value& RA::JSON::GetFullObject() const
{
    ThrowNoJSON();
    return *MoFullJson;
}

const web::json::value& RA::JSON::GetJson() const
{
    ThrowNoJSON();
    return GetZoomedObject();
}

const web::json::array& RA::JSON::GetJsonArray() const
{
    ThrowNoJSON();
    return GetZoomedObject().as_array();
}

const web::json::object& RA::JSON::GetJsonObject() const
{
    ThrowNoJSON();
    return GetZoomedObject().as_object();
}

const web::json::value& RA::JSON::GetZoomedObject() const
{
    ThrowNoJSON();
    if (MoZoomedJson.get())
        return *MoZoomedJson.get();
    else
        return *MoFullJson;
}
