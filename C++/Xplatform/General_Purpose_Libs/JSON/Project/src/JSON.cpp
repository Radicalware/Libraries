#include "JSON.h"
#include "Macros.h"
#include <nlohmann/json.hpp>

#define ThrowNoJSON() \
    if (!MoFullJsonPtr.get())  ThrowIt("No JSON");

#define ThrowNoBSON() \
    if (!MoBsonValuePtr.get()) ThrowIt("No BSON");

void RA::JSON::Print(const web::json::value & FoWebJson)
{
    Begin();
    std::cout << nlohmann::json::parse(WTXS(FoWebJson.serialize().c_str()).c_str()).dump(4, ' ', true).c_str() << std::endl;
    RescueThrow();
}

void RA::JSON::Clear()
{
    MoBsonValuePtr.reset();
    MoFullJsonPtr.reset();
    MoZoomedJsonPtr.reset();
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
    Begin();
    utility::stringstream_t StringStream;
    xstring JsonStr = '[';
    JsonStr += FoString.Sub(R"((\$oid))", "oid");
    JsonStr += ']';
    StringStream << JsonStr.c_str();
    
    web::json::value JsonValue = web::json::value::parse(StringStream);
    if (!MoFullJsonPtr.get())
        MoFullJsonPtr = std::make_shared<web::json::value>(JsonValue);
    else
        *MoFullJsonPtr = JsonValue;
    ZoomReset();
    RescueThrow();
}

void RA::JSON::SetBSON(const BSON::Value& FoBSON)
{
    Begin();
    if (!MoBsonValuePtr.get())
        MoBsonValuePtr = std::make_shared<BSON::Value>(FoBSON);
    else
        *MoBsonValuePtr = FoBSON;
    ZoomReset();
    RescueThrow();
}

void RA::JSON::Set(const xstring& FoString, Init FeInit)
{
    Begin();
    if(FeInit == Init::Both || FeInit == Init::JSON) // not exclusive to only JSON
        SetJSON(FoString);
    
    SetBSON(bsoncxx::from_json(FoString.c_str()));
    RescueThrow();
}

void RA::JSON::Set(const BSON::Value& FoBSON, Init FeInit)
{
    Begin();
    if (FeInit == Init::Default)
        FeInit = Init::Both;

    if (FeInit == Init::Both || FeInit == Init::BSON)  // If not exclusivly only JSON
        SetBSON(FoBSON);

    if (FeInit == Init::Both || FeInit == Init::JSON)
        SetJSON(bsoncxx::to_json(FoBSON));
    RescueThrow();
}

void RA::JSON::Set(const web::json::value& FoJson, Init FeInit)
{
    Begin();
    if (FeInit == Init::Default)
        FeInit = Init::Both;

    if (FeInit == Init::Both || FeInit == Init::JSON) // If not exclusivly only BSON
    {
        if (!MoFullJsonPtr.get())
            MoFullJsonPtr = std::make_shared<web::json::value>(FoJson);
        else
            *MoFullJsonPtr = FoJson;
    }

    if(FeInit == Init::Both || FeInit == Init::BSON)
        SetBSON(bsoncxx::from_json(GetSingleLineJson().c_str()));
    ZoomReset();
    RescueThrow();
}

xstring RA::JSON::GetPrettyJson(const int FnSpaceCount, const char FcIndentChar, const bool FbEnsureAscii) const
{
    Begin();
    ThrowNoJSON();
    const web::json::value& JsonTarget = (MoZoomedJsonPtr) ? *MoZoomedJsonPtr : *MoFullJsonPtr;
    return nlohmann::json::parse(WTXS(JsonTarget.serialize().c_str()).c_str()).dump(FnSpaceCount, FcIndentChar, FbEnsureAscii).c_str();
    RescueThrow();
}

xstring RA::JSON::GetSingleLineJson() const
{
    Begin();
    ThrowNoJSON();
    return nlohmann::json::parse(WTXS(GetZoomedObject().serialize().c_str()).c_str()).dump(4, ' ', true).c_str();
    RescueThrow();
}

void RA::JSON::PrintJson() const
{
    Begin();
    ThrowNoJSON();
    std::cout << nlohmann::json::parse(WTXS(GetZoomedObject().serialize().c_str()).c_str()).dump(4, ' ', true).c_str() << std::endl;
    RescueThrow();
}

bool RA::JSON::Has(const xstring& FsObjectName) const
{
    Begin();
    GSS(MoFullJson);
    return MoFullJson.has_field(FsObjectName.ToStdWString());
    RescueThrow();
}

bool RA::JSON::Has(const wchar_t* FsObjectName) const
{
    Begin();
    GSS(MoFullJson);
    return MoFullJson.has_field(FsObjectName);
    RescueThrow();
}

uint RA::JSON::Size() const
{
    Begin();
    if (!MoFullJsonPtr && !MoBsonValuePtr) return 0;
    GSS(MoBsonValue);
    return MoBsonValue.view().length();
    RescueThrow();
}

RA::JSON& RA::JSON::Zoom(const char* FacObject)
{
    Begin();
    ThrowNoJSON();
    size_t Size = strlen(FacObject) + 1;
    wchar_t* LacWideChar = new wchar_t[Size];
    mbstowcs_s(NULL, LacWideChar, strlen(FacObject) + 1, FacObject, strlen(FacObject));
    LacWideChar[Size - 1] = L'\0';
    Zoom(LacWideChar);
    delete[] LacWideChar;
    return *this;
    RescueThrow();
}

RA::JSON& RA::JSON::Zoom(const xstring& FsObject)
{
    Begin();
    ThrowNoJSON();
    return Zoom(FsObject.c_str());
    RescueThrow();
}

RA::JSON& RA::JSON::Zoom(const wchar_t* FacObject)
{
    Begin();
    GSS(MoFullJson);
    try {

        if (!MoZoomedJsonPtr.get())
            MoZoomedJsonPtr = std::make_shared<web::json::value>(MoFullJson.at(FacObject));
        else {
            GSS(MoZoomedJson);
            auto Zoomed = MoZoomedJson.at(FacObject);
            MoZoomedJson = Zoomed;
        }
    }
    catch (...)
    {
        PrintJson();
        std::cout << "\n\n FAILED!! \n\n";
        exit(1);
    }
    return *this;
    RescueThrow();
}

RA::JSON& RA::JSON::ZoomReset()
{
    Begin();
    MoZoomedJsonPtr.reset();
    return *this;
    RescueThrow();
}

const web::json::value& RA::JSON::GetFullObject() const
{
    Begin();
    ThrowNoJSON();
    return *MoFullJsonPtr;
    RescueThrow();
}

const BSON::Value& RA::JSON::GetBSON() const
{
    Begin();
    ThrowNoBSON();
    return *MoBsonValuePtr;
    RescueThrow();
}

const web::json::value& RA::JSON::GetJSON() const
{
    Begin();
    ThrowNoJSON();
    return GetZoomedObject();
    RescueThrow();
}

const web::json::array& RA::JSON::GetJsonArray() const
{
    Begin();
    ThrowNoJSON();
    return GetZoomedObject().as_array();
    RescueThrow();
}

const web::json::object& RA::JSON::GetJsonObject() const
{
    Begin();
    ThrowNoJSON();
    return GetZoomedObject().as_object();
    RescueThrow();
}

const web::json::value& RA::JSON::GetZoomedObject() const
{
    Begin();
    ThrowNoJSON();
    if (MoZoomedJsonPtr.get())
        return *MoZoomedJsonPtr.get();
    else
        return *MoFullJsonPtr;
    RescueThrow();
}
