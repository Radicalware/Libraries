#include "JSON.h"
#include "Macros.h"
#include <nlohmann/json.hpp>
#include <cpprest/json.h>

#define ThrowNoJSON() \
    if (!MoFullJsonPtr.get())  ThrowIt("No JSON");

#define ThrowNoBSON() \
    if (!MoBsonValuePtr.get()) ThrowIt("No BSON");

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
    if (!FoString.Size())
        ThrowIt("No Input");
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

void RA::JSON::operator=(const RA::JSON& Other)
{
    MoBsonValuePtr.Clone(Other.MoBsonValuePtr);
    MoFullJsonPtr.Clone(Other.MoFullJsonPtr);
    MoZoomedJsonPtr.Clone(Other.MoZoomedJsonPtr);
}

void RA::JSON::operator=(JSON&& Other) noexcept
{
    MoBsonValuePtr  = std::move(Other.MoBsonValuePtr);
    MoFullJsonPtr   = std::move(Other.MoFullJsonPtr);
    MoZoomedJsonPtr = std::move(Other.MoZoomedJsonPtr);
}

void RA::JSON::SetJSON(const xstring& FoString)
{
    Begin();
    //static const RE2 LsOID(R"(\$oid)");
    //utility::stringstream_t StringStream;
    //StringStream << '[';
    //if (FoString.Has('$'))
    //    StringStream << FoString.Sub(LsOID, "oid").Ptr();
    //else
    //    StringStream << FoString.Ptr();
    //StringStream << ']';


    auto HandleIncommingStr = [this](const xstring& FFoString) ->void
    {
        if (FFoString.Has('$'))
        {
            if (!MoFullJsonPtr)
                MoFullJsonPtr = RA::MakeShared<web::json::value>(web::json::value::parse(FFoString.Remove('$').Ptr()));
            else
                *MoFullJsonPtr = web::json::value::parse(FFoString.Remove('$').Ptr());
        }
        else
        {
            if (!MoFullJsonPtr)
                MoFullJsonPtr = RA::MakeShared<web::json::value>(web::json::value::parse(FFoString.Ptr()));
            else
                *MoFullJsonPtr = web::json::value::parse(FFoString.Ptr());
        }
    };

    auto Last1 = FoString.Last();
    auto Last2 = FoString[FoString.size() - 1];
    if(Last1 != Last2)
        cout << "not equal: " << Last1 << " " << Last2 << endl;
    if (Last2 == ',')
    {
        xstring LoString = '[' + FoString;
        LoString.Last() = ']';
        HandleIncommingStr(LoString);
    }
    else
        HandleIncommingStr(FoString);

    MoZoomedJsonPtr = nullptr;
    RescueThrow([&FoString]() { RA::JSON(FoString).GetPrettyJson().Print(); });
}

void RA::JSON::SetBSON(const BSON::Value& FoBSON)
{
    Begin();
    if (MoBsonValuePtr == nullptr)
        MoBsonValuePtr = RA::MakeShared<BSON::Value>(FoBSON);
    else
        *MoBsonValuePtr = FoBSON;
    MoZoomedJsonPtr = nullptr;
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

void RA::JSON::Set(const xstring& FoString, Init FeInit)
{
    Begin();
    if (FoString == "null")
        return;

    if (FeInit == Init::Default)
    {
        SetJSON(FoString);
        return;
    }

    if(FeInit == Init::Both || FeInit == Init::JSON) // not exclusive to only JSON
        SetJSON(FoString);
    if(FeInit == Init::Both || FeInit == Init::BSON)
        SetBSON(bsoncxx::from_json(FoString.c_str()));

    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
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
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

void RA::JSON::Set(const web::json::value& FoJson, Init FeInit)
{
    Begin();
    if (FoJson.is_null())
        return;
    if (FeInit == Init::Default)
        FeInit = Init::Both;

    try
    {
        if (FeInit == Init::Both || FeInit == Init::JSON) // If not exclusivly only BSON
        {
            if (MoFullJsonPtr == nullptr)
                MoFullJsonPtr = RA::MakeShared<web::json::value>(FoJson);
            else
                *MoFullJsonPtr = FoJson;
        }
        if ((FeInit == Init::Both || FeInit == Init::BSON) && !IsNull())
            SetBSON(bsoncxx::from_json(GetSingleLineJson().c_str()));
    }
    catch (...)
    {
        ThrowIt("Error >>\n", ZoomReset().GetPrettyJson());
    }
    MoZoomedJsonPtr = nullptr;
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

xstring RA::JSON::GetPrettyJson(const int FnSpaceCount, const char FcIndentChar, const bool FbEnsureAscii) const
{
    Begin();
    if (MoFullJsonPtr == nullptr)
        return "null";
    const web::json::value& JsonTarget = (MoZoomedJsonPtr) ? *MoZoomedJsonPtr : *MoFullJsonPtr;
    if (JsonTarget.is_null())
        return "{ null }\n";
    return nlohmann::json::parse(WTXS(JsonTarget.serialize().c_str()).c_str())
        .dump(FnSpaceCount, FcIndentChar, FbEnsureAscii).c_str();
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

xstring RA::JSON::GetSingleLineJson() const
{
    Begin();
    const web::json::value& JsonTarget = (MoZoomedJsonPtr) ? *MoZoomedJsonPtr : *MoFullJsonPtr;
    if (JsonTarget.is_null())
        return "{ null }\n";
    return nlohmann::json::parse(WTXS(GetJSON().serialize().c_str()).c_str()).dump(4, ' ', true).c_str();
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

void RA::JSON::PrintJson() const
{
    Begin();
    const web::json::value& JsonTarget = (MoZoomedJsonPtr) ? *MoZoomedJsonPtr : *MoFullJsonPtr;
    if (JsonTarget.is_null())
    {
        cout << "{ null }\n";
        return;
    }
    std::cout << nlohmann::json::parse(WTXS(GetJSON().serialize().c_str()).c_str()).dump(4, ' ', true).c_str() << std::endl;
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

bool RA::JSON::IsNull() const
{
    Begin();
    if (MoFullJsonPtr == nullptr && MoBsonValuePtr == nullptr)
        return true;
    if (MoBsonValuePtr)
        return false;
    GSS(MoFullJson);
    return MoFullJson.is_null();
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

bool RA::JSON::Has(const xstring& FsObjectName) const
{
    Begin();
    GSS(MoFullJson);
    return MoFullJson.has_field(FsObjectName.ToStdWString());
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

bool RA::JSON::Has(const wchar_t* FsObjectName) const
{
    Begin();
    GSS(MoFullJson);
    return MoFullJson.has_field(FsObjectName);
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

xint RA::JSON::Size() const
{
    Begin();
    if (MoZoomedJsonPtr)
        return MoZoomedJsonPtr->size();
    else if (MoFullJsonPtr)
        return MoFullJsonPtr->size();
    else if (MoBsonValuePtr)
        return MoBsonValuePtr->view().length();
    else
        return 0;
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

RA::JSON& RA::JSON::Zoom(const xint Idx)
{
    Begin();
    GSS(MoFullJson);
    if (MoFullJson.to_string() == L"null")
        return *this;
    if (MoZoomedJsonPtr == nullptr)
    {
        try
        {
            MoZoomedJsonPtr = RA::MakeShared<web::json::value>(MoFullJson.at(Idx));
        }
        catch (...)
        {
            cout << "Start of Failed JSON: " << WTXS(MoFullJson.to_string().c_str())(0, 100) << endl;
            ThrowIt("Failed to Get JSON");
        }
    }
    else
    {
        GSS(MoZoomedJson);
        MoZoomedJson = MoZoomedJson.at(Idx);
    }
    return *this;
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

RA::JSON& RA::JSON::Zoom(const char* FacObject)
{
    Begin();
    ThrowNoJSON();
    size_t Size = strlen(FacObject) + 1;
    auto LacWideCharSPtr = RA::SharedPtr<wchar_t[]>(new wchar_t[Size]); auto LacWideCharPtr = LacWideCharSPtr.Ptr();
    mbstowcs_s(NULL, LacWideCharPtr, strlen(FacObject) + 1, FacObject, strlen(FacObject));
    LacWideCharPtr[Size - 1] = L'\0';
    Zoom(LacWideCharPtr);
    return *this;
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

RA::JSON& RA::JSON::Zoom(const xstring& FsObject)
{
    Begin();
    ThrowNoJSON();
    return Zoom(FsObject.c_str());
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

RA::JSON& RA::JSON::Zoom(const wchar_t* FacObject)
{
    Begin();
    GSS(MoFullJson);
    if (MoFullJson.to_string() == L"null")
        return *this;
    if (MoZoomedJsonPtr == nullptr)
    {
        try
        {
            if (MoFullJson.is_array())
            {
                MoZoomedJsonPtr = RA::MakeShared<web::json::value>(MoZoomedJsonPtr.Get()
                    //.as_array().at(0)
                    .as_object().at(FacObject));
            }else
                MoZoomedJsonPtr = RA::MakeShared<web::json::value>(MoFullJson.at(FacObject));
        }
        catch (...)
        {
            cout << "Start of Failed JSON: " << WTXS(MoFullJson.to_string().c_str())(0, 100) << endl;
            ThrowIt("Failed to Get JSON");
        }
    }
    else
    {
        GSS(MoZoomedJson);
        MoZoomedJson = MoZoomedJson.at(FacObject);
    }
    return *this;
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

RA::JSON& RA::JSON::ZoomReset()
{
    Begin();
    MoZoomedJsonPtr = nullptr;
    return *this;
    Rescue();
}

web::json::value RA::JSON::GetValue(const char* FacObject) const
{
    Begin();
    size_t Size = strlen(FacObject) + 1;
    auto LacWideCharSPtr = RA::SharedPtr<wchar_t[]>(new wchar_t[Size]); auto LacWideCharPtr = LacWideCharSPtr.Ptr();
    mbstowcs_s(NULL, LacWideCharPtr, strlen(FacObject) + 1, FacObject, strlen(FacObject));
    LacWideCharPtr[Size - 1] = L'\0';

    if (!!MoZoomedJsonPtr)
    {
        GET(MoZoomedJson);
        if (MoZoomedJson.is_array())
        {
            return MoZoomedJson
                .as_array().at(0)
                .as_object().at(LacWideCharPtr);
        }
        else
            return MoZoomedJson
            .as_object().at(LacWideCharPtr);
    }
    else
    {
        GET(MoFullJson);
        if (MoFullJson.is_array())
        {
            return MoFullJson
                .as_array().at(0)
                .as_object().at(LacWideCharPtr);
        }
        else
            return MoFullJson
            .as_object().at(LacWideCharPtr);
    }

    Rescue();
}

web::json::value RA::JSON::GetValue(const wchar_t* FacObject) const
{
    Begin();
    GET(MoFullJson);
    if (MoFullJson.is_array())
    {
        return MoFullJson
            .as_array().at(0)
            .as_object().at(FacObject);
    }
    else
        return MoFullJson
        .as_object().at(FacObject);
    Rescue();
}

web::json::object RA::JSON::GetObj(const xint Idx) const
{
    Begin();
    return MoFullJsonPtr.Get().as_array().at(Idx).as_object();
    Rescue();
}

web::json::object RA::JSON::GetObj(const char* FacObject) const
{
    Begin();
    return GetValue(FacObject).as_object();
    Rescue();
}

web::json::object RA::JSON::GetObj(const xstring& FsObject) const
{
    Begin();
    return GetObj(FsObject.Ptr());
    Rescue();
}

web::json::array RA::JSON::GetArr(const xint Idx) const
{
    Begin();
    return MoFullJsonPtr.Get().as_array().at(Idx).as_array();
    Rescue();
}

web::json::array RA::JSON::GetArr(const char* FacObject) const
{
    Begin();
    return GetValue(FacObject).as_array();
    Rescue();
}

web::json::array RA::JSON::GetArr(const wchar_t* FacObject) const
{
    Begin();
    GET(MoFullJson);
    return GetValue(FacObject).as_array();
    Rescue();
}

web::json::array RA::JSON::GetArr(const xstring& FsObject) const
{
    Begin();
    return GetArr(FsObject.Ptr());
    Rescue();
}

int RA::JSON::GetInt(const char* FacObject) const
{
    Begin();
    return GetValue(FacObject).as_integer();
    Rescue();
}

int RA::JSON::GetInt(const wchar_t* FacObject) const
{
    Begin();
    return GetValue(FacObject).as_integer();
    Rescue();
}

int RA::JSON::GetInt(const xstring& FacObject) const
{
    Begin();
    return GetValue(FacObject.Ptr()).as_integer();
    Rescue();
}

xint RA::JSON::GetUInt(const char* FacObject) const
{
    Begin();
    return GetValue(FacObject).as_number().to_uint64();
    Rescue();
}

xint RA::JSON::GetUInt(const wchar_t* FacObject) const
{
    Begin();
    return GetValue(FacObject).as_number().to_uint64();
    Rescue();
}

xint RA::JSON::GetUInt(const xstring& FacObject) const
{
    Begin();
    return GetValue(FacObject.Ptr()).as_number().to_uint64();
    Rescue();
}

xstring RA::JSON::GetString(const std::wstring& FwObject) const
{
    return RA::WTXS(This.GetJSON().as_object().at(FwObject).as_string().c_str());
}

std::wstring RA::JSON::GetWString(const std::wstring& FwObject) const
{
    return This.GetJSON().as_object().at(FwObject).as_string().c_str();
}

double RA::JSON::GetStringAsDouble(const std::wstring& FwObject) const
{
    return std::stod(This.GetJSON().as_object().at(FwObject).as_string().c_str());
}

const web::json::value& RA::JSON::GetFullObject() const
{
    Begin();
    ThrowNoJSON();
    return *MoFullJsonPtr;
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

const web::json::value& RA::JSON::GetJSON() const
{
    Begin();
    ThrowNoJSON();
    if (MoZoomedJsonPtr == nullptr)
        return *MoFullJsonPtr;
    else
        return *MoZoomedJsonPtr;
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });

}

const web::json::array& RA::JSON::GetJsonArray() const
{
    Begin();
    ThrowNoJSON();
    return GetJSON().as_array();
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

const web::json::object& RA::JSON::GetJsonObject() const
{
    Begin();
    ThrowNoJSON();
    return GetJSON().as_object();
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

const BSON::Value& RA::JSON::GetBSON() const
{
    Begin();
    ThrowNoBSON();
    return *MoBsonValuePtr;
    RescueThrow([this]() { RA::JSON(This).ZoomReset().GetPrettyJson().Print(); });
}

xstring RA::JSON::ToString() const
{
    if (!!MoZoomedJsonPtr)
        return RA::WTXS((*MoZoomedJsonPtr).to_string().c_str());
    if (!!MoFullJsonPtr)
        return RA::WTXS((*MoFullJsonPtr).to_string().c_str());
    return xstring::static_class;
}

xstring RA::JSON::ToString(const web::json::value& FwObject)
{
    return RA::WTXS(FwObject.serialize().c_str());
}