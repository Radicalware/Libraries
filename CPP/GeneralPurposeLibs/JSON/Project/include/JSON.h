#pragma once

// --------------------------------------------------
// MONGO
#include "bsoncxx/json.hpp"
#include "bsoncxx/builder/stream/helpers.hpp"
#include "bsoncxx/builder/stream/document.hpp"
#include "bsoncxx/builder/stream/array.hpp"
#include "bsoncxx/view_or_value.hpp"
#include "bsoncxx/string/view_or_value.hpp"

#include "mongocxx/instance.hpp"
#include "mongocxx/uri.hpp"
#include "mongocxx/stdx.hpp"
#include "mongocxx/client.hpp"
#include "bsoncxx/types.hpp"
// --------------------------------------------------
// REST API
#include <cpprest/http_client.h>
#include <cpprest/json.h>
#include <memory>
// --------------------------------------------------
// Radicalware
#include "Macros.h"
#include "xstring.h"
#include "xvector.h"
#include "SharedPtr.h"
#include "RawMapping.h"
// --------------------------------------------------

namespace BSON
{
    using Document = bsoncxx::builder::basic::document;
    using Start    = bsoncxx::builder::stream::document;
    using Options  = mongocxx::options::update;

    constexpr decltype(bsoncxx::builder::stream::finalize) Finish;

    constexpr decltype(bsoncxx::builder::stream::open_document)  OpenDoc;
    constexpr decltype(bsoncxx::builder::stream::close_document) CloseDoc;

    constexpr decltype(bsoncxx::builder::stream::open_array)  OpenArray;
    constexpr decltype(bsoncxx::builder::stream::close_array) CloseArray;

    using OID = bsoncxx::oid;
    using Value = bsoncxx::document::value;
    using View = bsoncxx::document::view;
    using Data = bsoncxx::document::view_or_value;
    using Cursor = mongocxx::cursor;

    using Pipeline  = mongocxx::pipeline;
    using Aggregate = mongocxx::options::aggregate;

// ----------------------------------------------------------------------------------------------------------------------------

#define ALIAS_TEMPLATE_FUNCTION_BSON_MAKE_DOCUMENT(highLevelF, lowLevelF)   ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF);
#define ALIAS_TEMPLATE_FUNCTION_BSON_KVP(highLevelF, lowLevelF)             ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF)

    ALIAS_TEMPLATE_FUNCTION_BSON_MAKE_DOCUMENT(MakeDocument, bsoncxx::builder::basic::make_document);
    ALIAS_TEMPLATE_FUNCTION_BSON_KVP(KVP, bsoncxx::builder::basic::kvp);

#define MongoKVP(__KEY__, __VALUE__) BSON::MakeDocument(BSON::KVP(__KEY__, __VALUE__))

// ----------------------------------------------------------------------------------------------------------------------------

#define ALIAS_TEMPLATE_FUNCTION_BSON_MAKE_ARRAY(highLevelF, lowLevelF)   ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF);
    ALIAS_TEMPLATE_FUNCTION_BSON_MAKE_ARRAY(MakeArray, bsoncxx::builder::basic::make_array);

// ----------------------------------------------------------------------------------------------------------------------------

    namespace Result
    {
        using InsertOne = bsoncxx::stdx::optional<mongocxx::result::insert_one>;
        using InsertMany = bsoncxx::stdx::optional<mongocxx::result::insert_many>;
        using Value = bsoncxx::stdx::optional<bsoncxx::document::value>;
        using Update = bsoncxx::stdx::optional<mongocxx::v_noabi::result::update>;
        using Delete = bsoncxx::stdx::optional<mongocxx::v_noabi::result::delete_result>;
    };
};

namespace RA
{
    class JSON
    {
    public:
        enum class Init
        {
            Default,
            JSON,
            BSON,
            Both
        };

        static void Print(const web::json::value& FoWebJson);

        void Clear();
        ~JSON();
        JSON() {};

        // If you have the string already, you want BSON as default
        JSON(const xstring& FoString, Init FeInit = Init::Default);
        // Default = Both >> You copy what you have (BSON) and get what you don't (JSON)
        JSON(const BSON::Value& FoBSON, Init FeInit = Init::Default);
        // Default = Both >> You copy what you have (BSON) and get what you don't (JSON)
        JSON(const web::json::value& FoJson, Init FeInit = Init::Default);

        JSON(const JSON&  Other){ The = Other;}
        JSON(      JSON&& Other) noexcept { The = std::move(Other); };

        void operator=(const JSON&  Other);
        void operator=(      JSON&& Other) noexcept;

        void SetJSON(const xstring& FoString);
        void SetBSON(const BSON::Value& FoBSON);
        void Set(const xstring& FoString, Init FeInit);
        void Set(const BSON::Value& FoBSON, Init FeInit);
        void Set(const web::json::value& FoJson, Init FeInit);

        xstring GetPrettyJson(
            const int FnSpaceCount = 4,
            const char FcIndentChar = ' ',
            const bool FbEnsureAscii = true
        ) const;

        xstring GetSingleLineJson() const;
        void PrintJson() const;
        bool IsNull() const;

        bool Has(const xstring& FsObjectName) const;
        bool Has(const wchar_t* FsObjectName) const;
        xint Size() const;

        RA::JSON& Zoom(const xint Idx = 0);
        RA::JSON& Zoom(const char* FacObject);
        RA::JSON& Zoom(const xstring& FsObject);
        RA::JSON& Zoom(const wchar_t* FacObject);
        RA::JSON& ZoomReset();

        web::json::value GetValue(const char* FacObject) const;
        web::json::value GetValue(const wchar_t* FacObject) const;

        web::json::object GetObj(const xint Idx = 0) const;
        web::json::object GetObj(const char* FacObject) const;
        web::json::object GetObj(const xstring& FsObject) const;

        web::json::array GetArr(const xint Idx = 0) const;
        web::json::array GetArr(const char* FacObject) const;
        web::json::array GetArr(const wchar_t* FacObject) const;
        web::json::array GetArr(const xstring& FsObject) const;

        int              GetInt(const char* FacObject) const;
        int              GetInt(const wchar_t* FacObject) const;
        int              GetInt(const xstring& FacObject) const;

        xint             GetUInt(const char* FacObject) const;
        xint             GetUInt(const wchar_t* FacObject) const;
        xint             GetUInt(const xstring& FacObject) const;

        xstring          GetString(const std::wstring& FwObject) const;
        std::wstring     GetWString(const std::wstring& FwObject) const;
        double           GetStringAsDouble(const std::wstring& FwObject) const;
        
        //const web::json::object& Get() const; // Default, get the base document
        const web::json::value&  GetFullObject() const; // gets the full doc (often just "Get()" in an array)

        const web::json::value&  GetJSON() const;       // Based on Zoom
        const web::json::array&  GetJsonArray() const;  // Based on Zoom
        const web::json::object& GetJsonObject() const; // Based on Zoom

        const BSON::Value& GetBSON() const;

        template<typename T = xstring> // xstring, int, 
        xvector<T> GetValuesFromObject(const xstring& FoJsonValue) const;

        template<typename T = xstring> // xstring, int, 
        xvector<T> GetValuesFromArray(const xstring& FoJsonValue) const;

        xstring ToString() const;
        static xstring ToString(const web::json::value& FwObject);

    private:
        RA::SharedPtr<BSON::Value>      MoBsonValuePtr;
        RA::SharedPtr<web::json::value> MoFullJsonPtr;
        RA::SharedPtr<web::json::value> MoZoomedJsonPtr;
    };

    // ==========================================================================================

    template<>
    inline xvector<xstring> JSON::GetValuesFromObject<xstring>(const xstring& FoJsonValue) const
    {
        xvector<xstring> LvxTargets;
        const std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetJSON();
        for (auto iter = LoJson.as_object().cbegin(); iter != LoJson.as_object().cend(); ++iter)
            LvxTargets << WTXS(iter->second.at(LsTarget).as_string().c_str());

        return LvxTargets;
    }

    template<>
    inline xvector<int> JSON::GetValuesFromObject<int>(const xstring& FoJsonValue) const
    {
        xvector<int> LvxTargets;
        const std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetJSON();
        for (auto iter = LoJson.as_object().cbegin(); iter != LoJson.as_object().cend(); ++iter)
            LvxTargets << iter->second.at(LsTarget).as_integer();

        return LvxTargets;
    }

    template<>
    inline xvector<double> JSON::GetValuesFromObject<double>(const xstring& FoJsonValue) const
    {
        xvector<double> LvxTargets;
        const std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetJSON();
        for (auto iter = LoJson.as_object().cbegin(); iter != LoJson.as_object().cend(); ++iter)
            LvxTargets << iter->second.at(LsTarget).as_double();

        return LvxTargets;
    }
    // ==========================================================================================
    template<>
    inline xvector<xstring> JSON::GetValuesFromArray<xstring>(const xstring& FoJsonValue) const
    {
        xvector<xstring> LvxTargets;
        const std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetJSON();
        for (auto iter = LoJson.as_array().cbegin(); iter != LoJson.as_array().cend(); ++iter)
            LvxTargets << WTXS(iter->at(LsTarget).as_string().c_str());

        return LvxTargets;
    }

    template<>
    inline xvector<int> JSON::GetValuesFromArray<int>(const xstring& FoJsonValue) const
    {
        xvector<int> LvxTargets;
        const std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetJSON();
        for (auto iter = LoJson.as_array().cbegin(); iter != LoJson.as_array().cend(); ++iter)
            LvxTargets << iter->at(LsTarget).as_integer();

        return LvxTargets;
    }

    template<>
    inline xvector<xint> JSON::GetValuesFromArray<xint>(const xstring& FoJsonValue) const
    {
        xvector<xint> LvxTargets;
        const std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetJSON();
        for (auto iter = LoJson.as_array().cbegin(); iter != LoJson.as_array().cend(); ++iter)
            LvxTargets << iter->at(LsTarget).as_number().to_uint64();

        return LvxTargets;
    }

    template<>
    inline xvector<double> JSON::GetValuesFromArray<double>(const xstring& FoJsonValue) const
    {
        xvector<double> LvxTargets;
        const std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetJSON();
        for (auto iter = LoJson.as_array().cbegin(); iter != LoJson.as_array().cend(); ++iter)
            LvxTargets << iter->at(LsTarget).as_number().to_double();

        return LvxTargets;
    }
};

//std::ostream& operator<<(std::ostream& out, const RA::JSON& FnJson)
//{
//    out << FnJson.GetPrettyJson();
//    return out;
//}
