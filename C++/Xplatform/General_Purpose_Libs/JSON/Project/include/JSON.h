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
// --------------------------------------------------
// REST API
#include <cpprest/http_client.h>
#include <cpprest/json.h>
#include <memory>
// --------------------------------------------------
// Radicalware
#include "xstring.h"
#include "xvector.h"
// --------------------------------------------------

using uint = size_t;

namespace BSON
{
    using Document = bsoncxx::builder::stream::document;
    using Start = bsoncxx::builder::stream::document;
    constexpr bsoncxx::builder::stream::finalize_type Finish;

    constexpr bsoncxx::builder::stream::open_document_type  OpenDoc;
    constexpr bsoncxx::builder::stream::close_document_type CloseDoc;

    constexpr bsoncxx::builder::stream::open_array_type  OpenArray;
    constexpr bsoncxx::builder::stream::close_array_type CloseArray;

    using Value = bsoncxx::document::value;
    using View = bsoncxx::document::view;
    using Data = bsoncxx::document::view_or_value;
    using Cursor = mongocxx::cursor;

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

        ~JSON();
        JSON() {};

        // If you have the string already, you want BSON as default
        JSON(const xstring& FoString, Init FeInit = Init::Default);
        // Default = Both >> You copy what you have (BSON) and get what you don't (JSON)
        JSON(const BSON::Value& FoBSON, Init FeInit = Init::Default);
        // Default = Both >> You copy what you have (BSON) and get what you don't (JSON)
        JSON(const web::json::value& FoJson, Init FeInit = Init::Default);

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

        RA::JSON& Zoom(const char* FacObject);
        RA::JSON& Zoom(const xstring& FsObject);
        RA::JSON& Zoom(const wchar_t* FacObject);
        RA::JSON& ZoomReset();

        const web::json::value& GetJson() const;
        const web::json::array& GetJsonArray() const;
        const web::json::object& GetJsonObject() const;
        const web::json::value& GetFullObject() const;
        const web::json::value& GetZoomedObject() const;

        template<typename T = xstring> // xstring, int, 
        xvector<T> GetValuesFromObject(const xstring& FoJsonValue) const;

        template<typename T = xstring> // xstring, int, 
        xvector<T> GetValuesFromArray(const xstring& FoJsonValue) const;

    private:
        std::shared_ptr<BSON::Value>      MoBsonValue;
        std::shared_ptr<web::json::value> MoFullJson;
        std::shared_ptr<web::json::value> MoZoomedJson;
    };

    // ==========================================================================================

    template<>
    inline xvector<xstring> JSON::GetValuesFromObject<xstring>(const xstring& FoJsonValue) const
    {
        xvector<xstring> LvxTargets;
        std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetZoomedObject();
        for (auto iter = LoJson.as_object().cbegin(); iter != LoJson.as_object().cend(); ++iter)
            LvxTargets << WTXS(iter->second.at(LsTarget).as_string().c_str());

        return LvxTargets;
    }

    template<>
    inline xvector<int> JSON::GetValuesFromObject<int>(const xstring& FoJsonValue) const
    {
        xvector<int> LvxTargets;
        std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetZoomedObject();
        for (auto iter = LoJson.as_object().cbegin(); iter != LoJson.as_object().cend(); ++iter)
            LvxTargets << iter->second.at(LsTarget).as_integer();

        return LvxTargets;
    }
    // ==========================================================================================
    template<>
    inline xvector<xstring> JSON::GetValuesFromArray<xstring>(const xstring& FoJsonValue) const
    {
        xvector<xstring> LvxTargets;
        std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetZoomedObject();
        for (auto iter = LoJson.as_array().cbegin(); iter != LoJson.as_array().cend(); ++iter)
            LvxTargets << WTXS(iter->at(LsTarget).as_string().c_str());

        return LvxTargets;
    }

    template<>
    inline xvector<int> JSON::GetValuesFromArray<int>(const xstring& FoJsonValue) const
    {
        xvector<int> LvxTargets;
        std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetZoomedObject();
        for (auto iter = LoJson.as_array().cbegin(); iter != LoJson.as_array().cend(); ++iter)
            LvxTargets << iter->at(LsTarget).as_integer();

        return LvxTargets;
    }

    template<>
    inline xvector<double> JSON::GetValuesFromArray<double>(const xstring& FoJsonValue) const
    {
        xvector<double> LvxTargets;
        std::wstring LsTarget = FoJsonValue.ToStdWString();
        auto& LoJson = GetZoomedObject();
        for (auto iter = LoJson.as_array().cbegin(); iter != LoJson.as_array().cend(); ++iter)
            LvxTargets << iter->at(LsTarget).as_number().to_int64();

        return LvxTargets;
    }
};
