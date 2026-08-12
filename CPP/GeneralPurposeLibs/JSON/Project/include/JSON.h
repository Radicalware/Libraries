#pragma once

// --------------------------------------------------
// MONGO
#include "bsoncxx/json.hpp"
#include "bsoncxx/builder/stream/helpers.hpp"
#include "bsoncxx/builder/stream/document.hpp"
#include "bsoncxx/builder/stream/array.hpp"
#include "bsoncxx/view_or_value.hpp"
#include "bsoncxx/string/view_or_value.hpp"
#include "bsoncxx/builder/concatenate.hpp"

#include "mongocxx/instance.hpp"
#include "mongocxx/uri.hpp"
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
#include "xmap.h"
#include "SharedPtr.h"
#include "RawMapping.h"
#include "Mutex.h"
// --------------------------------------------------

namespace RA::BSON
{
    using Builder = bsoncxx::builder::basic::document;
    using Start = bsoncxx::builder::stream::document;
    using Options = mongocxx::options::update;
    constexpr decltype(bsoncxx::builder::stream::finalize) Finish;

    constexpr decltype(bsoncxx::builder::stream::open_document)  OpenDoc;
    constexpr decltype(bsoncxx::builder::stream::close_document) CloseDoc;

    constexpr decltype(bsoncxx::builder::stream::open_array)  OpenArray;
    constexpr decltype(bsoncxx::builder::stream::close_array) CloseArray;

    using Pipeline = mongocxx::pipeline;
    using Aggregate = mongocxx::options::aggregate;

    using OID = bsoncxx::v_noabi::oid;
    using Value = bsoncxx::v_noabi::document::value;
    using View = bsoncxx::v_noabi::document::view;
    using Data = bsoncxx::v_noabi::document::view_or_value;
    using Cursor = mongocxx::v_noabi::cursor;

    // using OID = bsoncxx::oid;
    // using Value = bsoncxx::document::value;
    // using View = bsoncxx::document::view;
    // using Data = bsoncxx::document::view_or_value;
    // using Cursor = mongocxx::cursor

// ----------------------------------------------------------------------------------------------------------------------------

#define ALIAS_TEMPLATE_FUNCTION_BSON_MAKE_DOCUMENT(highLevelF, lowLevelF)   ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF);
#define ALIAS_TEMPLATE_FUNCTION_BSON_KVP(highLevelF, lowLevelF)             ALIAS_TEMPLATE_FUNCTION(highLevelF, lowLevelF)

    ALIAS_TEMPLATE_FUNCTION_BSON_MAKE_DOCUMENT(NewBuilderKVP, bsoncxx::builder::basic::make_document);
    ALIAS_TEMPLATE_FUNCTION_BSON_KVP(KVP, bsoncxx::builder::basic::kvp);

#define MongoKVP(__KEY__, __VALUE__) BSON::NewBuilderKVP(BSON::KVP(__KEY__, __VALUE__))

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

    template<typename T>
    inline xstring GetResultOID(const T& FoResult) { Begin() return FoResult->inserted_id().get_oid().value.to_string(); Rescue() }
    template<typename T>
    inline int GetResultInt(const T& FoResult) { Begin() return FoResult->inserted_id().get_int32(); Rescue() }

    enum class EReq {
        Set,          // $set
        Inc,          // $inc
        Unset,        // $unset
        Rename,       // $rename
        Push,         // $push
        Pull,         // $pull
        AddToSet,     // $addToSet
        Min,          // $min
        Max,          // $max
        Mul,          // $mul
        // Add more as needed
    };

    inline static const xmap<EReq, xstring> SmReqs = {
        {EReq::Set,      "$set"},
        {EReq::Inc,      "$inc"},
        {EReq::Unset,    "$unset"},
        {EReq::Rename,   "$rename"},
        {EReq::Push,     "$push"},
        {EReq::Pull,     "$pull"},
        {EReq::AddToSet, "$addToSet"},
        {EReq::Min,      "$min"},
        {EReq::Max,      "$max"},
        {EReq::Mul,      "$mul"}
    };

};

namespace RA
{
    template<typename T>
    concept CharKey = std::convertible_to<T, xstring>
        && !std::same_as<std::remove_cvref_t<T>, std::wstring>;

    template<typename T>
    concept WCharKey = std::convertible_to<T, std::wstring>
        && !std::same_as<std::remove_cvref_t<T>, xstring>;

    template<typename T>
    concept XStringKey = std::same_as<std::remove_cvref_t<T>, xstring>;

    template<typename T>
    concept WStringKey = std::same_as<std::remove_cvref_t<T>, std::wstring>;


    class JSON : public web::json::value, public RA::MutexHandler
    {
    public:
        using web::json::value::operator=;

        static void Print(const web::json::value& FoWebJson);
        static void Print(const RA::JSON& FoWebJson);

        ~JSON();
        JSON() : value(web::json::value::object()) {};

        JSON(const xstring& FoString);
        JSON(const BSON::Value& FoBSON);
        JSON(const web::json::value& FoJson);

        JSON(const JSON& Other) { The = Other; }
        JSON(JSON&& Other) noexcept { The = std::move(Other); };

        void operator=(const JSON& Other);
        void operator=(JSON&& Other) noexcept;

        RA::JSON& operator[](const std::wstring& FsKey);
        RA::JSON& operator[](const wchar_t* FsKey);

        RA::JSON& operator=(const std::wstring& FsValue);
        RA::JSON& operator=(const wchar_t* FsValue);

        static BSON::Value ToBSON(const xstring& FsObject);
        BSON::Value ToBSON() const;

        void Set(xstring&& FoString);
        void Set(const xstring& FoString);
        void Set(const BSON::Value& FoBSON);
        void Set(const BSON::View& FoBSON);
        void Set(const web::json::value& FoJson);
    private:
        static web::json::value BSONViewToJson(const BSON::View& FoBSON);
        TTT static web::json::value BSONElementToJson(const T& FoElem);

    public:
        xstring GetPrettyJson(
            const int FnSpaceCount = 4,
            const char FcIndentChar = ' ',
            const bool FbEnsureAscii = true
        ) const;

        xstring           GetSingleLineJson() const;
        void              PrintJson() const;
        IC bool           IsNull() const { return The.is_null(); }
    private:
        template<typename... Rest>
        bool HasImpl(const std::wstring& FsKey, const Rest&... FoRestKeys) const;
    public:
        template<typename First, typename... Rest>
        bool Has(const First& FoFirstKey, const Rest&... FoRestKeys) const;

        IC xint           Size() const { return size(); }

        RA::JSON& At(const xint Idx = 0, const bool FbSilentError = false);
        const RA::JSON& At(const xint Idx = 0, const bool FbSilentError = false) const;
        template<typename T = xstring>
        T                 At(const xint Idx = 0, const bool FbSilentError = false) const;

        // FC = From Char* (and is slower due to converting to std::wstring)
        RA::JSON& KeyFC(const char* FsKey, const bool FbSilentError = false);
        RA::JSON& KeyFC(const xstring& FsKey, const bool FbSilentError = false);
        RA::JSON& Key(const std::wstring& FsKey, const bool FbSilentError = false);
        template<typename T = xstring>
        T                 Key(const std::wstring& FsKey, const bool FbSilentError = false) const;

        const RA::JSON& KeyFC(const char* FsKey, const bool FbSilentError = false) const;
        const RA::JSON& KeyFC(const xstring& FsKey, const bool FbSilentError = false) const;
        const RA::JSON& Key(const std::wstring& FsKey, const bool FbSilentError = false) const;

        bool              BxNullFC(const char* FsKey) const;
        bool              BxNullFC(const xstring& FsKey) const;
        bool              BxNull() const;
        bool              BxNull(const std::wstring& FsKey) const;

        template<typename T = xstring>
        xmap<xstring, T> GetMap(const std::wstring& FoKey) const;

        template<typename T = xstring> // Empty string if JSON is an array (so no key)
        xvector<T> ToArray(const std::wstring& FoKey = L"") const;


        xstring ToString() const;
        static xstring ToString(const web::json::value& FwObject);

        RIN explicit operator int()          const { return as_integer(); }
        RIN explicit operator double()       const { return as_double(); }
        RIN explicit operator xint()         const { return as_number().to_uint64(); }
        RIN explicit operator xstring()      const { return WTXS(as_string().c_str()); }
        RIN explicit operator std::wstring() const { return as_string(); }
        RIN explicit operator bool() const {
            if (is_null())    return false;
            if (is_boolean()) return as_bool();
            if (is_number())  return RA::Abs(as_double()) > std::numeric_limits<double>::epsilon();
            return true; // Object that is Not Null
        }
        RIN bool operator!() const { return !static_cast<bool>(The); }

        TTT T To() const { return static_cast<T>(*this); }

        auto& AsArray()  const { return as_array(); }
        auto& AsArray()        { return as_array(); }
        auto& AsObject() const { return as_object(); }
        auto& AsObject()       { return as_object(); }

    private:
        template<typename T>
        static constexpr bool IsNumeric = std::is_arithmetic_v<std::remove_cvref_t<T>>;
        template<typename T> requires IsNumeric<T>
        T GetExact() const
        {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::is_floating_point_v<U>)
            {
                if (!is_double())
                    ThrowIt("JSON type mismatch: expected a double but held: ", ToString());
                return static_cast<T>(as_double());
            }
            else if constexpr (std::is_same_v<U, bool>)
            {
                if (!is_boolean())
                    ThrowIt("JSON type mismatch: expected a bool but held: ", ToString());
                return as_bool();
            }
            else // integral
            {
                if (!is_integer())
                    ThrowIt("JSON type mismatch: expected an integer but held: ", ToString());
                if constexpr (std::is_unsigned_v<U>)
                    return static_cast<T>(as_number().to_uint64());
                else
                    return static_cast<T>(as_number().to_int64());
            }
        }

    public:
        template<typename T> requires JSON::IsNumeric<T>
        friend T operator+(const JSON& L, const T R) { return L.GetExact<T>() + R; }
        template<typename T> requires JSON::IsNumeric<T>
        friend T operator-(const JSON& L, const T R) { return L.GetExact<T>() - R; }
        template<typename T> requires JSON::IsNumeric<T>
        friend T operator*(const JSON& L, const T R) { return L.GetExact<T>() * R; }
        template<typename T> requires JSON::IsNumeric<T>
        friend T operator/(const JSON& L, const T R) { return L.GetExact<T>() / R; }

        // arithmetic <op> JSON  (result type = T)
        template<typename T> requires JSON::IsNumeric<T>
        friend T operator+(const T L, const JSON& R) { return L + R.GetExact<T>(); }
        template<typename T> requires JSON::IsNumeric<T>
        friend T operator-(const T L, const JSON& R) { return L - R.GetExact<T>(); }
        template<typename T> requires JSON::IsNumeric<T>
        friend T operator*(const T L, const JSON& R) { return L * R.GetExact<T>(); }
        template<typename T> requires JSON::IsNumeric<T>
        friend T operator/(const T L, const JSON& R) { return L / R.GetExact<T>(); }


        // ============================================================
        // Equality operators (all data types)
    private:
        bool EqualsStr(const std::wstring& FsValue) const
        {
            if (!is_string())
                return false;
            return as_string() == FsValue;
        }

        template<typename T> requires IsNumeric<T>
        bool EqualsNum(const T FnValue) const
        {
            using U = std::remove_cvref_t<T>;
            if constexpr (std::is_same_v<U, bool>)
            {
                if (!is_boolean())
                    return false;
                return as_bool() == FnValue;
            }
            else
            {
                if (!is_number())
                    return false;
                if (is_integer())
                {
                    if constexpr (std::is_unsigned_v<U>)
                        return as_number().to_uint64() == static_cast<uint64_t>(FnValue);
                    else
                        return as_number().to_int64() == static_cast<int64_t>(FnValue);
                }
                return as_double() == static_cast<double>(FnValue);
            }
        }
    public:
        // JSON == string types
        bool operator==(const std::wstring& FsValue) const { return EqualsStr(FsValue); }
        bool operator==(const wchar_t* FsValue) const { return EqualsStr(FsValue); }
        bool operator==(const xstring& FsValue) const { return EqualsStr(FsValue.ToStdWString()); }
        bool operator==(const char* FsValue) const { return EqualsStr(xstring(FsValue).ToStdWString()); }

        template<typename T> requires IsNumeric<T>
        bool operator==(const T FnValue) const { return EqualsNum(FnValue); }
        bool operator==(const JSON& Other) const // JSON == JSON (deep compare via base class)
        {
            return static_cast<const web::json::value&>(The) == static_cast<const web::json::value&>(Other);
        }

        // != counterparts
        bool operator!=(const std::wstring& FsValue) const { return !EqualsStr(FsValue); }
        bool operator!=(const wchar_t* FsValue) const { return !EqualsStr(FsValue); }
        bool operator!=(const xstring& FsValue) const { return !EqualsStr(FsValue.ToStdWString()); }
        bool operator!=(const char* FsValue) const { return !EqualsStr(xstring(FsValue).ToStdWString()); }
        template<typename T> requires IsNumeric<T>
        bool operator!=(const T FnValue) const { return !EqualsNum(FnValue); }
        bool operator!=(const JSON& Other) const { return !(The == Other); }

        // Reversed operands (value == JSON)
        friend bool operator==(const std::wstring& L, const JSON& R) { return R.EqualsStr(L); }
        friend bool operator==(const wchar_t* L, const JSON& R) { return R.EqualsStr(L); }
        friend bool operator==(const xstring& L, const JSON& R) { return R.EqualsStr(L.ToStdWString()); }
        friend bool operator==(const char* L, const JSON& R) { return R.EqualsStr(xstring(L).ToStdWString()); }
        template<typename T> requires IsNumeric<T>
        friend bool operator==(const T L, const JSON& R) { return R.EqualsNum(L); }

        friend bool operator!=(const std::wstring& L, const JSON& R) { return !R.EqualsStr(L); }
        friend bool operator!=(const wchar_t* L, const JSON& R) { return !R.EqualsStr(L); }
        friend bool operator!=(const xstring& L, const JSON& R) { return !R.EqualsStr(L.ToStdWString()); }
        friend bool operator!=(const char* L, const JSON& R) { return !R.EqualsStr(xstring(L).ToStdWString()); }
        template<typename T> requires IsNumeric<T>
        friend bool operator!=(const T L, const JSON& R) { return !R.EqualsNum(L); }

    };  // RA::JSON

    template<typename ...Rest>
    bool JSON::HasImpl(const std::wstring& FsKey, const Rest & ...FoRestKeys) const
    {
        if constexpr (sizeof...(Rest) == 0)
            return has_field(FsKey);
        else
        {
            if (!is_object() || !has_field(FsKey))
                return false;
            return SCast<const JSON&>(as_object().at(FsKey)).Has(FoRestKeys...);
        }
    }

    template<typename First, typename ...Rest>
    bool JSON::Has(const First& FoFirstKey, const Rest & ...FoRestKeys) const
    {
        if      constexpr (XStringKey<First>)  return HasImpl(FoFirstKey.ToStdWString(), FoRestKeys...);
        else if constexpr (CharKey<First>)     return HasImpl(xstring(FoFirstKey).ToStdWString(), FoRestKeys...);
        else if constexpr (WCharKey<First>)    return HasImpl(std::wstring(FoFirstKey), FoRestKeys...);
        else if constexpr (WStringKey<First>)  return HasImpl(FoFirstKey, FoRestKeys...);
        else
            ThrowIt("Invalid key type for JSON::Has() >> (wstring/wchar_t*/xstring/char*)");
    }

    // ============================================================================================================================
    // JSON::At<T> specializations

    template<>
    inline xstring JSON::At<xstring>(const xint Idx, const bool FbSilentError) const
    {
        auto& LoObj = At(Idx, FbSilentError);
        if (LoObj.is_string())
            return WTXS(LoObj.as_string().c_str());
        return WTXS(LoObj.to_string().c_str());
    }

    template<>
    inline int JSON::At<int>(const xint Idx, const bool FbSilentError) const
    {
        return At(Idx, FbSilentError).as_integer();
    }

    template<>
    inline double JSON::At<double>(const xint Idx, const bool FbSilentError) const
    {
        return At(Idx, FbSilentError).as_double();
    }

    template<>
    inline bool JSON::At<bool>(const xint Idx, const bool FbSilentError) const
    {
        return At(Idx, FbSilentError).as_bool();
    }

    template<>
    inline xint JSON::At<xint>(const xint Idx, const bool FbSilentError) const
    {
        return At(Idx, FbSilentError).as_number().to_uint64();
    }
    // ============================================================================================================================
    // JSON::Key<T> specializations

    template<>
    inline xstring RA::JSON::Key<xstring>(const std::wstring& FsKey, const bool FbSilentError) const
    {
        auto& LoObj = Key(FsKey, FbSilentError);
        if (LoObj.is_string())
            return WTXS(LoObj.as_string().c_str());
        return WTXS(LoObj.to_string().c_str());
    }

    template<>
    inline int RA::JSON::Key<int>(const std::wstring& FsKey, const bool FbSilentError) const
    {
        if (is_string())
            return std::stoi(Key(FsKey, FbSilentError).as_string());
        return Key(FsKey, FbSilentError).as_integer();
    }

    template<>
    inline double RA::JSON::Key<double>(const std::wstring& FsKey, const bool FbSilentError) const
    {
        if (is_string())
            return std::stod(Key(FsKey, FbSilentError).as_string());
        return Key(FsKey, FbSilentError).as_double();
    }

    template<>
    inline bool RA::JSON::Key<bool>(const std::wstring& FsKey, const bool FbSilentError) const
    {
        return Key(FsKey, FbSilentError).as_bool();
    }

    template<>
    inline xint RA::JSON::Key<xint>(const std::wstring& FsKey, const bool FbSilentError) const
    {
        if (is_string())
            return std::stoull(Key(FsKey, FbSilentError).as_string());
        return Key(FsKey, FbSilentError).as_number().to_uint64();
    }

    // ============================================================================================================================

    template<>
    inline xmap<xstring, xstring> JSON::GetMap<xstring>(const std::wstring& FoKey) const
    {
        Begin()
            xmap<xstring, xstring> LmTargets;
        const auto& LObj = as_object();
        for (auto iter = LObj.cbegin(); iter != LObj.cend(); ++iter)
        {
            if (iter->second.at(FoKey).is_string())
                LmTargets.AddPair(
                    WTXS(iter->first.c_str()),
                    WTXS(iter->second.at(FoKey).as_string().c_str()));
            else
                LmTargets.AddPair(
                    WTXS(iter->first.c_str()),
                    WTXS(iter->second.at(FoKey).to_string().c_str()));
        }
        return LmTargets;
        RescueThrow([this]() { GetPrettyJson().Print(); });
    }

    template<>
    inline xmap<xstring, int> JSON::GetMap<int>(const std::wstring& FoKey) const
    {
        Begin()
            xmap<xstring, int> LmTargets;
        const auto& LObj = as_object();
        for (auto iter = LObj.cbegin(); iter != LObj.cend(); ++iter)
        {
            if (iter->second.at(FoKey).is_string())
                LmTargets.AddPair(
                    WTXS(iter->first.c_str()),
                    std::stoi(iter->second.at(FoKey).as_string()));
            else
                LmTargets.AddPair(
                    WTXS(iter->first.c_str()),
                    iter->second.at(FoKey).as_integer());
        }
        return LmTargets;
        RescueThrow([this]() { GetPrettyJson().Print(); });
    }

    template<>
    inline xmap<xstring, xint> JSON::GetMap<xint>(const std::wstring& FoKey) const
    {
        Begin()
            xmap<xstring, xint> LmTargets;
        const auto& LObj = as_object();
        for (auto iter = LObj.cbegin(); iter != LObj.cend(); ++iter)
        {
            if (iter->second.at(FoKey).is_string())
                LmTargets.AddPair(
                    WTXS(iter->first.c_str()),
                    std::stoull(iter->second.at(FoKey).as_string()));
            else
                LmTargets.AddPair(
                    WTXS(iter->first.c_str()),
                    iter->second.at(FoKey).as_number().to_uint64());
        }
        return LmTargets;
        RescueThrow([this]() { GetPrettyJson().Print(); });
    }

    template<>
    inline xmap<xstring, double> JSON::GetMap<double>(const std::wstring& FoKey) const
    {
        Begin()
            xmap<xstring, double> LmTargets;
        const auto& LObj = as_object();
        for (auto iter = LObj.cbegin(); iter != LObj.cend(); ++iter)
        {
            if (iter->second.at(FoKey).is_string())
                LmTargets.AddPair(
                    WTXS(iter->first.c_str()),
                    std::stod(iter->second.at(FoKey).as_string()));
            else
                LmTargets.AddPair(
                    WTXS(iter->first.c_str()),
                    iter->second.at(FoKey).as_double());
        }
        return LmTargets;
        RescueThrow([this]() { GetPrettyJson().Print(); });
    }
    // ============================================================================================================================
    template<>
    inline xvector<xstring> JSON::ToArray<xstring>(const std::wstring& FoKey) const
    {
        Begin()
            xvector<xstring> LvxTargets;
        auto& LvArr = (FoKey.size())
            ? as_object().at(FoKey).as_array()
            : as_array();
        for (auto iter = LvArr.cbegin(); iter != LvArr.cend(); ++iter)
        {
            if (iter->is_string())
                LvxTargets << WTXS(iter->as_string().c_str());
            else
                LvxTargets << WTXS(iter->to_string().c_str());
        }
        return LvxTargets;
        RescueThrow([this]() { GetPrettyJson().Print(); });
    }

    template<>
    inline xvector<int> JSON::ToArray<int>(const std::wstring& FoKey) const
    {
        Begin()
            xvector<int> LvxTargets;
        auto& LvArr = (FoKey.size())
            ? as_object().at(FoKey).as_array()
            : as_array();
        for (auto iter = LvArr.cbegin(); iter != LvArr.cend(); ++iter)
        {
            if (iter->is_string())
                LvxTargets << std::stoi(iter->as_string());
            else
                LvxTargets << iter->as_integer();
        }
        return LvxTargets;
        RescueThrow([this]() { GetPrettyJson().Print(); });
    }

    template<>
    inline xvector<xint> JSON::ToArray<xint>(const std::wstring& FoKey) const
    {
        Begin()
            xvector<xint> LvxTargets;
        auto& LvArr = (FoKey.size())
            ? as_object().at(FoKey).as_array()
            : as_array();
        for (auto iter = LvArr.cbegin(); iter != LvArr.cend(); ++iter)
        {
            if (iter->is_string())
                LvxTargets << std::stoull(iter->as_string());
            else
                LvxTargets << iter->as_number().to_uint64();
        }
        return LvxTargets;
        RescueThrow([this]() { GetPrettyJson().Print(); });
    }

    template<>
    inline xvector<double> JSON::ToArray<double>(const std::wstring& FoKey) const
    {
        Begin()
            xvector<double> LvxTargets;
        auto& LvArr = (FoKey.size())
            ? as_object().at(FoKey).as_array()
            : as_array();
        for (auto iter = LvArr.cbegin(); iter != LvArr.cend(); ++iter)
        {
            if (iter->is_string())
                LvxTargets << std::stod(iter->as_string());
            else
                LvxTargets << iter->as_double();
        }
        return LvxTargets;
        RescueThrow([this]() { GetPrettyJson().Print(); });
    }

}; // RA::JSON

namespace RA
{
    inline std::ostream& operator<<(std::ostream& out, const RA::JSON& FnJson) {
        out << FnJson.GetPrettyJson();
        return out;
    }
}

// Converting Between Data Types
namespace RA::BSON
{
    inline RA::BSON::OID MakeEpochOID(const std::time_t FnEpochSeconds)
    {
        RA::BSON::OID LoRandom; // default ctor: unique, current-time-based OID
        std::array<std::uint8_t, 12> LaBytes;
        std::memcpy(LaBytes.data(), LoRandom.bytes(), 12);

        const auto LnEpoch = static_cast<std::uint32_t>(FnEpochSeconds);
        LaBytes[0] = static_cast<std::uint8_t>((LnEpoch >> 24) & 0xFF);
        LaBytes[1] = static_cast<std::uint8_t>((LnEpoch >> 16) & 0xFF);
        LaBytes[2] = static_cast<std::uint8_t>((LnEpoch >> 8) & 0xFF);
        LaBytes[3] = static_cast<std::uint8_t>(LnEpoch & 0xFF);

        return RA::BSON::OID(reinterpret_cast<const char*>(LaBytes.data()), LaBytes.size());
    }

    template<typename K, typename V>
    inline RA::BSON::Value NewKVP(K&& FoKey, V&& FoValue) { return BSON::NewBuilderKVP(BSON::KVP(std::forward<K>(FoKey), std::forward<V>(FoValue))); }

    inline BSON::Value ToValue(const xstring& FsJson) { Begin() return BSON::Value{ bsoncxx::from_json(FsJson.Ptr()) }; Rescue() }
    inline BSON::Value ToView(const xstring& FsJson) { Begin() return BSON::Value{ bsoncxx::from_json(FsJson.Ptr()).view() }; Rescue() }
    // ====================== ViewOrValue <-> View ============================================
    /// Convert view to view_or_value (non-owning)
    inline BSON::Data  ToData(const BSON::View& FoView) { return BSON::Data{ FoView }; }

    /// Convert view_or_value to view (always cheap)
    inline BSON::View  ToView(const BSON::Data& FoView) noexcept { return FoView.view(); }
    inline BSON::View  ToView(BSON::Data&& FoView)      noexcept { return FoView.view(); }
    // ====================== ViewOrValue <-> Value ============================================

    /// Convert to owning value - moves if possible, copies otherwise
    inline BSON::View  ToValue(BSON::Data FoView) {
        if (FoView.is_owning())
            return BSON::View{ std::move(FoView) };
        else
            return BSON::View{ FoView.view() };
    }
    /// Force a deep copy to value
    inline BSON::View ToValueCopy(const BSON::Data& FoView) { return BSON::View{ FoView.view() }; }

    inline BSON::Builder ToBuilder(const BSON::Data& FoData) {
        BSON::Builder LoDocument;
        LoDocument.append(BSON::KVP("copy", FoData.view()));
        return LoDocument;
    }

    inline BSON::Builder ToBuilder(const BSON::View& FoView) {
        BSON::Builder LoDocument;
        LoDocument.append(BSON::KVP("copy", FoView));
        return LoDocument;
    }
}

// View & Builder Builders
namespace RA::BSON
{
    // ============================================================================================================================
    // Concepts from C++20 to constrain the template parameters for the AppendToBsonBuilder function
    // ============================================================================================================================
    template<typename T>
    concept Container = requires(T t) {
        t.begin();
        t.end();
    };

    template<typename T>
    concept MapLike = Container<T> && requires(T t) {
        typename T::key_type;
        typename T::mapped_type;
    };

    // ============================================================================================================================
    // Forward Declarations (required so recursive calls can see all overloads regardless of definition order)
    // ============================================================================================================================
    template<typename BuilderType>
    void AppendToBsonBuilder(BuilderType&);

    template<typename BuilderType, typename Key, MapLike MapType, typename... Rest>
    void AppendToBsonBuilder(BuilderType& Builder, Key&& KeyArg, MapType&& MapValue, Rest&&... RestArgs);

    template<typename BuilderType, typename Key, Container ContainerType, typename... Rest>
    void AppendToBsonBuilder(BuilderType& Builder, Key&& KeyArg, ContainerType&& ContainerValue, Rest&&... RestArgs)
        requires (!MapLike<ContainerType> &&
    !std::is_convertible_v<std::decay_t<ContainerType>, const char*> &&
        !std::is_base_of_v<std::string, std::decay_t<ContainerType>> &&
        !std::is_same_v<std::decay_t<ContainerType>, RA::BSON::Value> &&
        !std::is_same_v<std::decay_t<ContainerType>, RA::BSON::View>);

    template<typename BuilderType, typename Key, typename Value, typename... Rest>
    void AppendToBsonBuilder(BuilderType& Builder, Key&& KeyArg, Value&& ValueArg, Rest&&... RestArgs)
        requires ((!Container<Value> || std::is_base_of_v<std::string, std::decay_t<Value>>) &&
    !std::is_same_v<std::decay_t<Value>, RA::BSON::Value> &&
        !std::is_same_v<std::decay_t<Value>, RA::BSON::View>);

    template<typename BuilderType, typename Key, typename... Rest>
    void AppendToBsonBuilder(BuilderType& Builder, Key&& KeyArg, const RA::BSON::Value& FoValue, Rest&&... RestArgs);

    template<typename BuilderType, typename Key, typename... Rest>
    void AppendToBsonBuilder(BuilderType& Builder, Key&& KeyArg, const RA::BSON::View& FoView, Rest&&... RestArgs);

    template<typename BuilderType, MapLike MapType>
    void AppendToBsonBuilder(BuilderType& Builder, MapType&& MapValue);

    template<typename BuilderType, Container ContainerType>
    void AppendToBsonBuilder(BuilderType& Builder, ContainerType&& ContainerValue)
        requires (!MapLike<ContainerType> &&
    !std::is_same_v<std::decay_t<ContainerType>, RA::BSON::Value> &&
        !std::is_same_v<std::decay_t<ContainerType>, RA::BSON::View> &&
        !std::is_convertible_v<std::decay_t<ContainerType>, const char*> &&
        !std::is_base_of_v<std::string, std::decay_t<ContainerType>>);

    template<typename BuilderType>
    void AppendToBsonBuilder(BuilderType& Builder, const RA::BSON::Value& FoValue);

    template<typename BuilderType>
    void AppendToBsonBuilder(BuilderType& Builder, const RA::BSON::View& FoView);

    // ============================================================================================================================
    // Base Case
    // ============================================================================================================================
    template<typename BuilderType>
    void AppendToBsonBuilder(BuilderType&) {}

    // ============================================================================================================================
    // Helper: disambiguate const char* / string literals
    // ============================================================================================================================
    template<typename T>
    decltype(auto) NormalizeBsonArg(T&& Val)
    {
        if constexpr (std::is_convertible_v<std::decay_t<T>, const char*>)
            return std::string(Val);
        else if constexpr (std::is_same_v<std::decay_t<T>, xstring>)
            return std::string(Val.Ptr());
        else
            return std::forward<T>(Val);
    }

    // ============================================================================================================================
    // Map → Nested Builder
    // ============================================================================================================================
    template<typename BuilderType, typename Key, MapLike MapType, typename... Rest>
    void AppendToBsonBuilder(BuilderType& Builder, Key&& KeyArg, MapType&& MapValue, Rest&&... RestArgs)
    {
        bsoncxx::builder::basic::document SubDoc;
        for (const auto& Pair : MapValue)
            SubDoc.append(KVP(NormalizeBsonArg(Pair.first), NormalizeBsonArg(Pair.second)));
        Builder.append(KVP(NormalizeBsonArg(std::forward<Key>(KeyArg)), SubDoc.extract()));
        AppendToBsonBuilder(Builder, std::forward<Rest>(RestArgs)...);
    }

    // ============================================================================================================================
    // Container (non-map) → Array
    // ============================================================================================================================
    template<typename BuilderType, typename Key, Container ContainerType, typename... Rest>
    void AppendToBsonBuilder(BuilderType& Builder, Key&& KeyArg, ContainerType&& ContainerValue, Rest&&... RestArgs)
        requires (!MapLike<ContainerType> &&
    !std::is_convertible_v<std::decay_t<ContainerType>, const char*> &&
        !std::is_base_of_v<std::string, std::decay_t<ContainerType>> &&
        !std::is_same_v<std::decay_t<ContainerType>, RA::BSON::Value> &&
        !std::is_same_v<std::decay_t<ContainerType>, RA::BSON::View>)
    {
        bsoncxx::builder::basic::array SubArr;
        for (auto& Item : ContainerValue)
            SubArr.append(NormalizeBsonArg(Item));
        Builder.append(KVP(NormalizeBsonArg(std::forward<Key>(KeyArg)), SubArr.extract()));
        AppendToBsonBuilder(Builder, std::forward<Rest>(RestArgs)...);
    }

    // ============================================================================================================================
    // Normal Key-Value
    // ============================================================================================================================
    template<typename BuilderType, typename Key, typename Value, typename... Rest>
    void AppendToBsonBuilder(BuilderType& Builder, Key&& KeyArg, Value&& ValueArg, Rest&&... RestArgs)
        requires ((!Container<Value> || std::is_base_of_v<std::string, std::decay_t<Value>>) &&
    !std::is_same_v<std::decay_t<Value>, RA::BSON::Value> &&
        !std::is_same_v<std::decay_t<Value>, RA::BSON::View>)
    {
        Builder.append(KVP(NormalizeBsonArg(std::forward<Key>(KeyArg)), NormalizeBsonArg(std::forward<Value>(ValueArg))));
        AppendToBsonBuilder(Builder, std::forward<Rest>(RestArgs)...);
    }

    // ============================================================================================================================
    // Key + BSON Value → nested key holding the sub-document
    // ============================================================================================================================
    template<typename BuilderType, typename Key, typename... Rest>
    void AppendToBsonBuilder(BuilderType& Builder, Key&& KeyArg, const RA::BSON::Value& FoValue, Rest&&... RestArgs)
    {
        Builder.append(KVP(NormalizeBsonArg(std::forward<Key>(KeyArg)), FoValue.view()));
        AppendToBsonBuilder(Builder, std::forward<Rest>(RestArgs)...);
    }

    // ============================================================================================================================
    // Key + BSON View → nested key holding the sub-document
    // ============================================================================================================================
    template<typename BuilderType, typename Key, typename... Rest>
    void AppendToBsonBuilder(BuilderType& Builder, Key&& KeyArg, const RA::BSON::View& FoView, Rest&&... RestArgs)
    {
        Builder.append(KVP(NormalizeBsonArg(std::forward<Key>(KeyArg)), FoView));
        AppendToBsonBuilder(Builder, std::forward<Rest>(RestArgs)...);
    }

    // ============================================================================================================================
    // Single Map (no preceding key) → Flatten directly
    // ============================================================================================================================
    template<typename BuilderType, MapLike MapType>
    void AppendToBsonBuilder(BuilderType& Builder, MapType&& MapValue)
    {
        for (const auto& Pair : MapValue)
            Builder.append(KVP(NormalizeBsonArg(Pair.first), NormalizeBsonArg(Pair.second)));
    }

    // ============================================================================================================================
    // Single Container (no preceding key) → Flatten by index
    // ============================================================================================================================
    template<typename BuilderType, Container ContainerType>
    void AppendToBsonBuilder(BuilderType& Builder, ContainerType&& ContainerValue)
        requires (!MapLike<ContainerType> &&
    !std::is_same_v<std::decay_t<ContainerType>, RA::BSON::Value> &&
        !std::is_same_v<std::decay_t<ContainerType>, RA::BSON::View> &&
        !std::is_convertible_v<std::decay_t<ContainerType>, const char*> &&
        !std::is_base_of_v<std::string, std::decay_t<ContainerType>>)
    {
        xint Idx = 0;
        for (auto& Item : ContainerValue)
        {
            Builder.append(KVP(std::to_string(Idx), NormalizeBsonArg(Item)));
            ++Idx;
        }
    }

    // ==========================================================================================
    // Single BSON Value/View (no preceding key) → Concatenate directly
    // ==========================================================================================
    template<typename BuilderType>
    void AppendToBsonBuilder(BuilderType& Builder, const RA::BSON::Value& FoValue)
    {
        Builder.append(bsoncxx::builder::concatenate(FoValue.view()));
    }

    template<typename BuilderType>
    void AppendToBsonBuilder(BuilderType& Builder, const RA::BSON::View& FoView)
    {
        Builder.append(bsoncxx::builder::concatenate(FoView));
    }

    // ============================================================================================================================
    // Main Functions
    // ============================================================================================================================
    template<typename... Args>
    RA::BSON::Value NewValue(Args&&... args)
    {
        Begin();
        auto Builder = RA::BSON::Builder{};
        AppendToBsonBuilder(Builder, std::forward<Args>(args)...);
        return Builder.extract();
        Rescue();
    }

    template<typename... Args>
    RA::BSON::Value NewReqValue(RA::BSON::EReq FeReq, Args&&... args)
    {
        Begin();
        auto Builder = RA::BSON::Builder{};
        auto SubDoc = RA::BSON::Builder{};
        AppendToBsonBuilder(SubDoc, std::forward<Args>(args)...);
        Builder.append(KVP(SCast<std::string>(RA::BSON::SmReqs.Key(FeReq)), SubDoc.extract()));
        return Builder.extract();
        Rescue();
    }

    template<typename... Args>
    RA::BSON::Builder NewBuilder(Args&&... args)
    {
        Begin();
        auto Builder = RA::BSON::Builder{};
        AppendToBsonBuilder(Builder, std::forward<Args>(args)...);
        return Builder;
        Rescue();
    }

    template<typename... Args>
    RA::BSON::View NewFilter(Args&&... args) // Same as Builder but mroe 
    {
        Begin();
        BSON::Builder LoBuilder = NewBuilder(std::forward<Args>(args)...);
        return LoBuilder.view();
        Rescue();
    }
}