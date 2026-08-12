#pragma once

#include "JSON.h" // Include First
#include "xstring.h"
#include "xmap.h"
#include "Date.h"
#include "re2/re2.h"


#define MongoModDoc(__DOC__) RA::BSON::Start{} << __DOC__ << RA::BSON::OpenDoc
#define MongoSetter()        RA::BSON::Start{} << "$set" << RA::BSON::OpenDoc

namespace Mongo
{
    using Instance   = mongocxx::instance;
    using URI        = mongocxx::uri;
    using Client     = mongocxx::client;
    using Database   = mongocxx::database;
    using Collection = mongocxx::collection;
};

namespace RA
{
    class Stash
    {
        enum class EOrder
        {
            NewOld = -1, // Newest first (descending order)
            OldNew = 1   // Oldest first (ascending order)
        };

        istatic re2::RE2 SoDateTimeRex = R"(^.*(_id|Epoch|Date|DateTime|Time).*$)";
    public:
        Stash(const xstring& URL = "mongodb://localhost:27017");

        Stash(const Stash& Other);
        Stash(Stash&& Other) noexcept;

        void operator=(const Stash& Other);
        void operator=(Stash&& Other) noexcept;

        Stash& SetDatabase(const xstring& FsDatabase);
        Stash& SetCollection(const xstring& FsCollection);
        void   SetUniqueIndex(const xstring& Key = "_id", const bool FbAscending = true);

        Mongo::Database&   GetDatabase();
        Mongo::Collection& GetCollection();

        void DropCollection();

        BSON::Result::InsertOne  operator<<(const BSON::Value& FoView);
        BSON::Result::InsertOne  operator<<(const xstring& FoJsonStr);
       
        static RA::JSON CursorToJSON(BSON::Cursor& FoCursor);
        RA::JSON ToJSON(const EOrder FnOrder = EOrder::NewOld);

        template<typename K, typename V>
        xint Count(const K& FxKey, const V& FxValue);

        template<typename K, typename V>
        RA::JSON Match(const K& FxKey, const V& FxValue);

        template<typename... Keys>
        bool     Has(const BSON::Data& FnData, Keys&&... FxKeys);

        RA::JSON FindOne(const BSON::Data& FnData);
        RA::JSON FindOne(const BSON::Value& FnData);
        RA::JSON FindOne(const BSON::Builder& FnData);

        RA::JSON FindMany(const BSON::Data& FnData);
        RA::JSON FindMany(const BSON::Value& FnData);

        static BSON::Builder AppendDocument(const BSON::Builder& FoSource);
        static std::map<RA::Date, xmap<xstring, double>> TimeSeriesObjToMap(const RA::JSON& FoJson);
        static std::map<RA::Date, xmap<xstring, double>> TimeSeriesArrToMap(const RA::JSON& FoJson);

        BSON::Result::InsertOne  InsertOne(const BSON::Value& FvBSON) { return The << FvBSON; }
        BSON::Result::InsertOne  InsertOne(const xstring& FoJsonStr)  { return The << FoJsonStr; }
        BSON::Result::InsertMany InsertMany(const xvector<BSON::Value>& FvBSON);

        BSON::Result::Update UpdateOne (const BSON::View& FoFind, const BSON::View& FoReplace);
        BSON::Result::Update UpdateMany(const BSON::View& FoFind, const BSON::View& FoReplace);

        BSON::Result::Update UpsertOne(const BSON::View& FoFilter, const BSON::Value& FoUpdate);
        BSON::Result::Update UpsertOne(const BSON::View& FoFilter, const BSON::Builder& FoUpdate);

        xmap<RA::Date::EpochTime, BSON::Result::Update>
            UpsertTimeSeries(const xmap<RA::Date, xmap<xstring, double>>& FoDataMap, const xstring& FsAppend = "");

        BSON::Result::Delete DeleteOne(const BSON::View& FnDocument);
        BSON::Result::Delete DeleteOne(const BSON::Data& FnDocument);

        BSON::Result::Delete DeleteMany(const BSON::View& FnDocument);
        BSON::Result::Delete DeleteMany(const BSON::Data& FnDocument);

        RA::JSON Sort(const xstring& FoKey, const int FnDirection);
        RA::JSON Aggrigate(const BSON::Pipeline& FoPipeline);

    private:
        xstring           MsDatabase;
        xstring           MsCollection;

        xstring           MoURL;

        Mongo::URI        MoURI;
        Mongo::Client     MoClient;

        Mongo::Database   MoDatabase;
        Mongo::Collection MoCollection;

        static Mongo::Instance SoInstance;
    };

}


template<typename K, typename V>
inline xint RA::Stash::Count(const K& FxKey, const V& FxValue)
{
    Begin();
    BSON::Pipeline Pipeline{};
    Pipeline.match(BSON::NewKVP(FxKey, FxValue));
    BSON::Cursor Cursor = MoCollection.aggregate(Pipeline, BSON::Aggregate{});
    xint Count = 0;
    for (auto& Val : Cursor)
        Count++;
    return Count;
    Rescue();
}

template<typename K, typename V>
inline RA::JSON RA::Stash::Match(const K& FxKey, const V& FxValue)
{
    Begin();
    BSON::Pipeline Pipeline{};
    Pipeline.match(BSON::NewKVP(FxKey, FxValue));
    BSON::Cursor Cursor = MoCollection.aggregate(Pipeline, BSON::Aggregate{});
    return RA::Stash::CursorToJSON(Cursor);
    Rescue();
}

template<typename ...Keys>
inline bool RA::Stash::Has(const BSON::Data& FnData, Keys&& ...FxKeys)
{
    Begin();
    const auto LoFound = MoCollection.find_one(FnData); // find_one is non-const
    if (!LoFound)
        return false;

    if constexpr (sizeof...(Keys) == 0)
        return true;
    else
        return RA::JSON(LoFound.value()).Has(std::forward<Keys>(FxKeys)...);
    Rescue();
}
