#pragma once

#include "JSON.h" // Include First
#include "xstring.h"


#define MongoOpenDoc(__DOC__) BSON::Start{} << __DOC__ << BSON::OpenDoc
#define MongoSetDoc()         BSON::Start{} << "$set" << BSON::OpenDoc
#define MongoCloseDoc()       BSON::CloseDoc << BSON::Finish

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
    public:
        Stash(const xstring& URL = "mongodb://localhost:27017");

        Stash(const Stash& Other);
        Stash(Stash&& Other) noexcept;

        void operator=(const Stash& Other);
        void operator=(Stash&& Other) noexcept;

        Stash& SetDatabase(const xstring& FsDatabase);
        Stash& SetCollection(const xstring& FsCollection);

        Mongo::Database&   GetDatabase();
        Mongo::Collection& GetCollection();

        void DropCollection();

        BSON::Result::InsertOne  operator<<(const BSON::Value& FoView);
        BSON::Result::InsertOne  operator<<(const xstring& FoJsonStr);
        //BSON::Result::InsertMany operator<<(const BSON::Document& FoDocument);
       
        static RA::JSON CursorToJSON(BSON::Cursor& FoCursor, RA::JSON::Init FeInit);
        RA::JSON GetAll(RA::JSON::Init FeInit = RA::JSON::Init::Both);

        template<typename K, typename V>
        pint Count(const K& FxKey, const V& FxValue);

        template<typename K, typename V>
        RA::JSON Match(const K& FxKey, const V& FxValue, RA::JSON::Init FeInit = RA::JSON::Init::Both);
        RA::JSON FindOne(const BSON::Data& FnData, RA::JSON::Init FeInit = RA::JSON::Init::Both);
        RA::JSON FindMany(const BSON::Data& FnData, RA::JSON::Init FeInit = RA::JSON::Init::Both);

        BSON::Result::Update UpdateOne (const BSON::Value& FoFind, const BSON::Value& FoReplace);
        BSON::Result::Update UpdateMany(const BSON::Value& FoFind, const BSON::Value& FoReplace);

        BSON::Result::Delete DeleteOne(const BSON::Data& FnDocument);
        BSON::Result::Delete DeleteMany(const BSON::Data& FnDocument);

        RA::JSON Sort(const xstring& FoKey, const int FnDirection, const RA::JSON::Init FeInit);
        RA::JSON Aggrigate(const BSON::Pipeline& FoPipeline, const RA::JSON::Init FeInit);

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
inline pint RA::Stash::Count(const K& FxKey, const V& FxValue)
{
    Begin();
    BSON::Pipeline Pipeline{};
    Pipeline.match(BSON::MakeDocument(BSON::KVP(FxKey, FxValue)));
    BSON::Cursor Cursor = MoCollection.aggregate(Pipeline, BSON::Aggregate{});
    pint Count = 0;
    for (auto& Val : Cursor)
        Count++;
    return Count;
    RescueThrow();
}

template<typename K, typename V>
inline RA::JSON RA::Stash::Match(const K& FxKey, const V& FxValue, RA::JSON::Init FeInit)
{
    Begin();
    BSON::Pipeline Pipeline{};
    Pipeline.match(BSON::MakeDocument(BSON::KVP(FxKey, FxValue)));
    BSON::Cursor Cursor = MoCollection.aggregate(Pipeline, BSON::Aggregate{});
    return RA::Stash::CursorToJSON(Cursor, FeInit);
    RescueThrow();
}