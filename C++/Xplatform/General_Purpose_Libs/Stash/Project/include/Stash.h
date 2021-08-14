#pragma once

#include "JSON.h" // Include First
#include "xstring.h"


#define MongoOpenDoc(__DOC__) BSON::Start{} << __DOC__ << BSON::OpenDoc
#define MongoSetDoc()         BSON::Start{} << "$set" << BSON::OpenDoc
#define MongoCloseDoc()       BSON::CloseDoc << BSON::Finish

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
        void DropCollection();

        BSON::Result::InsertOne  operator<<(const BSON::Value& FoView);
        BSON::Result::InsertOne  operator<<(const xstring& FoJsonStr);
        //BSON::Result::InsertMany operator<<(const BSON::Document& FoDocument);
       
        static RA::JSON CursorToJSON(BSON::Cursor& FoCursor, RA::JSON::Init FeInit);
        RA::JSON GetAll(RA::JSON::Init FeInit = RA::JSON::Init::Both);
        uint Count(const BSON::Value& FnData);
        RA::JSON FindOne(const BSON::Data& FnData, RA::JSON::Init FeInit = RA::JSON::Init::Both);
        RA::JSON FindMany(const BSON::Data& FnData, RA::JSON::Init FeInit = RA::JSON::Init::Both);

        BSON::Result::Update UpdateOne (const BSON::Value& FoFind, const BSON::Value& FoReplace);
        BSON::Result::Update UpdateMany(const BSON::Value& FoFind, const BSON::Value& FoReplace);

        BSON::Result::Delete DeleteOne(const BSON::Data& FnDocument);
        BSON::Result::Delete DeleteMany(const BSON::Data& FnDocument);

        RA::JSON Sort(const BSON::Value& FoFind, const RA::JSON::Init FeInit);

    private:
        std::string          MoURL;

        mongocxx::uri        MoURI;
        mongocxx::client     MoClient;

        mongocxx::database   MoDatabase;
        mongocxx::collection MoCollection;

        static mongocxx::instance SoInstance;
    };
}
