#include "Stash.h"
#include "Macros.h"

#include <bson/bson.h>

Mongo::Instance RA::Stash::SoInstance{};


RA::Stash::Stash(const xstring& URL)
{
    Begin()
    MoURL    = URL;
    MoURI    = Mongo::URI(MoURL.c_str());
    MoClient = Mongo::Client(MoURI);
    Rescue()
}

RA::Stash::Stash(const Stash& Other)
{
    Begin()
    *this = Other;
    Rescue()
}

RA::Stash::Stash(Stash&& Other) noexcept
{
    *this = std::move(Other);
}

void RA::Stash::operator=(const Stash& Other)
{
    Begin()
    MoURL        = Other.MoURL;
    MoURI        = Mongo::URI(MoURL);
    MoClient     = Mongo::Client(MoURI);

    MoDatabase   = Other.MoDatabase;
    MoCollection = Other.MoCollection;
    Rescue()
}

void RA::Stash::operator=(Stash&& Other) noexcept
{
    MoURL        = std::move(Other.MoURL);
    MoURI        = Mongo::URI(MoURL);
    MoClient     = Mongo::Client(MoURI);

    MoDatabase   = std::move(Other.MoDatabase);
    MoCollection = std::move(Other.MoCollection);
}

RA::Stash& RA::Stash::SetDatabase(const xstring& FsDatabase)
{
    Begin();
    MsDatabase = FsDatabase;
    MoDatabase = MoClient[MsDatabase.c_str()];
    return *this;
    Rescue();
}

RA::Stash& RA::Stash::SetCollection(const xstring& FsCollection)
{
    Begin();
    if (!MsDatabase)
        ThrowIt("Database Not Set");
    MsCollection = FsCollection;
    MoCollection = MoDatabase[MsCollection.c_str()];
    return *this;
    Rescue([&]() { cout << "Exception: " << "Failed to Set Collection " << FsCollection << endl; });
}

Mongo::Database& RA::Stash::GetDatabase()
{
    return MoDatabase;
}

Mongo::Collection& RA::Stash::GetCollection()
{
    return MoCollection;
}

void RA::Stash::DropCollection()
{
    Begin();
    MoCollection.drop();
    Rescue();
}

BSON::Result::InsertOne RA::Stash::operator<<(const BSON::Value& FoView)
{
    Begin();
    return MoCollection.insert_one(FoView.view());
    Rescue();
}

BSON::Result::InsertOne RA::Stash::operator<<(const xstring& FoJsonStr)
{
    Begin();
    return *this << bsoncxx::from_json(FoJsonStr.c_str());
    Rescue();
}

 //BSON::Result::InsertMany RA::Stash::operator<<(const BSON::Document& FoDocument)
 //{
 //   return MoCollection.insert_many(FoDocument);
 //}

RA::JSON RA::Stash::CursorToJSON(BSON::Cursor& FoCursor, RA::JSON::Init FeInit)
{
    Begin();
    std::ostringstream Json;
    uint Count = 0;

    Json << '[';
    for (const BSON::View& Document : FoCursor)
    {
        if (Count > 0)
            Json << ',';
        Count++;
        const unsigned char* Chr = (const unsigned char*)Document.data();
        auto Len = Document.length();
        auto Num = *Document.data();

        bson_t bson;
        bson_init_static(&bson, Document.data(), Document.length());

        size_t size;
        auto LsCharJSON = bson_as_json(&bson, &size);
        if (!LsCharJSON)
            ThrowIt("Error converting to json");

        Json << LsCharJSON;

        bson_free(LsCharJSON);
    }
    Json << ']';

    if(Count == 0)
        return RA::JSON();

    if (Count == 1)
    {
        auto JsonStr = xstring(Json);
        return RA::JSON(JsonStr(1, -1), FeInit);
    }

    return RA::JSON(Json, FeInit);

    Rescue();
}

RA::JSON RA::Stash::GetAll(RA::JSON::Init FeInit)
{
    Begin();
    auto Data = MoCollection.find({});
    return  RA::Stash::CursorToJSON(Data, FeInit);
    Rescue();
}

RA::JSON RA::Stash::FindOne(const BSON::Data& FnData, RA::JSON::Init FeInit)
{
    Begin();
    return RA::JSON(MoCollection.find_one(FnData).get(), FeInit);
    Rescue();
}

RA::JSON RA::Stash::FindMany(const BSON::Data& FnData, RA::JSON::Init FeInit)
{
    Begin();
    auto Data = MoCollection.find(FnData);
    return RA::Stash::CursorToJSON(Data, FeInit);
    Rescue();
}

BSON::Result::Update RA::Stash::UpdateOne(const BSON::Value& FoFind, const BSON::Value& FoReplace)
{
    Begin();
    return MoCollection.update_one(FoFind.view(), FoReplace.view());
    Rescue();
}

BSON::Result::Update RA::Stash::UpdateMany(const BSON::Value& FoFind, const BSON::Value& FoReplace)
{
    Begin();
    return MoCollection.update_many(FoFind.view(), FoReplace.view());
    Rescue();
}

BSON::Result::Delete RA::Stash::DeleteOne(const BSON::Data& FnDocument)
{
    Begin();
    return MoCollection.delete_one(FnDocument);
    Rescue();
}

BSON::Result::Delete RA::Stash::DeleteMany(const BSON::Data& FnDocument)
{
    Begin();
    return MoCollection.delete_many(FnDocument);
    Rescue();
}

RA::JSON RA::Stash::Sort(const xstring& FoKey, const int FnDirection, const RA::JSON::Init FeInit)
{
    Begin();
    BSON::Pipeline Pipeline{};
    Pipeline.sort(MongoKVP(FoKey.ToStdString(), FnDirection));
    BSON::Cursor Cursor = MoCollection.aggregate(Pipeline, BSON::Aggregate{});
    return RA::Stash::CursorToJSON(Cursor, FeInit);
    Rescue();
}

RA::JSON RA::Stash::Aggrigate(const BSON::Pipeline& FoPipeline, const RA::JSON::Init FeInit)
{
    Begin();
    BSON::Cursor Cursor = MoCollection.aggregate(FoPipeline, BSON::Aggregate{});
    return RA::Stash::CursorToJSON(Cursor, FeInit);
    Rescue();
}

