#include "Stash.h"
#include "Macros.h"

Mongo::Instance RA::Stash::SoInstance{};


 RA::Stash::Stash(const xstring& URL)
 {
     MoURL    = URL;
     MoURI    = Mongo::URI(MoURL);
     MoClient = Mongo::Client(MoURI);
 }

 RA::Stash::Stash(const Stash& Other)
 {
     *this = Other;
 }

 RA::Stash::Stash(Stash&& Other) noexcept
 {
     *this = std::move(Other);
 }

 void RA::Stash::operator=(const Stash& Other)
 {
     MoURL        = Other.MoURL;
     MoURI        = Mongo::URI(MoURL);
     MoClient     = Mongo::Client(MoURI);

     MoDatabase   = Other.MoDatabase;
     MoCollection = Other.MoCollection;
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
     MoDatabase = MoClient[FsDatabase.c_str()];
     return *this;
     RescueThrow();
 }

 RA::Stash& RA::Stash::SetCollection(const xstring& FsCollection)
 {
     Begin();
     MoCollection = MoDatabase[FsCollection.c_str()];
     return *this;
     RescueThrow();
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
     RescueThrow();
 }

 BSON::Result::InsertOne RA::Stash::operator<<(const BSON::Value& FoView)
 {
     Begin();
     return MoCollection.insert_one(FoView.view());
     RescueThrow();
 }

 BSON::Result::InsertOne RA::Stash::operator<<(const xstring& FoJsonStr)
 {
     Begin();
     return *this << bsoncxx::from_json(FoJsonStr.c_str());
     RescueThrow();
 }

 //BSON::Result::InsertMany RA::Stash::operator<<(const BSON::Document& FoDocument)
 //{
 //   return MoCollection.insert_many(FoDocument);
 //}

 RA::JSON RA::Stash::CursorToJSON(BSON::Cursor& FoCursor, RA::JSON::Init FeInit)
 {
     Begin();
     xstring Json;
     bool LbFirstPass = true;
     for (const BSON::View& Document : FoCursor)
     {
         if (LbFirstPass)
             LbFirstPass = false;
         else
             Json += ',';
         Json += bsoncxx::to_json(Document);
     }

     if(!Json.Size())
         return RA::JSON();
     return RA::JSON(Json, FeInit);
     RescueThrow();
 }

 RA::JSON RA::Stash::GetAll(RA::JSON::Init FeInit)
 {
     Begin();
     return  RA::Stash::CursorToJSON(MoCollection.find({}), FeInit);
     RescueThrow();
 }

 RA::JSON RA::Stash::FindOne(const BSON::Data& FnData, RA::JSON::Init FeInit)
 {
     Begin();
     return RA::JSON(MoCollection.find_one(FnData).get(), FeInit);
     RescueThrow();
 }

 RA::JSON RA::Stash::FindMany(const BSON::Data& FnData, RA::JSON::Init FeInit)
 {
     Begin();
     return RA::Stash::CursorToJSON(MoCollection.find(FnData), FeInit);
     RescueThrow();
 }

 BSON::Result::Update RA::Stash::UpdateOne(const BSON::Value& FoFind, const BSON::Value& FoReplace)
 {
     Begin();
     return MoCollection.update_one(FoFind.view(), FoReplace.view());
     RescueThrow();
 }

 BSON::Result::Update RA::Stash::UpdateMany(const BSON::Value& FoFind, const BSON::Value& FoReplace)
 {
     Begin();
     return MoCollection.update_many(FoFind.view(), FoReplace.view());
     RescueThrow();
 }

 BSON::Result::Delete RA::Stash::DeleteOne(const BSON::Data& FnDocument)
 {
     Begin();
     return MoCollection.delete_one(FnDocument);
     RescueThrow();
 }

 BSON::Result::Delete RA::Stash::DeleteMany(const BSON::Data& FnDocument)
 {
     Begin();
     return MoCollection.delete_many(FnDocument);
     RescueThrow();
 }

RA::JSON RA::Stash::Sort(const xstring& FoKey, const int FnDirection, const RA::JSON::Init FeInit)
{
    Begin();
    BSON::Pipeline Pipeline{};
    Pipeline.sort(MongoKVP(FoKey.ToStdString(), FnDirection));
    BSON::Cursor Cursor = MoCollection.aggregate(Pipeline, BSON::Aggregate{});
    return RA::Stash::CursorToJSON(Cursor, FeInit);
    RescueThrow();
}

RA::JSON RA::Stash::Aggrigate(const BSON::Pipeline& FoPipeline, const RA::JSON::Init FeInit)
{
    Begin();
    BSON::Cursor Cursor = MoCollection.aggregate(FoPipeline, BSON::Aggregate{});
    return RA::Stash::CursorToJSON(Cursor, FeInit);
    RescueThrow();
}
