#include "Stash.h"


 mongocxx::instance RA::Stash::SoInstance{};


 RA::Stash::Stash(const xstring& URL)
 {
     MoURI = mongocxx::uri(URL);
     MoClient = mongocxx::client(MoURI);
 }

 //RA::Stash::Stash(const Stash& Other)
 //{
 //    *this = Other;
 //}

 //RA::Stash::Stash(Stash&& Other) noexcept
 //{
 //    *this = std::move(Other);
 //}

 //void RA::Stash::operator=(const Stash& Other)
 //{

 //}

 //void RA::Stash::operator=(Stash&& Other) noexcept
 //{
 //}

 RA::Stash& RA::Stash::SetDatabase(const xstring& FsDatabase)
 {
     MoDatabase = MoClient[FsDatabase.c_str()];
     return *this;
 }

 RA::Stash& RA::Stash::SetCollection(const xstring& FsCollection)
 {
     MoCollection = MoDatabase[FsCollection.c_str()];
     return *this;
 }

 BSON::Result::InsertOne RA::Stash::operator<<(const BSON::Value& FoView)
 {
     return MoCollection.insert_one(FoView.view());
 }

 BSON::Result::InsertOne RA::Stash::operator<<(const xstring& FoJsonStr)
 {
     return *this << bsoncxx::from_json(FoJsonStr.c_str());
 }

 //BSON::Result::InsertMany RA::Stash::operator<<(const BSON::Document& FoDocument)
 //{
 //   return MoCollection.insert_many(FoDocument);
 //}

 RA::JSON RA::Stash::CursorToJSON(BSON::Cursor& FoCursor, RA::JSON::Init FeInit)
 {
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
     return RA::JSON(Json, FeInit);
 }

 RA::JSON RA::Stash::GetAll(RA::JSON::Init FeInit)
 {
     return  RA::Stash::CursorToJSON(MoCollection.find({}), FeInit);
 }

 RA::JSON RA::Stash::FindOne(const BSON::Data& FnData, RA::JSON::Init FeInit)
 {
     return RA::JSON(MoCollection.find_one(FnData).get(), FeInit);
 }

 RA::JSON RA::Stash::FindMany(const BSON::Data& FnData, RA::JSON::Init FeInit)
 {
     return RA::Stash::CursorToJSON(MoCollection.find(FnData), FeInit);
 }

 BSON::Result::Update RA::Stash::UpdateOne(const BSON::Value& FoFind, const BSON::Value& FoReplace)
 {
     return MoCollection.update_one(FoFind.view(), FoReplace.view());
 }

 BSON::Result::Update RA::Stash::UpdateMany(const BSON::Value& FoFind, const BSON::Value& FoReplace)
 {
     return MoCollection.update_many(FoFind.view(), FoReplace.view());
 }

 BSON::Result::Delete RA::Stash::DeleteOne(const BSON::Data& FnDocument)
 {
     return MoCollection.delete_one(FnDocument);
 }

 BSON::Result::Delete RA::Stash::DeleteMany(const BSON::Data& FnDocument)
 {
     return MoCollection.delete_many(FnDocument);
 }
