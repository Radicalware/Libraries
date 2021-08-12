// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Stash.h"  // Include First

#include "Macros.h"


using std::cout;
using std::endl;



int main() 
{
    Nexus<>::Start();

    Begin();
    RA::Stash Stash;
    Stash.SetDatabase("MyDatabase");
    Stash.SetCollection("MyCollection");
    Stash.DropCollection();
    Stash.SetCollection("MyCollection");

    BSON::Value Document = BSON::Start{}
        << "name" << "MongoDB"
        << "type" << "database"
        << "count" << 1

        << "versions" << BSON::OpenArray
            << "v3.2"
            << "v3.0"
            << "v2.6"
        << BSON::CloseArray

        << "info" << BSON::OpenDoc
            << "x" << 203
            << "y" << 102
        << BSON::CloseDoc
    << BSON::Finish;

    std::cout << '\n';
    xstring Line = "-----------------------------------------------------";
    Line.Print();

    // insert
    BSON::Result::InsertOne InsertResult = Stash << Document;
    if (InsertResult) {
        std::cout << "New document ID: " 
            << InsertResult->inserted_id().get_oid().value.to_string() << "\n";
        Line.Print();
    }

    RA::JSON Result = Stash.GetAll();
    Result.GetPrettyJson().Print();

    auto tmp = Stash << xstring(R"({ "_id" : { "oid" : "F0fe4538ab7b00008b605b02" }, "name" : "Inline String", "type" : "database", "count" : 1, "versions" : [ "v3.2", "v3.0", "v2.6" ], "info" : { "x" : 203, "y" : 102 } })");

    Stash.FindOne(BSON::Start{} << "name" << "Inline String" << BSON::Finish).GetPrettyJson().Print();

    BSON::Value Find    = MongoOpenDoc("Info") << "x"      << 203 <<      "y" << 102 << MongoCloseDoc();
    BSON::Value Replace = MongoSetDoc()        << "info.x" <<   0 << "info.y" <<   0 << MongoCloseDoc();
    Stash.UpdateMany(Find, Replace);

    // Delete (note that both x and y need to be correct, delete won't work if you only add x
    Stash.DeleteOne( MongoOpenDoc("Info") << "x" <<   0 << "y" <<   0 << MongoCloseDoc());
    Stash.DeleteMany(MongoOpenDoc("Info") << "x" << 203 << "y" << 102 << MongoCloseDoc());
    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
