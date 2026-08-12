// Copyright[2021][Joel Leagues aka Scourge] under the Apache V2 Licence

#include "Stash.h"  // Include First

#include "Macros.h"
#include <map>
#include "Timer.h"

using std::cout;
using std::endl;


void BuildStreamDoc(RA::Stash& FoStash)
{
    Begin()
    PrintStart();
    FoStash.DropCollection();
    FoStash.SetCollection("MyCollection");
    RA::BSON::Value LoStreamBuilt = RA::BSON::Start{}
        << "name" << "MongoDB"
        << "type" << "database"
        << "count" << 1
        << "versions" << RA::BSON::OpenArray
            << "v1.1"
            << "v1.2"
            << "v1.3"
        << RA::BSON::CloseArray
        << "info" << RA::BSON::OpenDoc
            << "x" << 4
            << "y" << 5
        << RA::BSON::CloseDoc 
    << RA::BSON::Finish;

    RA::BSON::Result::InsertOne LoResult = FoStash << LoStreamBuilt;
    FoStash.ToJSON().PrintJson();
    Rescue();
}

void BuildValueDoc(RA::Stash& FoStash)
{
    Begin()
    PrintStart();
    RA::Date::SetAutoFormat(true);
    RA::Date::SetAutoCreateStrings(true);

    FoStash.DropCollection();
    FoStash.SetCollection("MyCollection");

    // MoDate   = RA::Date(static_cast<xint>(It.at(L"_id").as_object().at(L"oid").as_number().to_uint64()));
    // MoDate   = RA::Date(FoJSON.at(0).as_number().to_uint64());

    auto LoTime = RA::GetTodayLOC();
    cout << LoTime.GetEpochTime() << endl;
    cout << LoTime << endl;
    auto LoDocument = RA::BSON::NewValue(
        "_id", LoTime.GetEpochTimeInt32(),
        "name", "MongoDB",
        "type", "database",
        "count", 1,
        "versions", xvector{ "v500.5","v600.5", "v700.5" },
        "info", xmap { kvps
            { "x", 800 },
            { "y", 900 } }
    );

    auto LoResult = FoStash << LoDocument;
    if (LoResult)
        std::cout << "New Int ID: " << RA::BSON::GetResultInt(LoResult) << "\n";
    cout << RA::GetTodayLOC().GetEpochTime() << endl;
    FoStash.ToJSON().PrintJson();
    FoStash.UpsertOne(RA::BSON::NewKVP("_id", LoTime.GetEpochTimeInt32()),
        RA::BSON::ToValue(R"({
            "name" : "Upsert String", 
            "type" : "database", 
            "versions" : [ "555555555555" ], 
            "info" : { "x" : 777 } }
        )"));
    FoStash.ToJSON().PrintJson();
    // WARNING: you will notice that it overwrites any change to your top level key
    // So no change was done to "count" so it is still there
    // but for info we lost the "y" because it a new key was overwritten that didn't include it
    Rescue();
}

void BuildStringDoc(RA::Stash& FoStash)
{
    Begin()
    PrintStart();
    FoStash.SetCollection("MyCollection");
    auto LoResult = FoStash << xstring(
        R"({
            "name" : "My Inline String", 
            "type" : "database", 
            "count" : 1, 
            "versions" : [ "v3.2", "v3.0", "v2.6" ], 
            "info" : { "x" : 203, "y" : 102 } }
        )");
    if (LoResult)
        std::cout << "New Hex ID: " << RA::BSON::GetResultOID(LoResult) << "\n";
    Rescue();
}

void UpdateDocumentWithMap(RA::Stash& FoStash)
{
    Begin()
    PrintStart();
    FoStash.DropCollection();
    FoStash.SetCollection("MyCollection");
    BuildStringDoc(FoStash);
    FoStash.ToJSON().PrintJson();
    auto LoFind    = RA::BSON::NewValue(xmap{ kvps { "info.x", 203 }, { "info.y", 102 } });
    auto LoReplace = RA::BSON::NewReqValue(
        RA::BSON::EReq::Set,
        RA::BSON::NewValue("info", xmap{ kvps { "x", 0 }, { "y", 0 } }));
    FoStash.UpdateMany(LoFind, LoReplace);
    FoStash.ToJSON().PrintJson();
    Rescue();
}


void UpdateDocumentWithString(RA::Stash& FoStash)
{
    Begin()
    PrintStart();
    FoStash.DropCollection();
    FoStash.SetCollection("MyCollection");
    BuildStringDoc(FoStash);
    FoStash.ToJSON().PrintJson();
    auto LoFind = RA::BSON::ToValue(R"({ "info.x" : 203, "info.y" : 102 })"); 
    auto LoReplace = RA::BSON::NewReqValue(
        RA::BSON::EReq::Set,
        RA::BSON::NewValue("info", xmap{ kvps { "x", 0 }, { "y", 0 } }));
    FoStash.UpdateMany(LoFind, LoReplace);
    FoStash.UpsertOne(LoFind, RA::BSON::ToValue(
        R"({
            "name" : "My Upsert String", 
            "type" : "database", 
            "count" : 1, 
            "versions" : [ "555555555555" ], 
            "info" : { "x" : 777, "y" : 888 } }
        )"));
    FoStash.ToJSON().PrintJson();
    Rescue();
}

void GetPrintExamples(RA::Stash& FoStash)
{
    Begin()
    PrintStart();
    FoStash.DropCollection();
    FoStash.SetCollection("MyCollection");
    BuildStringDoc(FoStash);

    auto LoJson = FoStash.ToJSON();
    LoJson.PrintJson();

    auto LsName = LoJson.Key<xstring>(L"name");
    cout << "Name: " << LsName << endl; // Is String
    cout << "Name: " << LoJson.Key(L"name") << endl; // converts obj to string
    cout << "Name: " << LoJson.Key(L"name").ToString() << endl; // converts obj to string
    cout << "Name: " << LoJson.Key<xstring>(L"name") << endl; // Is String

    cout << "Versions[2] = " << LoJson.Key(L"versions").At<xstring>(2) << endl;

    for (auto& LsVersion : LoJson.GetArr<xstring>(L"versions"))
        cout << "Version: " << LsVersion << endl;

    Rescue();
}

void SetPrintValues(RA::Stash& FoStash)
{
    Begin()
    PrintStart();
    auto LoJson = RA::JSON();
    auto LnEpoch = RA::GetTodayLOC().GetEpochTimeInt32();
    LoJson[L"_id"] = LnEpoch;
    LoJson[L"Symbol"] = L"Token";
    LoJson[L"Current"] = 280.0;
    LoJson[L"Variance"] = RA::BSON::NewValue(
                "Spread", xmap{kvps
                    { "high", 285.0 }, 
                    { "low",  275.0 } },
                "Range", 10.0);

    FoStash << LoJson.ToBSON();
    FoStash.ToJSON().PrintJson();

    // 3 ways to get the same value from the JSON object
    cout << "Current: " << LoJson.Key<double>(L"Current")  << endl; // Key Cast
    cout << "Current: " << SC<double>(LoJson[L"Current"])  << endl; // static cast
    cout << "Current: " << LoJson[L"Current"].To<double>() << endl; // Converts obj to double

    auto LbSpread = FoStash
        .Has(RA::BSON::NewKVP("_id", LnEpoch), 
            L"Variance", L"Spread");

    if (LbSpread)
        FoStash
            .FindOne(RA::BSON::NewKVP("_id", LnEpoch))
            .Key(L"Variance")
            .Key(L"Spread")
            .PrintJson();
    else
        ThrowIt("(Date >> Variance >> Spread) not found");
    Rescue()
}

int main() 
{
    Nexus<>::Start();

    Begin();
    RA::Stash Stash;
    Stash.SetDatabase("MyDatabase");
    Stash.SetCollection("MyCollection");
    Stash.DropCollection();
    Stash.SetCollection("MyCollection");

    //BuildStreamDoc(Stash);
    //BuildValueDoc(Stash);
    //UpdateDocumentWithMap(Stash);
    //UpdateDocumentWithString(Stash);
    //GetPrintExamples(Stash);
    SetPrintValues(Stash);

    Stash.DropCollection();

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}
