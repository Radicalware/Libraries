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

void RA::Stash::SetUniqueIndex(const xstring& Key, const bool FbAscending)
{
    Begin()
    auto LoIndexBuilder = BSON::Document();
    LoIndexBuilder.append(BSON::KVP(Key.ToStdString(), static_cast<int>(FbAscending))); // 1 for ascending order
    auto LoOptions = mongocxx::options::index{};
    if(Key != "_id")
        LoOptions.unique(true);
    MoCollection.create_index(LoIndexBuilder.view(), LoOptions);
    Rescue([&]() { cout << "Exception: " << "Failed to Index Collection " << MsCollection << endl; });
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

RA::JSON RA::Stash::CursorToJSON(BSON::Cursor& FoCursor, RA::JSON::Init FeInit)
{
    Begin();
    std::ostringstream Json;
    xint Count = 0;

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

RA::JSON RA::Stash::GetAll(const int FnOrder, RA::JSON::Init FeInit)
{
    Begin();
    auto LoCursor = MoCollection.find(
        BSON::MakeDocument(),
        mongocxx::options::find{}.sort(BSON::MakeDocument(bsoncxx::builder::basic::kvp("_id", FnOrder)))
    );
    auto LoJSON = xstring('[');
    for (auto&& doc : LoCursor)
        LoJSON += bsoncxx::to_json(doc) + ',';
    if (LoJSON.Last() == ',')
        LoJSON.Last() = ']';
    else
        LoJSON += ']';
    return LoJSON;
    Rescue();
}

RA::JSON RA::Stash::FindOne(const BSON::Data& FnData, RA::JSON::Init FeInit)
{
    Begin();
    return RA::JSON(MoCollection.find_one(FnData).value(), FeInit);
    Rescue();
}

RA::JSON RA::Stash::FindMany(const BSON::Data& FnData, RA::JSON::Init FeInit)
{
    Begin();
    auto Data = MoCollection.find(FnData);
    return RA::Stash::CursorToJSON(Data, FeInit);
    Rescue();
}

// AppendDocument(BSON::Document().extract()).extract());
BSON::Document RA::Stash::AppendDocument(const BSON::Document& FoSource) {
    auto LoTarget = BSON::Document();
    for (const auto& element : FoSource.view())
        LoTarget.append(BSON::KVP(std::string(element.key().data(), element.key().length()), element.get_value()));
    return LoTarget;
}

// WARNING!! NEED std::map to be auto-sorted
std::map<RA::Date, xmap<xstring, double>> RA::Stash::TimeSeriesObjToMap(const RA::JSON& FoJson)
{
    Begin();
    auto LmResults = std::map<RA::Date, xmap<xstring, double>>();
    for (const auto& item : FoJson.GetJsonObject()) {
        const auto& LoDailyData = item.second.as_object();
        auto LmVals = xmap<xstring, double>();
        for (const auto& Pair : LoDailyData)
        {
            if (xstring(Pair.first).Match(SoDateTimeRex)) // Pair.first is wchar*
                continue;
            LmVals.AddPair(
                xstring(Pair.first).Sub(re2::RE2(R"(^\d+\. )"), ""),
                std::stod(Pair.second.as_string())); // << std::stod(LmVals.second.as_string());
        }
        LmResults[RA::Date(item.first)] = LmVals;
    }
    return LmResults;
    Rescue();
}

std::map<RA::Date, xmap<xstring, double>> RA::Stash::TimeSeriesArrToMap(const RA::JSON& FoJson)
{
    Begin();
    auto LmResults = std::map<RA::Date, xmap<xstring, double>>();
    for (const auto& LoJSON : FoJson.GetJsonArray()) 
    {
        auto LmVals = xmap<xstring, double>();
        xp<RA::Date> LoDatePtr;
        for (const auto& Pair : LoJSON.as_object())
        {
            if (Pair.first == L"_id")
                LoDatePtr = MKP<RA::Date>(Pair.second.as_integer());
            else if (xstring(Pair.first).Match(SoDateTimeRex)) // Pair.first is wchar*
                continue;
            else
                LmVals.AddPair(
                    xstring(Pair.first).Sub(re2::RE2(R"(^\d+\. )"), ""),
                    std::stod(Pair.second.to_string())); // << std::stod(LmVals.second.as_string());
        }
        LmResults[*LoDatePtr] = LmVals;
    }
    return LmResults;
    Rescue();
}

BSON::Result::InsertMany RA::Stash::InsertMany(const xvector<BSON::Value>& FvBSON)
{
    Begin();
    return MoCollection.insert_many(FvBSON);
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

BSON::Result::Update RA::Stash::UpsertOne(const BSON::Document& FoUpdate, const BSON::Value& FoFilter) {
    Begin();
    return The.MoCollection.update_one(
        FoFilter.view(),
        BSON::MakeDocument(
            BSON::KVP("$set", FoUpdate)
        ),
        BSON::Options{}.upsert(true)
    );
    Rescue();
}

xmap<RA::Date::EpochTime, BSON::Result::Update>
RA::Stash::UpsertTimeSeries(const xmap<RA::Date, xmap<xstring, double>>& FoDataMap, const xstring& FsAppend) {
    Begin();
    auto LvResults = xmap<RA::Date::EpochTime, BSON::Result::Update>();
    for (const auto& LmDateVec : FoDataMap) {
        //auto LoFilter = BSON::MakeDocument(BSON::KVP("_id", BSON::OID(LmDateVec.first.GetEpochTimeStr().Ptr(), 12)));
        const auto LnTime = static_cast<int>(LmDateVec.first.GetEpochTime());
        const auto LoFilter = BSON::MakeDocument(BSON::KVP("_id", LnTime));
        auto LoInnerDocBuilder = BSON::Document();
        LoInnerDocBuilder.append(BSON::KVP("DateTime", LmDateVec.first.GetStr().ToStdString()));
        for (const auto& Pair : LmDateVec.second)
            LoInnerDocBuilder.append(BSON::KVP(Pair.first.ToStdString() + FsAppend, Pair.second));
        if (!LvResults.Has(LnTime))
            LvResults.AddPair(
                LnTime,
                UpsertOne(LoInnerDocBuilder, LoFilter));
        else
            cout << "Error: Duplicate Key: " << LnTime << endl;
    }
    return LvResults;
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

