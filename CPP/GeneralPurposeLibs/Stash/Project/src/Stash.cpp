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
    auto LoIndexBuilder = BSON::Builder();
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

RA::BSON::Result::InsertOne RA::Stash::operator<<(const BSON::Value& FoView)
{
    Begin();
    return MoCollection.insert_one(FoView.view());
    Rescue();
}

RA::BSON::Result::InsertOne RA::Stash::operator<<(const xstring& FoJsonStr)
{
    Begin();
    return *this << bsoncxx::from_json(FoJsonStr.c_str());
    Rescue();
}

RA::JSON RA::Stash::CursorToJSON(BSON::Cursor& FoCursor)
{
    Begin();
    std::ostringstream Json;
    xint Count = 0;

    Json << '[';
    for (const BSON::View& Builder : FoCursor)
        Json << bsoncxx::to_json(Builder);
    Json << ']';

    if(Count == 0)
        return RA::JSON();

    if (Count == 1)
    {
        auto JsonStr = xstring(Json);
        return RA::JSON(JsonStr(1, -1));
    }

    return RA::JSON(Json);

    Rescue();
}

RA::JSON RA::Stash::ToJSON(const EOrder FnOrder)
{
    Begin();
    auto LoCursor = MoCollection.find(
        BSON::NewBuilderKVP(),
        mongocxx::options::find{}.sort(BSON::NewKVP("_id", static_cast<int>(FnOrder)))
    );

    auto LsJSON = xstring();
    int LnCount = 0;
    for (auto&& doc : LoCursor)
    {
        if (LnCount == 1)
        {
            LsJSON.insert(0, "[");
        }
        LsJSON += bsoncxx::to_json(doc) + ',';
        if(LnCount == 0)
            ++LnCount;
        else if (LnCount == 1)
            ++LnCount;
    }
    if(LnCount > 1)
        LsJSON.Last() = ']';
    else
        LsJSON.erase(LsJSON.size() - 1);
    return LsJSON;
    Rescue();
}

RA::JSON RA::Stash::FindOne(const BSON::Data& FnData)
{
    Begin();
    const auto LoJson = MoCollection.find_one(FnData);
    if (!LoJson || LoJson->length() == 0)
        return RA::JSON();
    return RA::JSON(LoJson.value());
    Rescue();
}

RA::JSON RA::Stash::FindOne(const BSON::Value& FnData)
{
    return FindOne(RA::BSON::ToData(FnData));
}

RA::JSON RA::Stash::FindOne(const BSON::Builder& FnData)
{
    Begin();
    const auto LoJson = MoCollection.find_one(FnData.view());
    if (!LoJson || LoJson->length() == 0)
        return RA::JSON();
    return RA::JSON(LoJson.value());
    Rescue();
}

RA::JSON RA::Stash::FindMany(const BSON::Data& FnData)
{
    Begin();
    auto Data = MoCollection.find(FnData);
    return RA::Stash::CursorToJSON(Data);
    Rescue();
}

RA::JSON RA::Stash::FindMany(const BSON::Value& FnData)
{
    return FindMany(RA::BSON::ToData(FnData));
}

// AppendDocument(BSON::Builder().extract()).extract());
RA::BSON::Builder RA::Stash::AppendDocument(const BSON::Builder& FoSource) {
    auto LoTarget = BSON::Builder();
    for (const auto& element : FoSource.view())
        LoTarget.append(BSON::KVP(std::string(element.key().data(), element.key().length()), element.get_value()));
    return LoTarget;
}

// WARNING!! NEED std::map to be auto-sorted
std::map<RA::Date, xmap<xstring, double>> RA::Stash::TimeSeriesObjToMap(const RA::JSON& FoJson)
{
    Begin();
    auto LmResults = std::map<RA::Date, xmap<xstring, double>>();
    for (const auto& item : FoJson.AsObject()) {
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
        LmResults[RA::Date(RA::ETimeZone::UTC, item.first)] = LmVals;
    }
    return LmResults;
    Rescue();
}

std::map<RA::Date, xmap<xstring, double>> RA::Stash::TimeSeriesArrToMap(const RA::JSON& FoJson)
{
    Begin();
    auto LmResults = std::map<RA::Date, xmap<xstring, double>>();
    for (const web::json::value& LoJSON : FoJson.AsArray()) 
    {
        auto LmVals = xmap<xstring, double>();
        xp<RA::Date> LoDatePtr;
        for (const auto& Pair : LoJSON.as_object())
        {
            if (Pair.first == L"_id")
                LoDatePtr.New(RA::ETimeZone::UTC, Pair.second.as_integer());
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

RA::BSON::Result::InsertMany RA::Stash::InsertMany(const xvector<BSON::Value>& FvBSON)
{
    Begin();
    return MoCollection.insert_many(FvBSON);
    Rescue();
}

RA::BSON::Result::Update RA::Stash::UpdateOne(const BSON::View& FoFind, const BSON::View& FoReplace)
{
    Begin();
    return MoCollection.update_one(FoFind, FoReplace);
    Rescue();
}

RA::BSON::Result::Update RA::Stash::UpdateMany(const BSON::View& FoFind, const BSON::View& FoReplace)
{
    Begin();
    return MoCollection.update_many(FoFind, FoReplace);
    Rescue();
}

RA::BSON::Result::Update RA::Stash::UpsertOne(const BSON::View& FoFilter, const BSON::Value& FoUpdate) {
    Begin();
    return The.MoCollection.update_one(
        FoFilter,
        BSON::NewKVP("$set", FoUpdate),
        BSON::Options{}.upsert(true)
    );
    Rescue();
}

RA::BSON::Result::Update RA::Stash::UpsertOne(const BSON::View& FoFilter, const BSON::Builder& FoUpdate) {
    Begin();
    return The.MoCollection.update_one(
        FoFilter,
        BSON::NewKVP("$set", FoUpdate),
        BSON::Options{}.upsert(true)
    );
    Rescue();
}

xmap<RA::Date::EpochTime, RA::BSON::Result::Update>
RA::Stash::UpsertTimeSeries(const xmap<RA::Date, xmap<xstring, double>>& FoDataMap, const xstring& FsAppend) {
    Begin();
    auto LvResults = xmap<RA::Date::EpochTime, BSON::Result::Update>();
    for (const auto& LmDateVec : FoDataMap) 
    {
        const auto LsTime = BSON::MakeEpochOID(LmDateVec.first.GetEpochTime());
        const auto LoFilter = BSON::NewKVP("_id", LsTime);
        auto LoInnerDocBuilder = BSON::Builder();
        LoInnerDocBuilder.append(BSON::KVP("DateTime", LmDateVec.first.RetStr(RA::ETimeFormat::DateTime).ToStdString()));
        for (const auto& Pair : LmDateVec.second)
            LoInnerDocBuilder.append(BSON::KVP(Pair.first.ToStdString() + FsAppend, Pair.second));
        LvResults.AddPair(
            LmDateVec.first.GetEpochTime(),
            UpsertOne(LoFilter, LoInnerDocBuilder));
    }
    return LvResults;
    Rescue();
}

RA::BSON::Result::Delete RA::Stash::DeleteOne(const BSON::View& FnDocument)
{
    return MoCollection.delete_one(FnDocument);
}

RA::BSON::Result::Delete RA::Stash::DeleteOne(const BSON::Data& FnDocument)
{
    Begin();
    return MoCollection.delete_one(FnDocument);
    Rescue();
}

RA::BSON::Result::Delete RA::Stash::DeleteMany(const BSON::View& FnDocument)
{
    return MoCollection.delete_many(FnDocument);
}

RA::BSON::Result::Delete RA::Stash::DeleteMany(const BSON::Data& FnDocument)
{
    Begin();
    return MoCollection.delete_many(FnDocument);
    Rescue();
}


RA::JSON RA::Stash::Sort(const xstring& FoKey, const int FnDirection)
{
    Begin();
    BSON::Pipeline Pipeline{};
    Pipeline.sort(MongoKVP(FoKey.ToStdString(), FnDirection));
    BSON::Cursor Cursor = MoCollection.aggregate(Pipeline, BSON::Aggregate{});
    return RA::Stash::CursorToJSON(Cursor);
    Rescue();
}

RA::JSON RA::Stash::Aggrigate(const BSON::Pipeline& FoPipeline)
{
    Begin();
    BSON::Cursor Cursor = MoCollection.aggregate(FoPipeline, BSON::Aggregate{});
    return RA::Stash::CursorToJSON(Cursor);
    Rescue();
}

