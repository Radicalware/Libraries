#pragma once

#include "JSON.h" // Include First

#include "xstring.h"

//#if (defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64))
//    #ifdef DLL_EXPORT
//        #define EXI __declspec(dllexport)
//        #define DllExport __declspec(dllexport)
//    #else
//        #define EXI __declspec(dllimport)
//        #define DllImport __declspec(dllimport)
//    #endif
//#else
//    #define EXI
//#endif


#define MongoInfo BSON::Start{} << "info" << BSON::OpenDoc
#define MongoSet  BSON::Start{} << "$set" << BSON::OpenDoc
#define MongoEnd  BSON::CloseDoc << BSON::Finish

namespace RA
{
    class Stash
    {
    public:
        Stash(const xstring& URL = "mongodb://localhost:27017");

        //Stash(const Stash& Other);
        //Stash(Stash&& Other) noexcept;

        //void operator=(const Stash& Other);
        //void operator=(Stash&& Other) noexcept;

        Stash& SetDatabase(const xstring& FsDatabase);
        Stash& SetCollection(const xstring& FsCollection);

        BSON::Result::InsertOne  operator<<(const BSON::Value& FoView);
        BSON::Result::InsertOne  operator<<(const xstring& FoJsonStr);
        //BSON::Result::InsertMany operator<<(const BSON::Document& FoDocument);


        template<typename T, typename Arg>
        static void JoinDocument(T& FoDocument, Arg&& FoArg);
        template<typename T, typename... Args>
        static void JoinDocument(T& FoDocument, Args&&... FoArgs);

        template<typename ...Args>
        static BSON::Value CreateInfoDocument(Args&& ...FoArgs);

        //template<typename ...Args>
        //static BSON::Value CreateSetDocument(Args ...FoArgs);
       
        static RA::JSON CursorToJSON(BSON::Cursor& FoCursor, RA::JSON::Init FeInit);
        RA::JSON GetAll(RA::JSON::Init FeInit = RA::JSON::Init::Both);
        RA::JSON FindOne(const BSON::Data& FnData, RA::JSON::Init FeInit = RA::JSON::Init::Both);
        RA::JSON FindMany(const BSON::Data& FnData, RA::JSON::Init FeInit = RA::JSON::Init::Both);

        BSON::Result::Update UpdateOne (const BSON::Value& FoFind, const BSON::Value& FoReplace);
        BSON::Result::Update UpdateMany(const BSON::Value& FoFind, const BSON::Value& FoReplace);

        BSON::Result::Delete DeleteOne(const BSON::Data& FnDocument);
        BSON::Result::Delete DeleteMany(const BSON::Data& FnDocument);

    private:


        mongocxx::uri          MoURI;
        mongocxx::client       MoClient;

        mongocxx::database     MoDatabase;
        mongocxx::collection   MoCollection;

        static mongocxx::instance SoInstance;
    };
}
