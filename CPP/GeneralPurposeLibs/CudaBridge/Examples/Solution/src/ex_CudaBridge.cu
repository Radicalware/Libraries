#include <iostream>
#include <vector>

#ifdef UsingVLD
#include <vld.h>
#endif

#include "Memory.h"

#pragma warning (disable : 6011 6387)

using namespace std;


namespace Vortex
{
    namespace Tracer
    {
        typedef struct
        {
            size_t MnSanityCheck;    // {3} >> (123/456/789) // bear,true,bull
            double MnTraceDeviation; // {0} >> double (Shift.Size() / Shift.Size())
            double MnPriceDeviation; // {1} ex: 0.98 = bearish && 1.05 = bullish
            double MnOccurances;     // {0}
        } Direction;

        typedef struct
        {
            size_t MnSanityCheck; // {987654321} // if this isn't equal, our struct is screwed up;
            size_t MnTotalTracerCount; // {0}
            size_t MnUsedTracerCount;  // {0}

            Direction* MoBear; // 123
            Direction* MoTrue; // 456
            Direction* MoBull; // 789

            short* MvConfig; // {nullptr} terminator = 0; unused = 1; Down Value = 2; Up Value = 3
            //               // idx 0 = 55 as sanity check

        } Object;

        void PvInitDirection(Direction& FoDirection, size_t FnSanity)
        {
            FoDirection.MnSanityCheck = FnSanity;
            FoDirection.MnTraceDeviation = 0;
            FoDirection.MnPriceDeviation = 1;
            FoDirection.MnOccurances = 0;
        }

        void Initialize(Object& State, size_t FnTotalTracerCount, size_t FnUsedTracerCount)
        {
            State.MnTotalTracerCount = FnTotalTracerCount;
            State.MnUsedTracerCount = FnUsedTracerCount;

            State.MoBear   = new Direction();
            State.MoTrue   = new Direction();
            State.MoBull   = new Direction();
            State.MvConfig = new short[FnTotalTracerCount + 1];

            for (size_t i = 0; i < FnTotalTracerCount; i++)
                State.MvConfig[i] = i;
            State.MvConfig[FnTotalTracerCount] = 0;

            State.MnSanityCheck = 987654321;
            Vortex::Tracer::PvInitDirection(*State.MoBear, 123);
            Vortex::Tracer::PvInitDirection(*State.MoTrue, 456);
            Vortex::Tracer::PvInitDirection(*State.MoBull, 789);
        }

        void Delete(Object& State)
        {
            delete[] State.MoBear;
            delete[] State.MoTrue;
            delete[] State.MoBull;
            delete[] State.MvConfig;
        }

        size_t GetMemSize(size_t FnTotalTracerCount) { // sizeof(uint) needed as a buffer space
            return sizeof(Object) + (sizeof(Direction) * 3) + sizeof(short) * FnTotalTracerCount + sizeof(uint); 
        }
        size_t GetMemSize(Object& State) { // sizeof(uint) needed as a buffer space
            return sizeof(Object) + (sizeof(Direction) * 3) + sizeof(short) * State.MnTotalTracerCount + sizeof(uint);
        }
    }
}

void TestVec()
{
    vector<Vortex::Tracer::Object*> Objs;

    Objs.push_back(new Vortex::Tracer::Object());
    Objs.push_back(new Vortex::Tracer::Object());
    Objs.push_back(new Vortex::Tracer::Object());

    const size_t LnTotalTracers = 5;

    for (auto* Obj : Objs)
        Vortex::Tracer::Initialize(*Obj, LnTotalTracers, LnTotalTracers);

    cout << Objs[0]->MnSanityCheck << endl;
    cout << Objs[1]->MnSanityCheck << endl;
    cout << Objs[2]->MnSanityCheck << endl;

    for (auto* Obj : Objs)
        Vortex::Tracer::Delete(*Obj);
}

int main()
{
    const size_t LnObjCount = 3;
    const size_t LnTotalTracers = 5;

    auto var = RA::SharedPtr<Vortex::Tracer::Object*>(LnObjCount);
    auto SharedObjs = RA::SharedPtr<Vortex::Tracer::Object*>(LnObjCount)
        .Initialize(&Vortex::Tracer::Initialize, LnTotalTracers, LnTotalTracers)
        .Destroy(&Vortex::Tracer::Delete);

    for (auto& Obj : SharedObjs)
    {
        cout << Obj.MnSanityCheck << endl;
        cout << Obj.MoBear->MnSanityCheck << endl;
        cout << Obj.MoTrue->MnSanityCheck << endl;
        cout << Obj.MoBull->MnSanityCheck << endl;
    }

    const auto MemSize = Vortex::Tracer::GetMemSize(LnTotalTracers);
    auto* Objs = (Vortex::Tracer::Object*)malloc(LnObjCount * MemSize);
    memcpy(Objs, SharedObjs.Ptr(), MemSize + sizeof(uint));

    cout << "Looping" << endl;
    for (int o = 0; o < LnObjCount; o++)
    {
        auto& Obj = Objs[o];
        cout << Obj.MnSanityCheck << endl;
        cout << Obj.MoBear->MnSanityCheck << endl;
        cout << Obj.MoTrue->MnSanityCheck << endl;
        cout << Obj.MoBull->MnSanityCheck << endl;

        for (int t = 0; t < LnTotalTracers; t++)
            cout << Obj.MvConfig[t] << ' ';

        cout << endl;
    }

    cout << "Deleting" << endl;
    //for (int i = 0; i < LnObjCount; i++)
    //    Vortex::Tracer::Delete(Objs[i]);
    free(Objs);
    cout << "Done" << endl;
    return 0;
}