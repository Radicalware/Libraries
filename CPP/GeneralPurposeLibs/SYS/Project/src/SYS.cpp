﻿/*
* Copyright[2018][Joel Leagues aka Scourge]
* Scourge /at\ protonmail /dot\ com
* www.Radicalware.com
* https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http ://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

// -------------------------------------------------------------------------------
// ALERT iterator will majorly slow down your performance if you don't
// optimize your compiler settings "-O2", else it will increase speed when
// not on windows (windows will give you even speeds with optimization else lag)
// also, on windows, be sure to remove the debugging for the iterator. 
// -------------------------------------------------------------------------------

#include "SYS.h"

// logical lok at how the KVP system works under 2 keys going to the same values
// char key map ---| -- int map --- | --- values
// str  key map ---|

RA::SYS::SYS(int argc, char** argv, char** env)
{
    Begin();
    SetArgs(argc, argv);
    if (env != nullptr)
    {
        ThrowIt("Not Implemented Yet");
    }
    Rescue();
};

// ======================================================================================================================
// >>>> args


void RA::SYS::SetArgs(int argc, char** argv)
{
    Begin();
    MbArgsSet = true;
    MvCliArgs.reserve(argc);
    for (int i = 0; i < argc; i++)
    {
        auto LsArg = xstring(argv[i]);
        if (IsArgType(LsArg))
            MvCliArgs << LsArg.ToLower();
        else
            MvCliArgs << std::move(LsArg);
    }
    MnSize = MvCliArgs.Size();

    size_t program_split_loc = MvCliArgs[0].find_last_of("/\\");
    MsPath = MvCliArgs[0].substr(0, program_split_loc + 1);
    MsFile = MvCliArgs[0].substr(program_split_loc + 1, MvCliArgs[0].size() - 1);

    for (xint i = 1; i < MnSize; i++)
    {
        const bool bLastArg = (i == (MnSize - 1));
        xstring& Arg = MvCliArgs[i];
        if (HasDoubleDash(Arg))
        {
            MvKeysStr << &Arg;
            if (MmAliasSC.Has(Arg))
                MvKeysChr << MmAliasSC.Key(Arg);
            if (bLastArg)
            {
                RA::XMapAddKeyArrIdx(MmArgs, MmAliasSC.Key(Arg), xstring::StaticClass);
                break;
            }

            for (xint j = i + 1; j < MnSize; j++)
            {
                xstring& SubArg = MvCliArgs[j];
                if (IsArgType(SubArg))
                {
                    i = j - 1; // -1 because the for-loop will increase it again
                    break;
                }
                if (MmAliasSC.Has(Arg))
                    RA::XMapAddKeyArrIdx(MmArgs, MmAliasSC.Key(Arg), SubArg);
            }
        }
        else if (HasSingleDash(Arg))
        {
            for (const char& ChrArg : Arg(1))
            {
                MvKeysChr << ChrArg; // Char Keys
                for (auto& Pair : MmAliasSC) // Str Keys
                {
                    if (Pair.second == ChrArg)
                        MvKeysStr << &Pair.first;
                }
                if (bLastArg) // Map
                {
                    RA::XMapAddKeyArrIdx(MmArgs, ChrArg, xstring::StaticClass);
                    continue;
                }

                // Add args to ever char in char list
                for (xint j = i + 1; j < MnSize; j++)
                {
                    xstring& SubArg = MvCliArgs[j];
                    if (IsArgType(SubArg))
                    {
                        if(&ChrArg == &Arg.back())
                            i = j - 1;
                        break;
                    }
                    RA::XMapAddKeyArrIdx(MmArgs, ChrArg, SubArg);
                }
            }
        }
    }

    for (auto [Key, Val] : MmAliasSC)
        MmAliasCS.AddPair(Val, Key);

    Rescue();
}

void RA::SYS::AddAlias(const char FChr, const xstring& FStr)
{
    MmAliasSC.AddPair(FStr.ToLower(), FChr);
}

// -------------------------------------------------------------------------------------------------------------------

xvector<xstring> RA::SYS::ArgV() const {
    return MvCliArgs;
}
xstring RA::SYS::ArgV(const size_t Idx) const
{
    Begin();
    if (!MvCliArgs.HasIndex(Idx))
        ThrowIt("Read the help menu; you have not entered enough arguments");
    return MvCliArgs[Idx];
    Rescue();
}

bool RA::SYS::Arg(const size_t Idx, const char FChar) const
{
    if (!MvCliArgs.HasIndex(Idx))
        return false;
    if (MvCliArgs[Idx].Size() < 2)
        return false;
    if (MvCliArgs[Idx][0] == '-' && MvCliArgs[Idx][1] != '-')
        return MvCliArgs[Idx].Has(FChar);
    else
        return MmArgs.Has(FChar) && MmArgs.Has(MmAliasSC.Key(MvCliArgs[Idx]));
}
xvector<char> RA::SYS::GetChrKeys() const { return MvKeysChr; }
xvector<const xstring*> RA::SYS::GetStrKeys() const { return MvKeysStr; }
// -------------------------------------------------------------------------------------------------------------------
xstring RA::SYS::FullPath() { return MvCliArgs[0]; }
xstring RA::SYS::Path() { return MsPath; }
xstring RA::SYS::File() { return MsFile; }
// -------------------------------------------------------------------------------------------------------------------
xvector<xstring> RA::SYS::Key(const xstring& FKey) const
{
    Begin();
    const auto LoKey = FKey.ToLower();
    if (!MvKeysStr.Has(LoKey))
        ThrowIt("CLI argument not found for key: ", FKey);
    return MmArgs.Key(MmAliasSC.Key(LoKey));
    Rescue();
}
xvector<xstring> RA::SYS::Key(const char FKey) const
{
    Begin();
    if (!MmArgs.Has(FKey))
        ThrowIt("CLI argument not found for key: ", FKey);
    return MmArgs.Key(FKey);
    Rescue();
}
bool RA::SYS::HasArgs() const { return MnSize > 1; }
// -------------------------------------------------------------------------------------------------------------------

bool RA::SYS::Has(const xstring& FKey) const 
{
    const auto LoKey = FKey.ToLower();
    return MvKeysStr.Has(LoKey) || (MmAliasSC.Has(LoKey) && (MmArgs.Has(MmAliasSC.at(LoKey)) || MvKeysChr.Has(MmAliasSC.at(LoKey))));
}

bool RA::SYS::Has(const char FKey) const {
    return MmArgs.Has(FKey) || MvKeysChr.Has(FKey);
}

bool RA::SYS::HasVal(const xstring& FKey) const
{
    return Has(FKey) && Key(FKey).First().size() >= 1;
}

bool RA::SYS::HasVal(const char FKey) const
{
    return Has(FKey) && Key(FKey).First().size() >= 1;
}

xstring RA::SYS::GetVal(const xstring& FKey) const
{
    Begin();
    if (!HasVal(FKey))
        ThrowIt("Does not have Arg: ", FKey);
    return Key(FKey)[0];
    Rescue();
}

xstring RA::SYS::GetVal(const char FKey) const
{
    Begin();
    if (!HasVal(FKey))
        ThrowIt("Does not have Arg: ", FKey);
    return Key(FKey)[0];
    Rescue();
}

// -------------------------------------------------------------------------------------------------------------------

xvector<xstring> RA::SYS::operator[](const xstring& FKey)
{
    return The.Key(FKey);
}

xvector<xstring> RA::SYS::operator[](const char FKey)
{
    return The.Key(FKey);
}

xstring RA::SYS::operator[](int FKey)
{
    Begin();
    if (!MvCliArgs.HasIndex(FKey))
        ThrowIt("CLI argument not found for key: ", FKey);
    return MvCliArgs[FKey];
    Rescue();
}

bool RA::SYS::operator()(const xstring& FKey) const
{
    return MvKeysStr.Has(FKey.ToLower());
}

bool RA::SYS::operator()(const xstring& FKey, const xstring& FValue) const
{
    Begin();
    const char ChrArg = MmAliasSC.Key(FKey.ToLower());
    return The(ChrArg, FValue);
    Rescue();
}

bool RA::SYS::operator()(const char FKey) const
{
    return MvKeysChr.Has(FKey);
}

bool RA::SYS::operator()(const char FKey, const xstring& FValue) const
{
    Begin();
    if (!MvKeysChr.Has(FKey))
        return false;
    if (!MmArgs.Key(FKey).Has(FValue))
        return false;
    return true;
    Rescue();
}

// -------------------------------------------------------------------------------------------------------------------
bool RA::SYS::Help()
{
    Begin();
    if (MnSize == 1 || MvCliArgs.MatchOne(R"(^[-]{1,2}[hH]((elp)?)$)")) {
        return true;
    }
    else if (!MbArgsSet)
    {
        ThrowIt("\n\nRemember SetArgs(int argc, char** argv) !!\n\n");
    }
    else {
        return false;
    }
    Rescue();
}

bool RA::SYS::HasSingleDash(const xstring& FsArg) const
{
    Begin();
    if (!FsArg || FsArg.Match(R"(^\-[\d\.]+$)")) // false if no size or negative number
        return false;
    return (FsArg[0] == '-'); // else do real test
    Rescue();
}

bool RA::SYS::HasDoubleDash(const xstring& FsArg) const
{
    if (FsArg.Size() <= 1)
        return false;
    return (FsArg[0] == '-' && FsArg[1] == '-');
}
bool RA::SYS::IsArgType(const xstring& FsArg) const
{
    Begin();
    return HasSingleDash(FsArg);
    Rescue();
}
// ======================================================================================================================

RA::SYS CliArgs;