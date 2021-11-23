/*
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

SYS::SYS(int argc, char** argv, char** env)
{
    Begin();
    (this)->SetArgs(argc, argv);
    if (env != nullptr)
    {
        ThrowIt("Not Implemented Yet");
    }
    Rescue();
};

// ======================================================================================================================
// >>>> args


void SYS::SetArgs(int argc, char** argv)
{
    Begin();
    MbArgsSet = true;
    MvCliArgs.reserve(argc);
    for (int i = 0; i < argc; i++)
        MvCliArgs << xstring(argv[i]);
    MnSize = MvCliArgs.Size();

    size_t program_split_loc = MvCliArgs[0].find_last_of("/\\");
    MsPath = MvCliArgs[0].substr(0, program_split_loc + 1);
    MsFile = MvCliArgs[0].substr(program_split_loc + 1, MvCliArgs[0].size() - 1);

    for (pint i = 1; i < MnSize; i++)
    {
        const bool bLastArg = (i == (MnSize - 1));
        xstring& Arg = MvCliArgs[i];
        if (HasDoubleDash(Arg))
        {
            MvKeysStr << &Arg;
            if (MmAliasChar.Has(Arg))
                MvKeysChr << MmAliasChar.Key(Arg);
            if (bLastArg)
            {
                RA::XMapAddToArray(MmArgs, MmAliasChar.Key(Arg), xstring::static_class);
                break;
            }

            for (pint j = i + 1; j < MnSize; j++)
            {
                xstring& SubArg = MvCliArgs[j];
                if (IsArgType(SubArg))
                {
                    i = j - 1; // -1 because the for-loop will increase it again
                    break;
                }
                if (MmAliasChar.Has(Arg))
                    RA::XMapAddToArray(MmArgs, MmAliasChar.Key(Arg), SubArg);
            }
        }
        else if (HasSingleDash(Arg))
        {
            for (const char& ChrArg : Arg(1))
            {
                MvKeysChr << ChrArg; // Char Keys
                for (auto& Pair : MmAliasChar) // Str Keys
                {
                    if (Pair.second == ChrArg)
                        MvKeysStr << &Pair.first;
                }
                if (bLastArg) // Map
                {
                    RA::XMapAddToArray(MmArgs, ChrArg, xstring::static_class);
                    continue;
                }

                // Add args to ever char in char list
                for (pint j = i + 1; j < MnSize; j++)
                {
                    xstring& SubArg = MvCliArgs[j];
                    if (IsArgType(SubArg))
                    {
                        if(&ChrArg == &Arg.back())
                            i = j - 1;
                        break;
                    }
                    RA::XMapAddToArray(MmArgs, ChrArg, SubArg);
                }
            }
        }
    }
    Rescue();
}

void SYS::AddAlias(const char FChr, const xstring& FStr)
{
    MmAliasChar.AddPair(FStr, FChr);
}

// -------------------------------------------------------------------------------------------------------------------
int SYS::ArgC() const {
    return MnSize;
}

xvector<xstring> SYS::ArgV() const {
    return MvCliArgs;
}
xstring SYS::ArgV(const size_t Idx) const
{
    Begin();
    if (!MvCliArgs.HasIndex(Idx))
        ThrowIt("Read the help menu; you have not entered enough arguments");
    return MvCliArgs[Idx];
    Rescue();
}

bool SYS::Arg(const size_t Idx, const char FChar) const
{
    if (!MvCliArgs.HasIndex(Idx))
        return false;
    if (MvCliArgs[Idx].Size() < 2)
        return false;
    if (MvCliArgs[Idx][0] == '-' && MvCliArgs[Idx][1] != '-')
        return MvCliArgs[Idx].Has(FChar);
    else
        return MmArgs.Has(FChar) && MmArgs.at(FChar).Has(MvCliArgs[Idx]);
}
xvector<char> SYS::GetChrKeys() const { return MvKeysChr; }
xvector<const xstring*> SYS::GetStrKeys() const { return MvKeysStr; }
// -------------------------------------------------------------------------------------------------------------------
xstring SYS::FullPath() { return MvCliArgs[0]; }
xstring SYS::Path() { return MsPath; }
xstring SYS::File() { return MsFile; }
// -------------------------------------------------------------------------------------------------------------------
xvector<xstring> SYS::Key(const xstring& FKey) const
{
    Begin();
    if (!MvKeysStr.Has(FKey))
        ThrowIt("CLI argument not found for key: ", FKey);
    return MmArgs.Key(MmAliasChar.Key(FKey));
    Rescue();
}
xvector<xstring> SYS::Key(const char FKey) const
{
    Begin();
    if (!MmArgs.Has(FKey))
        ThrowIt("CLI argument not found for key: ", FKey);
    return MmArgs.Key(FKey);
    Rescue();
}
bool SYS::HasArgs() const { return MnSize > 1; }
// -------------------------------------------------------------------------------------------------------------------

bool SYS::Has(const xstring& FKey) const {
    return MvKeysStr.Has(FKey) || MmArgs.Has(MmAliasChar.at(FKey)) || MvKeysChr.Has(MmAliasChar.at(FKey));
}

bool SYS::Has(const char FKey) const {
    return MmArgs.Has(FKey) || MvKeysChr.Has(FKey);
}

// -------------------------------------------------------------------------------------------------------------------

xvector<xstring> SYS::operator[](const xstring& FKey)
{
    return The.Key(FKey);
}

xvector<xstring> SYS::operator[](const char FKey)
{
    return The.Key(FKey);
}

xstring SYS::operator[](int FKey)
{
    Begin();
    if (!MvCliArgs.HasIndex(FKey))
        ThrowIt("CLI argument not found for key: ", FKey);
    return MvCliArgs[FKey];
    Rescue();
}

bool SYS::operator()(const xstring& FKey) const
{
    return MvKeysStr.Has(FKey);
}

bool SYS::operator()(const xstring& FKey, const xstring& FValue) const
{
    Begin();
    const char ChrArg = MmAliasChar.Key(FKey);
    return The(ChrArg, FValue);
    Rescue();
}

bool SYS::operator()(const char FKey) const
{
    return MvKeysChr.Has(FKey);
}

bool SYS::operator()(const char FKey, const xstring& FValue) const
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
bool SYS::Help()
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

bool SYS::HasSingleDash(const xstring& FsArg) const
{
    Begin();
    if (!FsArg.Size() || FsArg.Match(R"(^\-[\d\.]+$)"))
        return false;
    return (FsArg[0] == '-');
    Rescue();
}

bool SYS::HasDoubleDash(const xstring& FsArg) const
{
    if (FsArg.Size() <= 1)
        return false;
    return (FsArg[0] == '-' && FsArg[1] == '-');
}
bool SYS::IsArgType(const xstring& FsArg) const
{
    Begin();
    return HasSingleDash(FsArg);
    Rescue();
}
// ======================================================================================================================

