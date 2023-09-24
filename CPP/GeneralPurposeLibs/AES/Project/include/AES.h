#pragma once

/*
*|| Copyright[2021][Joel Leagues aka Scourge]
*|| Scourge /at\ protonmail /dot\ com
*|| www.Radicalware.net
*|| https://www.youtube.com/channel/UCivwmYxoOdDT3GmDnD0CfQA/playlists
*||
*|| Licensed under the Apache License, Version 2.0 (the "License");
*|| you may not use this file except in compliance with the License.
*|| You may obtain a copy of the License at
*||
*|| http ://www.apache.org/licenses/LICENSE-2.0
*||
*|| Unless required by applicable law or agreed to in writing, software
*|| distributed under the License is distributed on an "AS IS" BASIS,
*|| WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*|| See the License for the specific language governing permissions and
*|| limitations under the License.
*/

#include<chrono>

#include "RawMapping.h"
#include "xvector.h"
#include "xstring.h"
#include "re2/re2.h"
#include "xmap.h"

namespace RA
{
	class EXI AES
	{
	public:
		AES(const xint FnEncryptionSize = 0); // Takes Smallest Possible Size
		~AES();

		AES(const RA::AES&  Other);
		AES(      RA::AES&& Other) noexcept;

		void operator=(const RA::AES&  Other);
		void operator=(      RA::AES&& Other) noexcept;

		AES& Encrypt();
		xstring Decrypt();

		AES& SetPlainText(const xstring& FsPlainText);
		AES& SetCipherText(const xstring& FsCipherText);

		AES& SetAllRandomValues();
		void SetRandomAAD();
		void SetRandomKey();
		void SetRandomIV();

		RIN void SetAAD(const xstring& Input) { MsAAD = Input; }
		RIN void SetKey(const xstring& Input) { MsKey = Input; }
		RIN void SetIV( const xstring& Input) { MsIV  = Input; }
		RIN void SetTag(const xstring& Input) { MsTag = Input; }

		RIN xstring GetAAD() const { return MsAAD; }
		RIN xstring GetKey() const { return MsKey; }
		RIN xstring GetIV()  const { return MsIV; }
		RIN xstring GetTag() const { return MsTag; }

		RIN xstring GetPlainText()  const { return MsPlaintext;  }
		RIN xstring GetCipherText() const { return MsCipherText; }
		RIN xstring GetCipherTextByteCode() const { return MsCipherText.ToByteCode().Sub(NullByteRex.Get(), ""); }

	private:
		inline static xp<RE2> NullByteRex = RA::MakeShared<RE2>(R"((\\x00))");
		// inline static std::shared_ptr<RE2> NullByteRex2 = std::shared_ptr<RE2>(new RE2(R"((\\x00))"));
		RA::SharedPtr<unsigned char[]> MsTagPtr = nullptr;
		xstring MsPlaintext;
		xstring MsCipherText;
		xstring MsAAD;
		xstring MsKey;
		xstring MsIV;
		xstring MsTag;
		xint    MnEncryptionSize = 0;
		xint    MnCipherTextSize = 0;

		void ThrowErrors() const;
	};
};
