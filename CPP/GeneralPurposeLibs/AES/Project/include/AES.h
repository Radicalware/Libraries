﻿#pragma once

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
		AES(const pint FnEncryptionSize = 0); // Takes Smallest Possible Size
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

		INL void SetAAD(const xstring& Input) { MsAAD = Input; }
		INL void SetKey(const xstring& Input) { MsKey = Input; }
		INL void SetIV(const xstring& Input)  { MsIV  = Input; }
		INL void SetTag(const xstring& Input) { MsTag = Input; }

		INL xstring GetAAD() const { return MsAAD; }
		INL xstring GetKey() const { return MsKey; }
		INL xstring GetIV()  const { return MsIV; }
		INL xstring GetTag() const { return MsTag; }

		INL xstring GetPlainText()  const { return MsPlaintext;  }
		INL xstring GetCipherText() const { return MsCipherText; }

	private:
		RA::SharedPtr<unsigned char[]> MsTagPtr;
		xstring MsPlaintext;
		xstring MsCipherText;
		xstring MsAAD;
		xstring MsKey;
		xstring MsIV;
		xstring MsTag;
		pint    MnEncryptionSize = 0;
		pint    MnCipherTextSize = 0;

		void ThrowErrors() const;
	};
};