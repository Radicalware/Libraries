#pragma once

/*
* Copyright[2019][Joel Leagues aka Scourge]
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

#include <QSize>

struct Trim
{
	enum Layout {
		full_pic,
		full_box
	};

	enum Side {
		none,
		width,
		height
	};

	int crop_pixils = 0;
	Layout layout = full_pic;
	QSize max_qsize;
	const QSize const_max_size = QSize(16777215, 16777215);
	inline void operator=(const Trim& other);

	template<typename ImgT>
	inline ImgT crop(const ImgT& icon, const QSize& out_size, int orig_width, int orig_height);
};



template<typename ImgT>
ImgT  Trim::crop(const ImgT& icon, const QSize& out_size, int orig_width, int orig_height) {

	float icon_ratio = static_cast<float>(orig_width) / static_cast<float>(orig_height);
	float out_ratio = static_cast<float>(out_size.width()) / static_cast<float>(out_size.height());

	if (icon_ratio > out_ratio) {
		int i_width = icon.width();
		int i_height = icon.height();
		int calc_width = i_height * out_size.width() / out_size.height();
		crop_pixils = i_width - calc_width;
		return icon.copy(QRect(crop_pixils, 0, icon.width() - crop_pixils, icon.height()));
	}
	else if (icon_ratio < out_ratio) {
		int i_width = icon.width();
		int i_height = icon.height();

		int calc_height = i_width * out_size.height() / out_size.width();
		crop_pixils = i_height - calc_height;
		return icon.copy(QRect(0, crop_pixils, icon.width(), icon.height() - crop_pixils));
	}

	return icon.copy(QRect(0, 0, icon.width(), icon.height()));
}

void Trim::operator=(const Trim& other) {
	layout = other.layout;
	max_qsize = other.max_qsize;
}