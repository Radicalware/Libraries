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


#include "IMG.h"

#include <QSize>
#include <QPixmap>
#include <string>


struct Style
{
public:
	struct Trim {
		enum Layout {
			full_pic,
			full_box
		};

		enum Side {
			none,
			width,
			height
		};
		const std::string standard_image = "image: url(";
		const std::string bordered_image = "border-image: url(";

		Layout layout = full_pic;
		int crop_pixils = 0;
		int* m_current_size;
		
		Trim();
		Trim(int** current_size, QPixmap* off_img_ptr, QPixmap* on_img_ptr, QEvent::Type* hover_status_ptr);
		void image_qss(QSize* current_size, const std::string* img_ptr, std::string& ret);

	private:
		QPixmap* m_on_img_ptr;
		QPixmap* m_off_img_ptr;
		QEvent::Type* m_hover_status_ptr;
	};

	struct Font
	{	// https://doc.qt.io/archives/qt-5.10/stylesheet-reference.html#font-style
		std::string other  = "";
		std::string family = "Arial Rounded MT Bold";
		std::string weight = "normal";
		std::string style  = "normal";
		std::string color = "white"; // light grey
		//std::string color  = "qradialgradient(cx:0, cy:0, radius: 1, fx:0.5, fy : 0.5, stop : 0 blue, stop : 1 red)"; // blue / red
		std::string align  = "center";

		int text_size = 10;
		float div_size = 10;

		std::string qss();
	};
	Trim* trim;
	Font* font;

	std::string m_saved_styleSheet;
	const int max_size = 16777215;
	int* m_current_size;

	// -------------------------------------------------------------------------------------------------
	Style();
	Style(int** i_current_size, Trim::Layout i_layout, IMG* image, QEvent::Type* hover_status);
	~Style();
	void operator=(const Style& other);
	std::string retStyleSheet(QSize* current_size, const std::string* img_ptr);
	void saveStyleSheet();
	// -------------------------------------------------------------------------------------------------
};

