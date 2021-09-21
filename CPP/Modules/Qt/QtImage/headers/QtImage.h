#pragma once

// QtImage v2.0.0

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
#include "Style.h"


#include <string>
#include <QEvent>

class QtImage : public QObject
{
protected:
    Q_OBJECT // Q_OBJECT can't be a Templated Class, hence the need for virtualization
	enum Interactor {
        e_Label,
		e_PushButton,
		e_ToolButton
	};
	Interactor m_interactor;
	QEvent::Type m_hover_status_tmp = QEvent::Type::Enter;

	virtual bool eventFilter(QObject* watched, QEvent* event) Q_DECL_OVERRIDE final;
public:
	QEvent::Type m_hover_status = QEvent::Type::Enter;
	// you can't access an addr function as const so just leave the var public

	QtImage();
	virtual ~QtImage();
	
	inline virtual void* handler() const = 0; // can't return template virtual value directly
	inline virtual QSize size() const = 0;
	inline virtual void update_size() = 0;
	inline virtual IMG* img() const = 0;
	inline virtual Style* style() const = 0;
	inline virtual Style::Trim* trim() const = 0;
	inline virtual Style::Font* font() const = 0;

	inline virtual void set_layout(Style::Trim::Layout layout) = 0;
	inline virtual void set_text(std::string input) = 0;
	inline virtual Style::Font* ret_font() = 0;
};

//// eventFilter >> triggered all the time
//// this will update the image if the image changes
//// of there is a change in moues-over, then reflect the changes
//
//// update_size >> triggered when the user changes the size of the objct
//
//
//// 	QtImage* m_banner_img = new QtImageT<QLabel>(ui.lb_banner, image_dir + "banner.png");
//// 	QtImage* m_banner_img = new QtImage_QLabel(ui.lb_banner, image_dir + "banner.png");
