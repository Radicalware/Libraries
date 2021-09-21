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

#include "QtImage.h"
#include "Style.h"
#include "IMG.h"

#include "assert.h"

#include <QEvent>
#include <string>
#include <QString>
#include <QLabel>

// boarder-image >> for Style::Trim::full_box
// image         >> for Style::Trim::full_pic


class QtImage_QLabel : public QtImage
{
private:
    bool m_init = false;
    QLabel* m_handler = nullptr;

	Style* m_style;
    IMG*   m_img;


public:
	// BUILDERS
    QtImage_QLabel();
    virtual ~QtImage_QLabel();
    explicit QtImage_QLabel(QLabel* t_push_button,
        std::string on_image, std::string image2 = "", Style::Trim::Layout layout = Style::Trim::full_pic);
    explicit QtImage_QLabel(QLabel* t_push_button,
        QString on_image, QString off_image = "", Style::Trim::Layout layout = Style::Trim::full_pic);

	inline void operator=(const QtImage& image);

	// METHODS
    virtual bool update_size(QObject* watched, QEvent* event);

	// GETTERS
	virtual void* handler() const;
    virtual IMG* img() const;
	virtual Style* style() const;
	virtual Style::Trim* trim() const;
	virtual Style::Font* font() const;
    virtual Style::Trim::Layout layout() const;

};


