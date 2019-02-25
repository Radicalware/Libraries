#pragma once

// QtImage v1.2.0

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

#include <QObject>
#include <QPushButton>
#include <QEvent>

#include "QtImage_T2.h"
#include "QtImage_T1.h"
#include "Trim.h"

#include <string>

class QtImage : public QObject , public QtImage_T2<QPushButton>
{
    Q_OBJECT;

public:
	QtImage();
    explicit QtImage(QObject* parent, QPushButton* t_push_button, 
		std::string image1, std::string image2 = "", Trim::Layout layout = Trim::full_pic);

	explicit QtImage(QObject* parent, QPushButton* t_push_button,
		QString image1, QString image2 = "", Trim::Layout layout = Trim::full_pic);

	~QtImage();
    virtual bool eventFilter(QObject* watched, QEvent* event) Q_DECL_OVERRIDE;
};



