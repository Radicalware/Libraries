#pragma once

// QtImage v1.1.0

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

#include <string>
#include <QString>
#include <QPoint>
#include <QPixmap>
#include <QImage>
#include <QIcon>
#include <QSizePolicy>
#include <QRect>
#include <QEvent>

#include <QObject>
#include <QPushButton>

#include "IMG.h"
#include "Trim.h"

class QtImage_T1
{
protected:
	struct Dim
	{
		int width;
		int height;
	};

	Dim _original;
	Dim _new;

    QSize m_last_size;
    Trim trim;

    inline void set_layout(Trim::Layout new_layout);

public:
    QSize last_size() const;
    int original_width() const;
    int original_height() const;
};

