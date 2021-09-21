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
#include "TextShadow.h"

#include "assert.h"

#include <QEvent>
#include <string>
#include <QString>
#include <QObject>

// boarder-image >> for Style::Trim::full_box
// image         >> for Style::Trim::full_pic

template<typename T>
class QtImageT : public QtImage
{
private:
	bool m_init = false;
	T* m_handler = nullptr;

	Style* m_style;
	IMG*   m_img;
	TextShadow* m_text_shadow;
	int* m_current_size;

public:
	// BUILDERS
	QtImageT();
	void init(const std::string& off_image, const std::string& on_image, Style::Trim::Layout layout);
	virtual ~QtImageT();
	explicit QtImageT(T* t_push_button,
		std::string on_image, std::string image2 = "", Style::Trim::Layout layout = Style::Trim::full_pic);
	explicit QtImageT(T* t_push_button,
		QString on_image, QString off_image = "", Style::Trim::Layout layout = Style::Trim::full_pic);

	inline void operator=(const QtImage& image);

	// METHODS
	inline virtual void update_size();

	// GETTERS
	inline virtual void* handler() const;
	inline virtual QSize size() const;
	inline virtual IMG* img() const;
	inline virtual Style* style() const;
	inline virtual Style::Trim* trim() const;
	inline virtual Style::Font* font() const;
	inline virtual Style::Trim::Layout layout() const;
	inline virtual void set_layout(Style::Trim::Layout layout);
	inline virtual void set_text(std::string input);
	inline virtual Style::Font* ret_font();
};

// =================================================================================

template<typename T>
QtImageT<T>::QtImageT() {}
template<typename T>
inline void QtImageT<T>::init(const std::string& off_image, const std::string& on_image, Style::Trim::Layout layout)
{
	m_handler->installEventFilter(this);
	m_current_size = new int[2]{ m_handler->size().width(),m_handler->size().height() };

	m_img = new IMG(&m_hover_status, off_image, on_image);
	m_style = new Style(&m_current_size, layout, m_img, &m_hover_status);
	m_text_shadow = new TextShadow;
	m_init = true;
	m_style->saveStyleSheet();
	m_handler->setStyle(m_text_shadow);
}

template<typename T>
QtImageT<T>::~QtImageT() {
	if (m_init) {
		delete m_img;
		delete m_style;
		delete m_text_shadow;
		delete[] m_current_size;
	}
};

template<typename T>
QtImageT<T>::QtImageT(T* t_handler, std::string off_image,
	std::string on_image, Style::Trim::Layout layout) :
	m_handler(t_handler)
{
	this->init(off_image, on_image, layout);
}

template<typename T>
QtImageT<T>::QtImageT(T* t_handler, QString off_image,
	QString on_image, Style::Trim::Layout layout) :
	m_handler(t_handler)
{
	//if(layout == Style::Trim::full_box)

	std::string off = off_image.toStdString();
	std::string on = on_image.toStdString();
	this->init(off, on, layout); // taking refernece (default is to use std::string)
}

template<typename T>
void QtImageT<T>::operator=(const QtImage& other)
{
	if (m_init)
		this->~QtImageT();

	m_img = new IMG;
	m_style = new Style;
	m_init = true;
	m_img = other.img();
	m_style = other.style();
}

template<typename T>
void QtImageT<T>::update_size()
{
	
	m_current_size[0] = m_handler->size().width();
	m_current_size[1] = m_handler->size().height();

	this->font()->text_size = static_cast<int>(1 + (this->size().width() / this->font()->div_size));
	if (m_hover_status == QEvent::Type::Enter)
		this->font()->align = "bottom"; // TODO: make enums
	else
		this->font()->align = "center";
	this->style()->saveStyleSheet();
	m_handler->setStyleSheet(m_style->retStyleSheet(&m_handler->size(), m_img->get_img_ptr()).c_str());
}

// ----------------- GETTERS ----------------------------------

template<typename T>
void* QtImageT<T>::handler() const
{
	return m_handler;
}

template<typename T>
inline QSize QtImageT<T>::size() const
{
	return m_handler->size();
}

template<typename T>
IMG* QtImageT<T>::img() const
{
	return m_img;
}

template<typename T>
Style* QtImageT<T>::style() const
{
	return m_style;
}

template<typename T>
Style::Trim* QtImageT<T>::trim() const
{
	return m_style->trim;
}

template<typename T>
Style::Font* QtImageT<T>::font() const
{
	return m_style->font;
}

template<typename T>
Style::Trim::Layout QtImageT<T>::layout() const
{
	return m_style->trim->layout;
}

template<typename T>
inline void QtImageT<T>::set_layout(Style::Trim::Layout layout)
{
	//if (layout == Style::Trim::Layout::full_box) {
	//	if (m_style->trim->layout != Style::Trim::Layout::full_box)
	//		this->installEventFilter(this);
	//}
	//else if (layout == Style::Trim::Layout::full_pic) {
	//	if (m_style->trim->layout == Style::Trim::Layout::full_box)
	//		this->removeEventFilter(this);
	//}
	m_style->trim->layout = layout;
}

template<typename T>
inline void QtImageT<T>::set_text(std::string input)
{
	m_handler->setText(input.c_str());
}

template<typename T>
inline Style::Font* QtImageT<T>::ret_font()
{
	return m_style->font;
}



