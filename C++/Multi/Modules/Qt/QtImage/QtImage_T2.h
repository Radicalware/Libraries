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

#include "QtImage_T1.h"
#include "Trim.h"

template<typename T>
class QtImage_T2 : public QtImage_T1 //, public QObject // Standard Qt Libs start with 'Q' not 'Qt'
{
protected:
	//Q_OBJECT;
	IMG img;

	T* m_handler;
	QSizePolicy* m_expanding;

public:
	QtImage_T2();
	virtual ~QtImage_T2();
private:
	void init();
public:
	inline QtImage_T2(QObject* parent, T* label, const std::string& on_image,
		const std::string& off_image = "", Trim::Layout layout = Trim::full_pic);

	inline QtImage_T2(QObject* parent, T* label, const QString& on_image,
		const QString& off_image = "", Trim::Layout layout = Trim::full_pic);

	inline void operator=(const QtImage_T2& image);

	inline IMG image() const;
	inline T* handler() const;

	inline void update_pixel_count(QEvent::Type ev = QEvent::Type::Leave);

	inline int height() const;
	inline int width() const;
};


template<typename T>
QtImage_T2<T>::QtImage_T2() { }

template<typename T>
inline QtImage_T2<T>::~QtImage_T2() {
	//delete m_expanding;
}

template<typename T>
void QtImage_T2<T>::init() {
	m_expanding = new QSizePolicy();
	m_expanding->setHorizontalStretch(1);
	m_expanding->setVerticalStretch(1);

	m_last_size = img.on_pix.size();
	_original.width = img.on_pix.width();
	_original.height = img.on_pix.height();
	m_handler->setMinimumSize(1, 1);
	img.on_pix.scaled(_original.width, _original.height);
}


template<typename T>
QtImage_T2<T>::QtImage_T2(QObject* parent, T* label, const std::string& on_image,
	const std::string& off_image, Trim::Layout layout) : m_handler(label)
{
	img = IMG(on_image, off_image);
	trim.layout = layout;
	this->init();
}

template<typename T>
QtImage_T2<T>::QtImage_T2(QObject* parent, T* label, const QString& on_image,
	const QString& off_image, Trim::Layout layout) : m_handler(label)
{
	img = IMG(on_image, off_image);
	trim.layout = layout;
	this->init();
}

template<typename T>
void QtImage_T2<T>::operator=(const QtImage_T2& other) {
	m_expanding = new QSizePolicy();
	m_expanding->setHorizontalStretch(100);
	m_expanding->setVerticalStretch(100);

	m_last_size = other.last_size();
	_original.width = other.original_width();
	_original.height = other.original_height();
	trim = other.trim;

	img = other.image();

	m_handler = other.handler();
	m_handler->setMinimumSize(1, 1);
	img.on_pix.scaled(_original.width, _original.height);
}

template<typename T>
IMG QtImage_T2<T>::image() const {
	return img;
}

template<typename T>
T* QtImage_T2<T>::handler() const {
	return m_handler;
}

template<typename T>
void QtImage_T2<T>::update_pixel_count(QEvent::Type ev) {

	//this->QtImage_T1::calc_ratio(m_handler->width(), m_handler->height());
    if (ev == QEvent::Leave) {
		m_handler->setPixmap(img.off_pix.scaled(m_handler->width()*0.97, m_handler->height()*0.97, Qt::KeepAspectRatio));
	}
	else {
		m_handler->setPixmap(img.on_pix.scaled(m_handler->width()*0.97, m_handler->height()*0.97, Qt::KeepAspectRatio));
	}
}

template<>
void QtImage_T2<QPushButton>::update_pixel_count(QEvent::Type ev) {

	QSize out_size = QSize(m_handler->width()*0.97, m_handler->height()*0.97);

	m_handler->setText("");
	if (trim.layout == Trim::full_box) {
		if (ev == QEvent::Leave)
			m_handler->setIcon(QIcon(trim.crop(img.off_pix, out_size, _original.width, _original.height)));
		else
			m_handler->setIcon(QIcon(trim.crop(img.on_pix, out_size, _original.width, _original.height)));
	}
	else if (trim.layout == Trim::full_pic) {
		if (ev == QEvent::Leave)
			m_handler->setIcon(img.off_icon);
		else
			m_handler->setIcon(img.on_icon);
	}

	m_handler->setIconSize(out_size);
}

template<typename T>
int QtImage_T2<T>::height() const {
	return m_handler->height();
}

template<typename T>
int QtImage_T2<T>::width() const {
	return m_handler->width();
}
