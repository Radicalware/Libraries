#pragma once

// QtImage v1.0.0

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
#include <QPixmap>
#include <QImage>
#include <QIcon>
#include <QSizePolicy>

class Base_QtImage
{
protected:

    int m_width;
    int m_height;

    int m_new_width;
    int m_new_height;

    inline void calc_ratio(int box_width, int box_height);
};

void Base_QtImage::calc_ratio(int box_width, int box_height) {

    m_new_width = m_width;
    m_new_height = m_height;

    if (box_width < m_new_width || box_height < m_new_height) {
        if (box_width <= m_new_width) {
            m_new_width = box_width;
            // ?? / width = m_hight / m_width; >> [(width * m_height) / m_width]
            m_new_height = (m_new_width * m_new_height) / m_new_width;
        }
        if (box_height <= m_new_height) {
            m_new_height = box_height;
            m_new_width = (m_new_height * m_new_width) / m_new_height;
        }
    }
}

template<typename T>
class QtImage : public Base_QtImage // Actual Qt Libs start with 'Q' not 'Qt'
{
    std::string m_image_str;
    QString m_image_qs;
    QPixmap m_pix;

    T* m_image;
    T* m_original_label;


public:

    QtImage();
    void init();
    QtImage(const std::string& image, T* label);
    QtImage(const QString& image, T* label);

    void operator=(const std::string& image);
    void operator=(const QString& image);
    void operator=(const QtImage& image);

    std::string str() const;
    QString qstr() const;
    QPixmap pix() const;
    T* label() const;

    inline void update_pixel_count();

    int height() const;
    int width() const;
};


template<typename T>
QtImage<T>::QtImage() { }


template<typename T>
void QtImage<T>::init() {
    m_original_label = m_image;

    m_pix = QPixmap(m_image_qs);
    m_width = m_pix.width();
    m_height = m_pix.height();
    m_image->setMinimumSize(1, 1);
    m_pix.scaled(m_width, m_height, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

template<typename T>
QtImage<T>::QtImage(const std::string& image, T* label) :
    m_image_str(image), m_image(label) {
    m_image_qs = QString(m_image_str.c_str());
    this->init();
}

template<typename T>
QtImage<T>::QtImage(const QString& image, T* label) :
    m_image_qs(image), m_image(label) {
    m_image_str = m_image_qs.toStdString();
    this->init();
}

template<typename T>
void QtImage<T>::operator=(const std::string& image) {
    m_image_str = image;
    m_image_qs = QString(m_image_str.c_str());
    this->init();
}

template<typename T>
void QtImage<T>::operator=(const QString& image) {
    m_image_qs = image;
    m_image_str = m_image_qs.toStdString();
    this->init();
}

template<typename T>
void QtImage<T>::operator=(const QtImage& image) {
    m_image_str = image.str();
    m_image_qs = image.qstr();
    m_image = image.label();
    this->init();
}

template<typename T>
std::string QtImage<T>::str() const {
    return m_image_str;
}

template<typename T>
QString QtImage<T>::qstr() const {
    return m_image_qs;
}

template<typename T>
QPixmap QtImage<T>::pix() const {
    return m_pix;
}

template<typename T>
T* QtImage<T>::label() const {
    return m_image;
}

template<typename T>
void QtImage<T>::update_pixel_count() {

    this->Base_QtImage::calc_ratio(m_original_label->width(), m_original_label->height());

    m_image->setPixmap(m_pix.scaled(m_new_width, m_new_height, Qt::KeepAspectRatio));
}

template<>
void QtImage<QPushButton>::update_pixel_count() {

    this->Base_QtImage::calc_ratio(m_original_label->width(), m_original_label->height());

    m_image->setText("");
    m_image->setIcon(QIcon(m_pix));
    m_image->setIconSize(QSize(m_new_width, m_new_height));
}

template<typename T>
int QtImage<T>::height() const {
    return m_height;
}

template<typename T>
int QtImage<T>::width() const {
    return m_height;
}

