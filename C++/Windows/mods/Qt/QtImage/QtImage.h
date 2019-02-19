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

class Base_QtImage
{
public:
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

        // y-from-top, x-from-left, total-width, total-height

        template<typename ImgT>
        ImgT  crop(const ImgT& icon, const QSize& out_size, int orig_width, int orig_height) {

            float icon_ratio = (float)orig_width / (float)orig_height;
            float out_ratio = (float)out_size.width() / (float)out_size.height();

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
        void operator=(const Trim& other) {
            layout = other.layout;
            max_qsize = other.max_qsize;
        }
    };

protected:
    int m_original_width;
    int m_original_height;

    int m_new_width;
    int m_new_height;

    QSize m_last_size;

    Trim trim;

    inline void set_layout(Trim::Layout new_layout);

public:
    inline QSize last_size() const;
    inline int original_width() const;
    inline int original_height() const;
};

inline QSize Base_QtImage::last_size() const{
    return m_last_size;
}

int Base_QtImage::original_width () const {
    return m_original_width;
}

int Base_QtImage::original_height() const {
    return m_original_height;
}

void Base_QtImage::set_layout(Trim::Layout new_layout){
    trim.layout = new_layout;
}


// **********************************************************************************************************************************
// **********************************************************************************************************************************

template<typename T>
class QtImage : public Base_QtImage // Standard Qt Libs start with 'Q' not 'Qt'
{
    std::string m_image_str;
    QString m_image_qs;
    QPixmap m_pix;

    T* m_handler;
    T* m_original_label;

    QSize* m_handler_size;

    QSizePolicy* m_expanding;

public:
    QtImage();
    ~QtImage();
private:
    void init();
public:
    QtImage(const std::string& image, T* label, Trim::Layout layout = Trim::full_pic);
    QtImage(const QString& image, T* label, Trim::Layout layout = Trim::full_pic);

    void operator=(const QtImage& image);

    std::string str() const;
    QString qstr() const;
    QPixmap pix() const;
    T* handler() const;

    inline void update_pixel_count();

    int height() const;
    int width() const;
};


template<typename T>
QtImage<T>::QtImage() { }

template<typename T>
inline QtImage<T>::~QtImage(){
    //delete m_expanding;
}

template<typename T>
void QtImage<T>::init() {


    m_expanding = new QSizePolicy();
    m_expanding->setHorizontalStretch(1);
    m_expanding->setVerticalStretch(1);

    m_pix = QPixmap(m_image_qs);
    m_last_size = m_pix.size();
    m_original_width = m_pix.width();
    m_original_height = m_pix.height();
    m_handler->setMinimumSize(1, 1);
    m_pix.scaled(m_original_width, m_original_height);
}


template<typename T>
QtImage<T>::QtImage(const std::string& image, T* label, Trim::Layout layout) :
    m_image_str(image), m_handler(label)
{
    trim.layout = layout;
    m_image_qs = QString(m_image_str.c_str());
    this->init();
}

template<typename T>
QtImage<T>::QtImage(const QString& image, T* label, Trim::Layout layout) :
    m_image_qs(image), m_handler(label)
{
    trim.layout = layout;
    m_image_str = m_image_qs.toStdString();
    this->init();
}

template<typename T>
void QtImage<T>::operator=(const QtImage& other) {
    m_expanding = new QSizePolicy();
    m_expanding->setHorizontalStretch(100);
    m_expanding->setVerticalStretch(100);

    m_last_size = other.last_size();
    m_original_width = other.original_width();
    m_original_height = other.original_height();
    trim = other.trim;

    m_image_qs = other.qstr();
    m_image_str = other.str();
    m_pix = QPixmap(other.qstr());

    m_handler = other.handler();
    m_handler->setMinimumSize(1, 1);
    m_pix.scaled(m_original_width, m_original_height);
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
T* QtImage<T>::handler() const {
    return m_handler;
}

template<typename T>
void QtImage<T>::update_pixel_count() {

    //this->Base_QtImage::calc_ratio(m_handler->width(), m_handler->height());

    m_handler->setPixmap(m_pix.scaled(m_handler->width()*0.97, m_handler->height()*0.97, Qt::KeepAspectRatio));
}

template<>
void QtImage<QPushButton>::update_pixel_count() {

    QSize out_size = QSize(m_handler->width()*0.97, m_handler->height()*0.97);

    m_handler->setText("");
    if (trim.layout == Trim::full_box) {
        m_handler->setIcon(QIcon(trim.crop(m_pix, out_size, m_original_width, m_original_height)));
    }
    else if (trim.layout == Trim::full_pic) {
        m_handler->setIcon(QIcon(m_pix));
    }

    m_handler->setIconSize(out_size);
}

template<typename T>
int QtImage<T>::height() const {
    return m_handler->height();
}

template<typename T>
int QtImage<T>::width() const {
    return m_handler->width();
}
