#pragma once

#include <string>

#include <QString>
#include <QPixmap>
#include <QIcon>
#include <QImage>
#include <QPainter>
#include <QGraphicsScene>
#include <QEvent>

struct IMG
{
    std::string on_str;
    std::string off_str;

    QString on_qs;
    QString off_qs;

    QPixmap on_pix;
    QPixmap off_pix;

    QIcon on_icon;
    QIcon off_icon;

private:
	int m_width;
	int m_height;
	float on_pic_ratio;
	float off_pic_ratio;
	QEvent::Type* m_hover_status_ptr;

public:
	IMG();
	IMG(QEvent::Type* hover_status, std::string off_img, std::string on_img = "");
	IMG(QEvent::Type* hover_status, QString off_img, QString on_img = "");

    void set_on(const std::string& on);
    void set_on(const QString& on);

    void set_off(const std::string& off);
    void set_off(const QString& off);
	
    void operator=(const IMG& other);

	const int width() const;
	const int height() const;

	const std::string* get_img_ptr();
	const QEvent::Type* hover_status() const;
};
