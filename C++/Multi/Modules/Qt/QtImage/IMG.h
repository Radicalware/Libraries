#pragma once

#include <string>

#include <QString>
#include <QPixmap>
#include <QIcon>

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

	IMG();
	IMG(std::string on_img, std::string off_img = "");
	IMG(QString on_img, QString off_img = "");

    void on(const std::string& on);
    void on(const QString& on);

    void off(const std::string& off);
    void off(const QString& off);

    void operator=(const IMG& other);
};
