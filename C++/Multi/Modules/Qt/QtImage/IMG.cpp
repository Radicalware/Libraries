#include "IMG.h"

#include <string>

#include <QString>
#include <QPixmap>
#include <QIcon>

IMG::IMG() {}

IMG::IMG(std::string on_img, std::string off_img) : 
	on_str(on_img), off_str(off_img) 
{
	this->on(on_img);
	if (off_img.size()) {
		this->off(off_img);
	}
	else {
		this->off(on_img);
	}
}

IMG::IMG(QString on_img, QString off_img)
{
	this->on(on_img);
	if (off_img.size()) {
		this->off(off_img);
	}
	else {
		this->off(on_img);
	}
}

void IMG::on(const std::string& on) {
    on_str = on;
    on_qs = on_str.c_str();
    on_pix = QPixmap(on_qs);
    on_icon = QIcon(on_qs);
}
void IMG::on(const QString& on) {
    on_qs = on;
    on_str = on_qs.toStdString();
    on_pix = QPixmap(on_qs);
    on_icon = QIcon(on_qs);
}

void IMG::off(const std::string& off) {
    off_str = off;
    off_qs = off_str.c_str();
    off_pix = QPixmap(off_qs);
    off_icon = QIcon(off_qs);
}
void IMG::off(const QString& off) {
    off_qs = off;
    off_str = off_qs.toStdString();
    off_pix = QPixmap(off_qs);
    off_icon = QIcon(off_qs);
}

void IMG::operator=(const IMG& other) {
    on_str = other.on_str; off_str = other.off_str;
    on_qs = other.on_qs; off_qs = other.off_qs;
    on_pix = other.on_pix; off_pix = other.off_pix;
    on_icon = other.on_icon; off_icon = other.off_icon;
}
