#include "../headers/IMG.h"

#include <string>

#include <QString>
#include <QPixmap>
#include <QIcon>
#include <QPen>
#include <QFont>


IMG::IMG() {
}

IMG::IMG(QEvent::Type* hover_status, std::string off_img, std::string on_img) :
	m_hover_status_ptr(hover_status), off_str(off_img), on_str(on_img)
{
	if (!on_str.size())
		on_str = off_str;

	this->set_off(off_str);
	this->set_on(on_str);
}

IMG::IMG(QEvent::Type* hover_status, QString off_img, QString on_img) :
	m_hover_status_ptr(hover_status), off_str(off_img.toStdString()), on_str(on_img.toStdString())
{
	if (!on_str.size())
		on_str = off_str;

	this->set_off(off_str);
	this->set_on(on_str);
}

void IMG::set_on(const std::string& on) {
    on_str = on;
    on_qs = on_str.c_str();
    on_pix = QPixmap(on_qs);
    on_icon = QIcon(on_qs);
}
void IMG::set_on(const QString& on) {
    on_qs = on;
    on_str = on_qs.toStdString();
    on_pix = QPixmap(on_qs);
    on_icon = QIcon(on_qs);
}

void IMG::set_off(const std::string& off) {
    off_str = off;
    off_qs = off_str.c_str();
    off_pix = QPixmap(off_qs);
    off_icon = QIcon(off_qs);
}
void IMG::set_off(const QString& off) {
    off_qs = off;
    off_str = off_qs.toStdString();
    off_pix = QPixmap(off_qs);
    off_icon = QIcon(off_qs);
}

const int IMG::width() const
{
	return m_width;
}

const int IMG::height() const
{
	return m_height;
}

const std::string* IMG::get_img_ptr()
{
	return (*m_hover_status_ptr == QEvent::Type::Enter) ? &on_str : &off_str;
}

const QEvent::Type * IMG::hover_status() const
{
	return m_hover_status_ptr;
}

void IMG::operator=(const IMG& other) {
	this->~IMG();
    on_str = other.on_str; off_str = other.off_str;
    on_qs = other.on_qs; off_qs = other.off_qs;
    on_pix = other.on_pix; off_pix = other.off_pix;
    on_icon = other.on_icon; off_icon = other.off_icon;
}
