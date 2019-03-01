#include "../headers/QtImage.h"

QtImage::QtImage() {};


QtImage::QtImage(QPushButton* t_push_button,
	std::string on_image, std::string off_image, Trim::Layout layout) :
    QtImage_T2<QPushButton>(t_push_button, on_image, off_image, layout)
{
	is_label = false;
	t_push_button->installEventFilter(this);
}

QtImage::QtImage(QPushButton* t_push_button,
	QString on_image, QString off_image, Trim::Layout layout) :
    QtImage_T2<QPushButton>(t_push_button, on_image, off_image, layout)
{
	is_label = false;
	t_push_button->installEventFilter(this);
}

QtImage::QtImage(QLabel* t_push_button, 
	std::string on_image, std::string off_image, Trim::Layout layout) :
	QtImage_T2<QLabel>(t_push_button, on_image, off_image, layout)
{
	is_label = true;
	t_push_button->installEventFilter(this);
}

QtImage::QtImage(QLabel* t_push_button, 
	QString on_image, QString off_image, Trim::Layout layout) :
	QtImage_T2<QLabel>(t_push_button, on_image, off_image, layout)
{
	is_label = true;
	t_push_button->installEventFilter(this);
}

bool QtImage::update_label(QObject* watched, QEvent* event)
{
	QLabel* label = qobject_cast<QLabel*>(watched);
	if (!label)
		return false;

	if (event->type() == QEvent::Enter) {
		this->QtImage_T2<QLabel>::update_pixel_count(QEvent::Enter);
		return true;
	}
	else if (event->type() == QEvent::Leave) {
		this->QtImage_T2<QLabel>::update_pixel_count(QEvent::Leave);
	}
	return true;
}

bool QtImage::update_button(QObject* watched, QEvent* event)
{
	QPushButton* button = qobject_cast<QPushButton*>(watched);
	if (!button)
		return false;

	if (event->type() == QEvent::Enter) {
		this->QtImage_T2<QPushButton>::update_pixel_count(QEvent::Enter);
		return true;
	}
	else if (event->type() == QEvent::Leave) {
		this->QtImage_T2<QPushButton>::update_pixel_count(QEvent::Leave);
		return true;
	}
}

// ------------------------------------------------------------------------------------
bool QtImage::eventFilter(QObject* watched, QEvent* event)
{
	if (is_label) {
		this->update_label(watched, event);
	}
	else {
		this->update_button(watched, event);
	}
	return false;
}


void QtImage::update() {
	if (is_label) {
		if (_hover_on) {
			this->QtImage_T2<QLabel>::update_pixel_count(QEvent::Enter);
		}
		else {
			this->QtImage_T2<QLabel>::update_pixel_count(QEvent::Leave);
		}
	}
	else {
		if (_hover_on) {
			this->QtImage_T2<QPushButton>::update_pixel_count(QEvent::Enter);
		}
		else {
			this->QtImage_T2<QPushButton>::update_pixel_count(QEvent::Leave);
		}
	}
}
