#include "QtImage.h"

QtImage::QtImage() {};

QtImage::QtImage(QObject* parent, QPushButton* t_push_button,
	std::string on_image, std::string off_image, Trim::Layout layout) :
	QObject(parent), QtImage_T2<QPushButton>(parent, t_push_button, on_image, off_image, layout)
{
	t_push_button->installEventFilter(this);
}

QtImage::~QtImage()
{
}

QtImage::QtImage(QObject* parent, QPushButton* t_push_button,
	QString on_image, QString off_image, Trim::Layout layout) :
	QObject(parent), QtImage_T2<QPushButton>(parent, t_push_button, on_image, off_image, layout)
{
	t_push_button->installEventFilter(this);
}

bool QtImage::eventFilter(QObject* watched, QEvent* event)
{
	QPushButton* button = qobject_cast<QPushButton*>(watched);
	if (!button) {
		return false;
	}
	if (event->type() == QEvent::Enter) {
		this->update_pixel_count(QEvent::Enter);
		return true;
	}
	else if (event->type() == QEvent::Leave) {
		this->update_pixel_count(QEvent::Leave);
		return true;
	}
	return false;
}
