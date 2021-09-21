#include "QtImage.h"

QtImage::QtImage() {}
QtImage::~QtImage(){}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
bool QtImage::eventFilter(QObject* watched, QEvent* event)
{
	m_hover_status = event->type();
	if (m_hover_status == QEvent::Type::Enter || m_hover_status == QEvent::Type::Leave ||
		m_hover_status == QEvent::Type::Resize) {
		if (m_hover_status != m_hover_status_tmp) {
			m_hover_status_tmp = m_hover_status;
			this->update_size();
		}
	}
	return QObject::eventFilter(watched, event);
}

