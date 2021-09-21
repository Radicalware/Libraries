
#include "QtImage_QLabel.h"


QtImage_QLabel::QtImage_QLabel() {};

QtImage_QLabel::~QtImage_QLabel() {
	if (m_init) {
		delete m_img;
		delete m_style;
	}
};


QtImage_QLabel::QtImage_QLabel(QLabel* t_handler, std::string off_image,
	std::string on_image, Style::Trim::Layout layout) :
	m_handler(t_handler)
{
	m_handler->installEventFilter(this);
	m_style = new Style(&m_handler->size());
	m_img = new IMG(this->hover_on_ptr(), off_image, on_image);
	m_init = true;
}


QtImage_QLabel::QtImage_QLabel(QLabel* t_handler, QString off_image,
	QString on_image, Style::Trim::Layout layout) :
	m_handler(t_handler)
{
	m_handler->installEventFilter(this);
	m_style = new Style(&m_handler->size());
	m_img = new IMG(&m_hover_on, off_image, on_image);
	m_init = true;
}


void QtImage_QLabel::operator=(const QtImage& other)
{
	if (m_init)
		this->~QtImage_QLabel();
	
	m_img = new IMG;
	m_style = new Style;
	m_init = true;
	m_img = other.img();
	m_style = other.style();
}

bool QtImage_QLabel::update_size(QObject* watched, QEvent* event)
{
	// if(m_style->trim->layout == Style::Trim::full_box) // You shouldn't need this turned on
	m_handler->setStyleSheet(m_style->retStyleSheet(m_img->get_img_ptr()).c_str());
	return true;
}

// ----------------- GETTERS ----------------------------------

void* QtImage_QLabel::handler() const
{
	return m_handler;
}

IMG* QtImage_QLabel::img() const
{
	return m_img;
}

Style* QtImage_QLabel::style() const
{
	return m_style;
}

Style::Trim* QtImage_QLabel::trim() const
{
	return m_style->trim;
}

Style::Font* QtImage_QLabel::font() const
{
	return m_style->font;
}

Style::Trim::Layout QtImage_QLabel::layout() const
{
	return m_style->trim->layout;
}


