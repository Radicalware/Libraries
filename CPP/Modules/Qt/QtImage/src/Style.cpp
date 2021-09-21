#include "Style.h"



Style::Trim::Trim()
{}

Style::Trim::Trim(int** current_size, QPixmap* off_img_ptr, QPixmap* on_img_ptr, QEvent::Type* hover_status_ptr):
	m_current_size(*current_size), m_off_img_ptr(off_img_ptr), m_on_img_ptr(on_img_ptr), m_hover_status_ptr(hover_status_ptr)
{
}

// ---------------------------------------------------------------------------
void Style::Trim::image_qss(QSize* current_size, const std::string* img_ptr, std::string& ret) {
	if (layout == full_pic)
		ret += standard_image;
	else if (layout == full_box)
		ret += bordered_image;
	else
		ret += standard_image;

	// use the following to remove pixels 
	// background-image: top, right, bottom, and left
	// or
	// top-left, top-center, top-right, 
	// center-left, center-center, center-right, 
	// bottom-left, bottom-center, and bottom-right
	ret += *img_ptr+')';
	if (m_current_size[1] == 0)
		return;
	if (layout == Layout::full_box) {
		float box_ratio = static_cast<float>(m_current_size[0]) / 
			static_cast<float>(m_current_size[1]);
		float pic_ratio;
		int pic_width = 0;
		int pic_height = 0;
		if (*m_hover_status_ptr == QEvent::Type::Enter) {
			pic_width = m_on_img_ptr->width();
			pic_height = m_on_img_ptr->height();
		}
		else {
			pic_width = m_off_img_ptr->width();
			pic_height = m_off_img_ptr->height();
		}
		pic_ratio = static_cast<float>(pic_width) / static_cast<float>(pic_height);

		int crop_size = 0;
		if (box_ratio > pic_ratio) { // box ratio has longer width (empty sides)
			// remove empty sides by trimming top/bottom
			crop_size = (pic_height - (m_current_size[1] * pic_width / m_current_size[0])) / 2;
			ret += std::to_string(crop_size) + " 0 " + std::to_string(crop_size) + " 0";
		}
		else if(box_ratio < pic_ratio) { // box ratio has longer height (empty top/bottom)
			crop_size = (pic_width - (m_current_size[0] * pic_height / m_current_size[1])) / 2;
			ret += " 0 " + std::to_string(crop_size) + " 0 " + std::to_string(crop_size);
		}
		else { // both are equal 
			return;
		}
	}
	ret += ';';
	// blank horizontal = x 0 x 0
	// blank vertical   = 0 x 0 x
	// identify the number of blank pixils left/right
	// and remove that SAME number from top/bottom
}

// ---------------------------------------------------------------------------

std::string Style::Font::qss()
{
	std::string ret;
	ret += other;
	ret += ";font-family: \"" + family+'"';
	ret += ";font-weight: " + weight;
	ret += ";font-size: " + std::to_string(text_size) + 'p' + 'x';;
	ret += ";font-style: " + style;
	ret += ";color: " + color;
	ret += ";text-align: " + align;
	ret += ';';

	return ret;
}
// ==============================================================================================================
Style::~Style()
{
	delete trim;
	delete font;
}

Style::Style(){
	font = new Font;
}

Style::Style(int** i_current_size, Trim::Layout i_layout, IMG* image, QEvent::Type* hover_status)
{
	m_current_size = *i_current_size;
	font = new Font;
	trim = new Trim(&m_current_size, &image->off_pix, &image->on_pix, hover_status);
	trim->layout = i_layout;
}

void Style::operator=(const Style& other) 
{
	// TODO FINISH THIS
	trim->layout = other.trim->layout;
	trim->crop_pixils = other.trim->crop_pixils;
}

std::string Style::retStyleSheet(QSize* current_size, const std::string* img_ptr)
{
	std::string ret; // I decided not to return the string to avoid calling the copy constructor
	trim->image_qss(current_size, img_ptr, ret);
	ret += m_saved_styleSheet;
	return ret;
}

void Style::saveStyleSheet()
{
	m_saved_styleSheet = font->qss();
}


