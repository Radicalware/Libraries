
#include "../headers/QtImage_T1.h"

// This is a templated class and therefore 
// is fully contained in the header file.


QSize QtImage_T1::last_size() const {
	return m_last_size;
}

int QtImage_T1::original_width() const {
	return _original.width;
}

int QtImage_T1::original_height() const {
	return _original.height;
}

void QtImage_T1::set_layout(Trim::Layout new_layout) {
	trim.layout = new_layout;
}