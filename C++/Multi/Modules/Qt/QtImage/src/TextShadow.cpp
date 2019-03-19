#include "TextShadow.h"


// Class by "Uga Buga" as stated in the HowTo.txt
TextShadow::TextShadow()
{}

void TextShadow::drawItemText(QPainter *painter, const QRect &rect, int flags, const QPalette &pal, bool enabled, const QString &text, QPalette::ColorRole textRole /* = QPalette::NoRole */) const
{
    if (textRole == QPalette::ButtonText && dynamic_cast<QAbstractButton*>(painter->device())){
        QPalette palShadow(pal);
        palShadow.setColor(QPalette::ButtonText, QColor(255, 63, 63, 150));
        QProxyStyle::drawItemText(painter, rect.adjusted(2, 2, 2, 2), flags, palShadow, enabled, text, textRole);
		// TODO: update to (5, 5, 5, 5) for 4k
    }
    QProxyStyle::drawItemText(painter, rect, flags, pal, enabled, text, textRole);
}
