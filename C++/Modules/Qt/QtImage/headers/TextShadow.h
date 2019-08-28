#pragma once

#include <QProxyStyle>
#include <QAbstractButton>
#include <QPalette>
#include <QPainter>
#include <QRect>
#include <QString>
#include <QColor>


// Class by "Uga Buga" as stated in the HowTo.txt
class TextShadow : public QProxyStyle
{
public:
    TextShadow();
    void drawItemText(QPainter *painter, const QRect &rect, int flags, const QPalette &pal, bool enabled, const QString &text, QPalette::ColorRole textRole /* = QPalette::NoRole */) const;
};
