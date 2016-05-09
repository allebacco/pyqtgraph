#include "ButtonItem.h"

#include "../mouseevents/MouseClickEvent.h"
#include "../mouseevents/HoverEvent.h"


ButtonItem::ButtonItem(const QString& imageFile, const double width, QGraphicsItem* parentItem) :
    GraphicsObject(parentItem)
{
    connect(this, SIGNAL(enabledChanged()), this, SLOT(onEnableChanged()));
    setOpacity(0.7);

    setImageFile(imageFile);

    if(width!=0)
    {
        double s = width / mPixmap.width();
        scale(s, s);
    }
}

ButtonItem::ButtonItem(const QPixmap& pix, const double width, QGraphicsItem* parentItem) :
    GraphicsObject(parentItem)
{
    connect(this, SIGNAL(enabledChanged()), this, SLOT(onEnableChanged()));
    setOpacity(0.7);

    setPixmap(pix);

    if(width!=0)
    {
        double s = width / pix.width();
        scale(s, s);
    }
}

ButtonItem::ButtonItem(QGraphicsItem* parentItem) :
    GraphicsObject(parentItem)
{
    connect(this, SIGNAL(enabledChanged()), this, SLOT(onEnableChanged()));
    setOpacity(0.7);
}

void ButtonItem::setImageFile(const QString& imageFile)
{
    setPixmap(QPixmap(imageFile));
}

void ButtonItem::setPixmap(const QPixmap& pix)
{
    mPixmap = pix;
    update();
}

QRectF ButtonItem::boundingRect() const
{
    return QRectF(mPixmap.rect());
}

void ButtonItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    painter->setRenderHint(QPainter::Antialiasing);
    painter->drawPixmap(0, 0, mPixmap);
}

void ButtonItem::disable()
{
    setEnabled(false);
}

void ButtonItem::enable()
{
    setEnabled(true);
}

void ButtonItem::mouseClickEvent(MouseClickEvent *event)
{
    if(isEnabled()==false)
        return;

    event->accept();
    emit clicked(this);
}

void ButtonItem::hoverEvent(HoverEvent *event)
{
    if(isEnabled()==false)
        return;

    event->accept();
    if(event->isEnter())
        setOpacity(0.7);
    else
        setOpacity(0.4);
}

void ButtonItem::onEnableChanged()
{
    if(isEnabled())
        setOpacity(0.7);
    else
        setOpacity(0.4);
}
