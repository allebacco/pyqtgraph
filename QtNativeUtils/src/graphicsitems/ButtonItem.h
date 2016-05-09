#ifndef BUTTONITEM_H
#define BUTTONITEM_H

#include <QPixmap>


#include "../GraphicsObject.h"

class ButtonItem : public GraphicsObject
{
    Q_OBJECT
public:
    ButtonItem(QGraphicsItem* parentItem=nullptr);
    ButtonItem(const QString& imageFile, const double width=0.0, QGraphicsItem* parentItem=nullptr);
    ButtonItem(const QPixmap& pix, const double width=0.0, QGraphicsItem* parentItem=nullptr);
    virtual ~ButtonItem() {}

    void setImageFile(const QString& imageFile);

    void setPixmap(const QPixmap& pix);

    QPixmap pixmap() const { return mPixmap; }

    virtual QRectF boundingRect() const;

    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget=nullptr);

    void disable();
    void enable();

    virtual void mouseClickEvent(MouseClickEvent* event);
    virtual void hoverEvent(HoverEvent* event);

protected slots:

    void onEnableChanged();

signals:

    void clicked(GraphicsObject* obj);

protected:

    QPixmap mPixmap;
};

#endif // BUTTONITEM_H
