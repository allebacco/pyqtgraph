%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class ButtonItem : public GraphicsObject
{

%TypeHeaderCode
#include "ButtonItem.h"
%End

public:
    ButtonItem(QGraphicsItem* parentItem=nullptr) /KeywordArgs="All" /;
    ButtonItem(const QString& imageFile, const double width=0.0, QGraphicsItem* parentItem=nullptr) /KeywordArgs="All" /;
    ButtonItem(const QPixmap& pix, const double width=0.0, QGraphicsItem* parentItem=nullptr) /KeywordArgs="All" /;
    virtual ~ButtonItem();

    void setImageFile(const QString& imageFile);

    void setPixmap(const QPixmap& pix);

    QPixmap pixmap() const;

    virtual QRectF boundingRect() const;

    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget=nullptr);

    void disable();
    void enable();

    virtual void mouseClickEvent(MouseClickEvent* event);
    virtual void hoverEvent(HoverEvent* event);

signals:

    void clicked(GraphicsObject* obj);
};
