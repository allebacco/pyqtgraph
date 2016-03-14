#include "GraphicsObject.h"

#include "QGraphicsWidget2.h"
#include "ViewBoxBase.h"
#include "GraphicsViewBase.h"


GraphicsObject::GraphicsObject(QGraphicsItem *parent) :
    QGraphicsObject(parent),
    ExtendedItem(this)
{
    setFlag(ItemSendsGeometryChanges, true);
}

GraphicsObject::~GraphicsObject()
{

}

QTransform GraphicsObject::sceneTransform() const
{
    if(scene()==nullptr)
        return transform();
    return QGraphicsObject::sceneTransform();
}

void GraphicsObject::viewRangeChanged(const QList<Point> &range)
{
    // Called whenever the view coordinates of the ViewBox containing this item have changed.
}

void GraphicsObject::viewTransformChanged()
{
    // Called whenever the transformation matrix of the view has changed.
    // (eg, the view range has changed or the view was resized)
}

QVariant GraphicsObject::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
    QVariant ret = QGraphicsObject::itemChange(change, value);

    if(change==ItemParentHasChanged || change==ItemSceneHasChanged)
        parentIsChanged();
    else if(change==ItemPositionHasChanged || change==ItemTransformHasChanged)
        informViewBoundsChanged();

    return ret;
}


QTransform GraphicsObject::deviceTransform() const
{
    GraphicsViewBase* view = getViewWidget();
    if(view==nullptr)
        return QTransform();
    return QGraphicsObject::deviceTransform(view->viewportTransform());
}


void GraphicsObject::setParentItem(QGraphicsItem* newParent)
{
    // Workaround for Qt bug: https://bugreports.qt-project.org/browse/QTBUG-18616
    if(newParent!=nullptr)
    {
        QGraphicsScene* pscene = newParent->scene();
        if(pscene!=nullptr && pscene!=scene())
            pscene->addItem(this);
    }
    QGraphicsObject::setParentItem(newParent);
}

void GraphicsObject::disconnectView(ViewBoxBase* view)
{
    QObject::disconnect(view, SIGNAL(sigRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::disconnect(view, SIGNAL(sigTransformChanged()), this, SLOT(viewTransformChanged()));
}

void GraphicsObject::disconnectView(GraphicsViewBase* view)
{
    QObject::disconnect(view, SIGNAL(sigDeviceRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::disconnect(view, SIGNAL(sigDeviceTransformChanged()), this, SLOT(viewTransformChanged()));
}

void GraphicsObject::connectView(ViewBoxBase* view)
{
    QObject::connect(view, SIGNAL(sigRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::connect(view, SIGNAL(sigTransformChanged()), this, SLOT(viewTransformChanged()));
}

void GraphicsObject::connectView(GraphicsViewBase* view)
{
    QObject::connect(view, SIGNAL(sigDeviceRangeChanged(QList<Point>)), this, SLOT(viewRangeChanged(QList<Point>)));
    QObject::connect(view, SIGNAL(sigDeviceTransformChanged()), this, SLOT(viewTransformChanged()));
}

