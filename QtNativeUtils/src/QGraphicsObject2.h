#ifndef BASEGRAPHICSITEM2_H
#define BASEGRAPHICSITEM2_H

#include <QGraphicsView>
#include <QGraphicsObject>

#include "Point.h"

class QGraphicsObject2: public QGraphicsObject
{
public:
    QGraphicsObject2(QGraphicsItem* parent=nullptr);
    ~QGraphicsObject2() {}

    QGraphicsView* getViewWidget() const
    {
        QGraphicsScene* s = scene();
        if(s==nullptr)
            return nullptr;
        QList<QGraphicsView*> views = s->views();
        if(views.size()>0)
            return views[0];
        return nullptr;
    }

    void forgetViewWidget()
    {}

    QTransform deviceTransform() const
    {
        QGraphicsView* view = getViewWidget();
        if(view==nullptr)
            return QTransform();
        return QGraphicsObject::deviceTransform(view->viewportTransform());
    }

    QTransform deviceTransform(const QTransform& viewportTransform) const
    {
        return QGraphicsObject::deviceTransform(viewportTransform);
    }

    QList<QGraphicsItem*> getBoundingParents() const;

    QVector<Point> pixelVectors() const
    {
        return pixelVectors(QPointF(1.0, 0.0));
    }

    QVector<Point> pixelVectors(const QPointF& direction) const;

    double pixelLength(const QPointF& direction, const bool ortho=false) const
    {
        QVector<Point> p = pixelVectors(direction);
        if(ortho)
            return p[1].length();
        return p[0].length();
    }


    QPointF	mapFromDevice(const QPointF& point) const { return deviceTransform().inverted().map(point); }
    QPointF	mapFromDevice(const QPoint& point) const { return deviceTransform().inverted().map(QPointF(point)); }
    QPolygonF mapFromDevice(const QRectF& rect) const { return deviceTransform().inverted().map(rect); }
    QPolygonF mapFromDevice(const QPolygonF& polygon) const { return deviceTransform().inverted().map(polygon); }
    QPainterPath mapFromDevice(const QPainterPath& path) const { return deviceTransform().inverted().map(path); }
    QPointF	mapFromDevice(qreal x, qreal y) const { return mapFromDevice(QPointF(x, y)); }

    QPointF	mapToDevice(const QPointF& point) const { return deviceTransform().map(point); }
    QPointF	mapToDevice(const QPoint& point) const { return deviceTransform().map(QPointF(point)); }
    QPolygonF mapToDevice(const QRectF& rect) const { return deviceTransform().map(rect); }
    QPolygonF mapToDevice(const QPolygonF& polygon) const { return deviceTransform().map(polygon); }
    QPainterPath mapToDevice(const QPainterPath& path) const { return deviceTransform().map(path); }
    QPointF	mapToDevice(qreal x, qreal y) const { return mapToDevice(QPointF(x, y)); }

    QRectF mapRectToDevice(const QRectF& rect) const { return deviceTransform().mapRect(rect); }
    QRect mapRectToDevice(const QRect& rect) const { return deviceTransform().mapRect(rect); }

    QRectF mapRectFromDevice(const QRectF& rect) const { return deviceTransform().inverted().mapRect(rect); }
    QRect mapRectFromDevice(const QRect& rect) const { return deviceTransform().inverted().mapRect(rect); }

    double transformAngle(QGraphicsItem* relativeItem=nullptr) const;
};

#endif // BASEGRAPHICSITEM2_H



/*
    def mapToView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        return vt.map(obj)

    def mapRectToView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        return vt.mapRect(obj)

    def mapFromView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = fn.invertQTransform(vt)
        return vt.map(obj)

    def mapRectFromView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = fn.invertQTransform(vt)
        return vt.mapRect(obj)

    def pos(self):
        return Point(self._qtBaseClass.pos(self))

    def viewPos(self):
        return self.mapToView(self.mapFromParent(self.pos()))

    def parentItem(self):
        ## PyQt bug -- some items are returned incorrectly.
        return GraphicsScene.translateGraphicsItem(self._qtBaseClass.parentItem(self))

    def setParentItem(self, parent):
        ## Workaround for Qt bug: https://bugreports.qt-project.org/browse/QTBUG-18616
        if parent is not None:
            pscene = parent.scene()
            if pscene is not None and self.scene() is not pscene:
                pscene.addItem(self)
        return self._qtBaseClass.setParentItem(self, parent)

    def childItems(self):
        ## PyQt bug -- some child items are returned incorrectly.
        return list(map(GraphicsScene.translateGraphicsItem, self._qtBaseClass.childItems(self)))


    def transformAngle(self, relativeItem=None):
        """Return the rotation produced by this item's transform (this assumes there is no shear in the transform)
        If relativeItem is given, then the angle is determined relative to that item.
        """
        if relativeItem is None:
            relativeItem = self.parentItem()


        tr = self.itemTransform(relativeItem)
        if isinstance(tr, tuple):  ## difference between pyside and pyqt
            tr = tr[0]
        #vec = tr.map(Point(1,0)) - tr.map(Point(0,0))
        vec = tr.map(QtCore.QLineF(0,0,1,0))
        #return Point(vec).angle(Point(1,0))
        return vec.angleTo(QtCore.QLineF(vec.p1(), vec.p1()+QtCore.QPointF(1,0)))
*/


