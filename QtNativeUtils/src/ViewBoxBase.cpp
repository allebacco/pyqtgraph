#include "ViewBoxBase.h"

#include <QSizePolicy>

ViewBoxBase::ViewBoxBase(QGraphicsItem *parent, Qt::WindowFlags wFlags, const bool invertX, const bool invertY) :
    QGraphicsWidget2(parent, wFlags),
    mMatrixNeedsUpdate(true),
    mAutoRangeNeedsUpdate(true),
    mXInverted(invertX),
    mYInverted(invertY)
{
    setFlag(ItemClipsChildrenToShape, true);
    setFlag(ItemIsFocusable, true);

    setZValue(-100);
    setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));

    mViewRange.clear();
    mViewRange << Point(0.0, 1.0) << Point(0.0, 1.0);
}

void ViewBoxBase::invertY(const bool b)
{
    if(mYInverted==b)
        return;

    mYInverted = b;
    mMatrixNeedsUpdate = true; // updateViewRange won't detect this for us
    updateViewRange();

    emit sigStateChanged(this);
    emit sigYRangeChanged(this, mViewRange[1]);
}

void ViewBoxBase::invertX(const bool b)
{
    if(mXInverted==b)
        return;

    mXInverted = b;
    mMatrixNeedsUpdate = true; // updateViewRange won't detect this for us
    updateViewRange();

    emit sigStateChanged(this);
    emit sigYRangeChanged(this, mViewRange[0]);
}

void ViewBoxBase::setViewRange(const Point& x, const Point& y)
{
    mViewRange[0] = x;
    mViewRange[1] = y;
}
