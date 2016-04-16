#include "ViewBoxBase.h"

#include <QSizePolicy>
#include <QDebug>

#include "QGraphicsScene2.h"
#include "graphicsitems/ChildGroup.h"

ViewBoxBase::ViewBoxBase(QGraphicsItem *parent, Qt::WindowFlags wFlags, const QPen& border, const double lockAspect, const bool invertX, const bool invertY, const bool enableMouse) :
    GraphicsWidget(parent, wFlags),
    mMatrixNeedsUpdate(true),
    mAutoRangeNeedsUpdate(true),
    mXInverted(invertX),
    mYInverted(invertY),
    mMouseMode(PanMode),
    mBorder(border),
    mWheelScaleFactor(-1.0/8.0),
    mLinksBlocked(false)
{
    Range::registerMetatype();

    setFlag(ItemClipsChildrenToShape);
    setFlag(ItemIsFocusable, true);

    setZValue(-100);
    setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));

    mViewRange.clear();
    mViewRange << Range(0.0, 1.0) << Range(0.0, 1.0);

    mTargetRange.clear();
    mTargetRange << Range(0.0, 1.0) << Range(0.0, 1.0);

    mAutoRangeEnabled.clear();
    mAutoRangeEnabled << true << true;

    mAutoPan.clear();
    mAutoPan << false << false;

    mAutoVisibleOnly.clear();
    mAutoVisibleOnly << false << false;

    mMouseEnabled << enableMouse << enableMouse;

    mChildGroup = new ChildGroup(this);
    mChildGroup->addListener(this);
    setInnerSceneItem(mChildGroup);

    mBackground = new QGraphicsRectItem(rect());
    mBackground->setParentItem(this);
    mBackground->setZValue(-1e6);
    mBackground->setPen(QPen(Qt::NoPen));
    mBackground->setBrush(QBrush(QColor(Qt::transparent)));
    updateBackground();

    mRbScaleBox = new QGraphicsRectItem(0.0, 0.0, 1.0, 1.0);
    mRbScaleBox->setPen(QPen(QBrush(QColor(255,255,100)), 0.0));
    mRbScaleBox->setBrush(QBrush(QColor(255,255,0,100)));
    mRbScaleBox->setZValue(1e9);
    mRbScaleBox->hide();
    addItem(mRbScaleBox, true);

    mAxHistory.clear();
    mAxHistoryPointer = -1;

    setAspectLocked(lockAspect!=0.0, lockAspect);

    mLinkedViews.clear();
    mLinkedViews << QWeakPointer<ViewBoxBase>() << QWeakPointer<ViewBoxBase>();
}

void ViewBoxBase::updateMatrix()
{
    QRectF vr = viewRect();
    if(vr.height()==0.0 || vr.width()==0.0)
        return;

    QRectF bounds = rect();

    QPointF scale(bounds.width()/vr.width(), bounds.height()/vr.height());
    if(!mYInverted)
        scale.ry() *= -1.0;
    if(mXInverted)
        scale.rx() *= -1.0;

    QTransform m;

    // First center the viewport at 0
    QPointF center = bounds.center();
    m.translate(center.x(), center.y());

    // Now scale and translate properly
    m.scale(scale.x(), scale.y());
    center = vr.center();
    m.translate(-center.x(), -center.y());

    mChildGroup->setTransform(m);

    emit sigTransformChanged();
    mMatrixNeedsUpdate = false;
}


void ViewBoxBase::itemBoundsChanged(QGraphicsItem *item)
{
    if(mAutoRangeEnabled[0] || mAutoRangeEnabled[1])
    {
        mAutoRangeNeedsUpdate = true;
        update();
    }
}

void ViewBoxBase::invertY(const bool b)
{
    if(mYInverted==b)
        return;

    mYInverted = b;
    mMatrixNeedsUpdate = true; // updateViewRange won't detect this for us
    updateViewRange();

    emit sigStateChanged(this);
    emit sigYRangeChanged(mViewRange[1]);
}

void ViewBoxBase::invertX(const bool b)
{
    if(mXInverted==b)
        return;

    mXInverted = b;
    mMatrixNeedsUpdate = true; // updateViewRange won't detect this for us
    updateViewRange();

    emit sigStateChanged(this);
    emit sigYRangeChanged(mViewRange[0]);
}

void ViewBoxBase::setBackgroundColor(const QColor &color)
{
    mBackground->setBrush(QBrush(color));
    updateBackground();
}

QColor ViewBoxBase::backgroundColor() const
{
    return mBackground->brush().color();
}

void ViewBoxBase::updateBackground()
{
    if(mBackground->brush().color().alpha()==0)
        mBackground->hide();
    else
    {
        mBackground->show();
        mBackground->setRect(rect());
    }
}

void ViewBoxBase::setAutoPan(const bool x, const bool y)
{
    mAutoPan[0] = x;
    mAutoPan[1] = y;

    updateAutoRange();
}

void ViewBoxBase::setAutoVisible(const bool x, const bool y)
{
    mAutoVisibleOnly[0] = x;
    if(x)
        mAutoVisibleOnly[1] = false;
    mAutoVisibleOnly[1] = y;
    if(y)
        mAutoVisibleOnly[0] = false;

    updateAutoRange();
}

void ViewBoxBase::setAspectLocked(const bool lock, const double ratio)
{
    //mAspectLocked = lock ? ratio: 0.0;

    if(lock==false)
    {
        if(mAspectLocked==0.0)
            return;
        else
            mAspectLocked = 0.0;
    }
    else
    {
        double newRatio = ratio;
        QRectF r = rect();
        QRectF vr = viewRect();
        double currentRatio = 1.0;
        if(r.height()!=0.0 && vr.width()!=0.0 && vr.height()!=0.0)
            currentRatio = (r.width()/r.height()) / (vr.width()/vr.height());
        if(newRatio==0.0)
            newRatio = currentRatio;
        if(mAspectLocked==newRatio) // nothing to change
            return;
        mAspectLocked = newRatio;
    }

    updateAutoRange();
    updateViewRange();
    emit sigStateChanged(this);

/*
    if not lock:
        if self.aspectLocked() == 0.0:
            return
        ViewBoxBase.setAspectLocked(self, False, 0.0)
    else:
        rect = self.rect()
        vr = self.viewRect()
        if rect.height() == 0 or vr.width() == 0 or vr.height() == 0:
            currentRatio = 1.0
        else:
            currentRatio = (rect.width()/float(rect.height())) / (vr.width()/vr.height())
        if ratio is None:
            ratio = currentRatio
        if self.aspectLocked() == ratio: # nothing to change
            return
        ViewBoxBase.setAspectLocked(self, True, ratio)
        #if ratio != currentRatio:  ## If this would change the current range, do that now
        #    self.updateViewRange()

    self.updateAutoRange()
    self.updateViewRange()
    self.sigStateChanged.emit(self)
*/

}

void ViewBoxBase::setMouseEnabled(const bool enabledOnX, const bool enabledOnY)
{
    mMouseEnabled[0] = enabledOnX;
    mMouseEnabled[1] = enabledOnY;

    emit sigStateChanged(this);
}

QRectF ViewBoxBase::viewRect() const
{
    const Range& p1 = mViewRange[0];
    const Range& p2 = mViewRange[1];
    QRectF r(p1.min(), p2.min(), p1.max()-p1.min(), p2.max()-p2.min());
    return r;
}

QRectF ViewBoxBase::targetRect() const
{
    const Range& p1 = mTargetRange[0];
    const Range& p2 = mTargetRange[1];
    return QRectF(p1.min(), p2.min(), p1.max()-p1.min(), p2.max()-p2.min());
}

GraphicsObject* ViewBoxBase::innerSceneItem() const
{
    return mInnerSceneItem;
}

void ViewBoxBase::itemsChanged()
{
    updateAutoRange();
}

ChildGroup *ViewBoxBase::getChildGroup() const
{
    return mChildGroup;
}

QTransform ViewBoxBase::childTransform() const
{
    // Return the transform that maps from child(item in the childGroup) coordinates to local coordinates.
    // (This maps from inside the viewbox to outside)

    if(mMatrixNeedsUpdate)
        const_cast<ViewBoxBase*>(this)->updateMatrix();

    return mChildGroup->transform();
}

const QList<QGraphicsItem *> &ViewBoxBase::addedItems() const
{
    return mAddedItems;
}

void ViewBoxBase::addItem(QGraphicsItem *item, const bool ignoreBounds)
{
    // Add a QGraphicsItem to this view. The view will include this item when determining how to set its range
    // automatically unless *ignoreBounds* is True.

    if(item->zValue() < zValue())
        item->setZValue(zValue()+1.0);

    QGraphicsScene* gScene = scene();
    if(gScene!=nullptr && gScene!=item->scene())
        gScene->addItem(item);
    item->setParentItem(mChildGroup);
    if(!ignoreBounds)
        mAddedItems.append(item);
    updateAutoRange();
}

void ViewBoxBase::removeItem(QGraphicsItem* item)
{
    // Remove an item from this view.
    mAddedItems.removeOne(item);
    scene()->removeItem(item);
    updateAutoRange();
}

void ViewBoxBase::clear()
{
    int count = mAddedItems.size();
    for(int i=0; i<count; ++i)
        scene()->removeItem(mAddedItems[i]);

    QList<QGraphicsItem*> cItems = mChildGroup->childItems();
    count = cItems.size();
    for(int i=0; i<count; ++i)
        cItems[i]->setParentItem(nullptr);

    mAddedItems.clear();
}

double ViewBoxBase::suggestPadding(const ViewBoxBase::Axis ax) const
{
    const double l = ax==Axis::XAxis ? width() : height();
    if(l>0.0)
        return std::min(std::max(1.0/std::sqrt(l), 0.02), 0.1);
    return 0.02;
}

void ViewBoxBase::enableAutoRange(const ViewBoxBase::Axis axis, const bool enable)
{
    // Enable (or disable) auto-range for *axis*, which may be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes for both
    // (if *axis* is omitted, both axes will be changed).
    // When enabled, the axis will automatically rescale when items are added/removed or change their shape.
    // The argument *enable* may optionally be a float (0.0-1.0) which indicates the fraction of the data that should
    // be visible (this only works with items implementing a dataRange method, such as PlotDataItem).


    if(axis==XYAxes || axis==XAxis)
    {
        if(mAutoRangeEnabled[0] != enable)
        {
            // If we are disabling, do one last auto-range to make sure that
            // previously scheduled auto-range changes are enacted
            if(!enable && mAutoRangeNeedsUpdate)
                updateAutoRange();

           mAutoRangeEnabled[0] = enable;
           mAutoRangeNeedsUpdate |= enable;
           update();
        }
    }

    if(axis==XYAxes || axis==YAxis)
    {
        if(mAutoRangeEnabled[1] != enable)
        {
            // If we are disabling, do one last auto-range to make sure that
            // previously scheduled auto-range changes are enacted
            if(!enable && mAutoRangeNeedsUpdate)
                updateAutoRange();

           mAutoRangeEnabled[1] = enable;
           mAutoRangeNeedsUpdate |= enable;
           update();
        }
    }

    if(mAutoRangeNeedsUpdate)
        updateAutoRange();

    emit sigStateChanged(this);
}

void ViewBoxBase::enableAutoRange(const QString& axis, const bool enable)
{
    if(axis=="xy")
        enableAutoRange(XYAxes, enable);
    else if(axis=="x")
        enableAutoRange(XAxis, enable);
    else if(axis=="y")
        enableAutoRange(YAxis, enable);
}

QRectF ViewBoxBase::itemBoundingRect(const QGraphicsItem *item) const
{
    // Return the bounding rect of the item in view coordinates
    return mapSceneToView(item->sceneBoundingRect()).boundingRect();
}

void ViewBoxBase::setRange(const Range& xRange, const Range& yRange, const double padding, const bool disableAutoRange)
{
    const Range range[2] {xRange, yRange};
    const bool changes[2] {xRange.isValid(), yRange.isValid()};

    // Update axes one at a time
    bool changed[2] {false, false};
    for(int i=0; i<2; ++i)
    {
        if(!range[i].isValid())
            continue;

        double mn = range[i].min();
        double mx = range[i].max();
        double xpad = padding;

        // If we requested 0 range, try to preserve previous scale.
        // Otherwise just pick an arbitrary scale.
        if(mn==mx)
        {
            double dy = mViewRange[i].max() - mViewRange[i].min();
            dy = (dy==0.0) ? 1.0 : dy;
            mn -= dy*0.5;
            mx += dy*0.5;
            xpad = 0.0;
        }

        if(!std::isfinite(xpad))
            xpad = suggestPadding(i==0 ? XAxis : YAxis);

        double p = (mx-mn) * xpad;
        mn -= p;
        mx += p;

        // Set target range
        Range tr(mn, mx);
        if(!tr.finiteEqual(mTargetRange[i]))
        {
            mTargetRange[i] = tr;
            changed[i] = true;
        }
    }

    // Update viewRange to match targetRange as closely as possible while
    // accounting for aspect ratio constraint
    if(changes[0] && changes[1])
        updateViewRange(false, false);
    else
        updateViewRange(changes[0], changes[1]);

    if(disableAutoRange)
    {
        if(changed[0])
            enableAutoRange(XAxis, false);
        if(changed[1])
            enableAutoRange(YAxis, false);
    }

    if(changed[0] || changed[1])
        emit sigStateChanged(this);

    if(changed[0] && mAutoVisibleOnly[0] && mAutoRangeEnabled[0])
        setAutoRangeNeedsUpdate(true);
    else if(changed[1] && mAutoVisibleOnly[1] && mAutoRangeEnabled[1])
        setAutoRangeNeedsUpdate(true);
}

void ViewBoxBase::setRange(const QRectF& rect, const double padding, const bool disableAutoRange)
{
    Range xRange;
    if(rect.width()>0)
        xRange.setRange(rect.left(), rect.right());
    Range yRange;
    if(rect.height()>0)
        yRange.setRange(rect.top(), rect.bottom());

    setRange(xRange, yRange, padding, disableAutoRange);
}

void ViewBoxBase::setXRange(const double minR, const double maxR, const double padding)
{
    setRange(Range(minR, maxR), Range(), padding);
}

void ViewBoxBase::setYRange(const double minR, const double maxR, const double padding)
{
    setRange(Range(), Range(minR, maxR), padding);
}

void ViewBoxBase::scaleBy(const QPointF& s, const QPointF& center)
{
    QPointF scale(s);

    if(mAspectLocked != 0.0)
        scale.setX(scale.y());

    QRectF vr = targetRect();

    QPointF tl = (vr.topLeft()-center);
    tl.rx() *= scale.x();
    tl.ry() *= scale.y();
    tl += center;

    QPointF br = (vr.bottomRight()-center);
    br.rx() *= scale.x();
    br.ry() *= scale.y();
    br += center;

    //QPointF tl = center + (vr.topLeft()-center) * scale;
    //QPointF br = center + (vr.bottomRight()-center) * scale;

    setRange(QRectF(tl, br), 0.0);
}

void ViewBoxBase::scaleBy(const QPointF& s)
{
    QPointF center = targetRect().center();
    scaleBy(s, center);
}

void ViewBoxBase::translateBy(const QPointF& t)
{
    setRange(targetRect().translated(t), 0.0);
}

void ViewBoxBase::setMouseMode(const ViewBoxBase::MouseMode mode)
{
    mMouseMode = mode;

    emit sigStateChanged(this);
}

void ViewBoxBase::prepareForPaint()
{
    // don't check whether auto range is enabled here--only check when setting dirty flag.
    if(mAutoRangeNeedsUpdate) // and autoRangeEnabled
        updateAutoRange();
    if(mMatrixNeedsUpdate)
        updateMatrix();
}

void ViewBoxBase::showAxRect(const QRectF& axRect)
{
    setRange(axRect.normalized(), 0.0);
    emit sigRangeChangedManually(mMouseEnabled[0], mMouseEnabled[1]);
}

void ViewBoxBase::scaleHistory(const int d)
{
    if(mAxHistory.isEmpty())
        return;

    int ptr = std::max(0, std::min(mAxHistory.size()-1, mAxHistoryPointer+d));
    if(ptr!=mAxHistoryPointer)
    {
        mAxHistoryPointer = ptr;
        showAxRect(mAxHistory[ptr]);
    }
}

void ViewBoxBase::linkedViewChanged(ViewBoxBase* view, const ViewBoxBase::Axis axis)
{
    if(mLinksBlocked || view==nullptr)
        return;

    QRectF vg = view->screenGeometry();
    QRectF sg = screenGeometry();
    if(vg.isNull() || sg.isNull())
        return;

    QRectF vr = view->viewRect();
    view->blockLink(true);
    if(axis==XAxis)
    {
        double overlap = std::min(sg.right(), vg.right() - std::max(sg.left(), vg.left()));
        double x1, x2;
        if(overlap < std::min(vg.width()/3.0, sg.width()/3.0))
        {
            // if less than 1/3 of views overlap, then just replicate the view
            x1 = vr.left();
            x2 = vr.right();
        }
        else
        {
            // views overlap; line them up
            double upp = vr.width() / vg.width();
            if(xInverted())
                x1 = vr.left() + (sg.right()-vg.right()) * upp;
            else
                x1 = vr.left() + (sg.x()-vg.x()) * upp;
            x2 = x1 + sg.width() * upp;
        }
        enableAutoRange(XAxis, false);
        setXRange(x1, x2, 0.0);
    }
    else //if(axis==YAxis)
    {
        double overlap = std::min(sg.bottom(), vg.bottom()) - std::max(sg.top(), vg.top());
        double y1, y2;
        if (overlap < std::min(vg.height()/3.0, sg.height()/3.0))
        {
            // if less than 1/3 of views overlap, then just replicate the view
            y1 = vr.top();
            y2 = vr.bottom();
        }
        else
        {
            // views overlap; line them up
            double upp = vr.height() / vg.height();
            if(yInverted())
                y2 = vr.bottom() + (sg.bottom()-vg.bottom()) * upp;
            else
                y2 = vr.bottom() + (sg.top()-vg.top()) * upp;
            y1 = y2 - sg.height() * upp;
        }
        enableAutoRange(YAxis, false);
        setYRange(y1, y2, 0.0);
    }

    view->blockLink(false);
}

void ViewBoxBase::linkedXChanged()
{
    ViewBoxBase* view = mLinkedViews[0].data();
    if(view!=nullptr)
        linkedViewChanged(view, XAxis);
}

void ViewBoxBase::linkedYChanged()
{
    ViewBoxBase* view = mLinkedViews[1].data();
    if(view!=nullptr)
        linkedViewChanged(view, YAxis);
}


void ViewBoxBase::setViewRange(const Range& x, const Range& y)
{
    mViewRange[0] = Range(x);
    mViewRange[1] = Range(y);
}

void ViewBoxBase::setTargetRange(const Range &x, const Range &y)
{
    mTargetRange[0] = Range(x);
    mTargetRange[1] = Range(y);
}

void ViewBoxBase::setAutoRangeEnabled(const bool enableX, const bool enableY)
{
    mAutoRangeEnabled[0] = enableX;
    mAutoRangeEnabled[1] = enableY;
}

void ViewBoxBase::_resetTarget()
{
    // Reset target range to exactly match current view range.
    // This is used during mouse interaction to prevent unpredictable
    // behavior (because the user is unaware of targetRange).
    if(mAspectLocked == 0.0)    // interferes with aspect locking
    {
        mTargetRange[0] = Range(mViewRange[0]);
        mTargetRange[1] = Range(mViewRange[1]);
    }
}

void ViewBoxBase::setInnerSceneItem(GraphicsObject* innerItem)
{
    mInnerSceneItem = innerItem;
}

QVariant ViewBoxBase::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
    if(change == QGraphicsItem::ItemSceneChange)
    {
        // Disconnect from old scene
        QGraphicsScene2* oldScene = qobject_cast<QGraphicsScene2*>(scene());
        if(oldScene)
            QObject::disconnect(oldScene, SIGNAL(sigPrepareForPaint()), this, SLOT(prepareForPaint()));
    } else if (change == QGraphicsItem::ItemSceneHasChanged)
    {
        // Connect to a new scene
        QGraphicsScene2* newScene = qobject_cast<QGraphicsScene2*>(scene());
        if(newScene)
            QObject::connect(newScene, SIGNAL(sigPrepareForPaint()), this, SLOT(prepareForPaint()));
    }

    return GraphicsWidget::itemChange(change, value);
}

void ViewBoxBase::linkedWheelEvent(QGraphicsSceneWheelEvent *event, const ViewBoxBase::Axis axis)
{
    double sx = std::pow(0.02 + 1.0, event->delta() * mWheelScaleFactor);
    double sy = std::pow(0.02 + 1.0, event->delta() * mWheelScaleFactor);

    if((axis!=XAxis && axis!= XYAxes) || !mMouseEnabled[0])
        sx = 1.0;
    if((axis!=YAxis && axis!= XYAxes) || !mMouseEnabled[1])
        sy = 1.0;

    QPointF center = mChildGroup->transform().inverted().map(event->pos());

    _resetTarget();
    scaleBy(QPointF(sx, sy), center);

    emit sigRangeChangedManually(mMouseEnabled[0], mMouseEnabled[1]);

    event->accept();
}

void ViewBoxBase::mouseDragEvent(MouseDragEvent *event)
{
    mouseDragEvent(event, XYAxes);
}

void ViewBoxBase::mouseDragEvent(MouseDragEvent *event, const ViewBoxBase::Axis axis)
{
    event->accept();

    const QPointF pos = event->pos();
    const QPointF lastPos = event->lastPos();

    QPointF mask;
    mask.setX((mMouseEnabled[0] && (axis==XAxis || axis==XYAxes)) ? 1.0 : 0.0);
    mask.setY((mMouseEnabled[1] && (axis==YAxis || axis==XYAxes)) ? 1.0 : 0.0);

    // Scale or translate based on mouse button
    if(event->button() & (Qt::LeftButton | Qt::MidButton))
    {
        if(mMouseMode==RectMode)
        {
            if(event->isFinish())
            {
                // This is the final move in the drag; change the view scale now
                hideScaleBox();
                QRectF ax(event->buttonDownPos(event->button()), pos);
                ax = mChildGroup->mapRectFromParent(ax);
                showAxRect(ax);
                addToHistory(ax);
            }
            else
            {
                updateScaleBox(event->buttonDownPos(), pos);
            }
        }
        else
        {
            QTransform tr = mChildGroup->transform().inverted();
            QPointF dif = lastPos - pos;
            QPointF p = tr.map(QPointF(dif.x()*mask.x(), dif.y()*mask.y())) - tr.map(QPointF(0.0, 0.0));
            double x = (mask.x() == 1.0) ? p.x() : 0.0;
            double y = (mask.y() == 1.0) ? p.y() : 0.0;

            _resetTarget();
            if(x!=0.0 || y!=0.0)
                translateBy(x, y);

            emit sigRangeChangedManually(mMouseEnabled[0], mMouseEnabled[1]);
        }
    }
    else if(event->button() & Qt::RightButton)
    {
        if(mAspectLocked!=0.0)
            mask.setX(0.0);

        QPointF dif = QPointF(event->screenPos()) - QPointF(event->lastScreenPos());
        dif.setX(dif.x() * -1.0);

        QPointF s = mask * 0.02 + QPointF(1.0, 1.0);
        s.setX(std::pow(s.x(), dif.x()));
        s.setY(std::pow(s.y(), dif.y()));

        QTransform tr = mChildGroup->transform().inverted();
        QPointF center = tr.map(event->buttonDownPos(Qt::RightButton));

        double x = s.x() * mask.x();
        double y = s.y() * mask.y();

        _resetTarget();
        scaleBy(x, y, center);

        emit sigRangeChangedManually(mMouseEnabled[0], mMouseEnabled[1]);
    }
}

QList<QGraphicsItem *> ViewBoxBase::allChildren(QGraphicsItem *item) const
{
    // Return a list of all children and grandchildren of this ViewBox
    if(item==nullptr)
        item = mChildGroup;

    QList<QGraphicsItem *> children;
    children << item;
    QList<QGraphicsItem *> chItems = item->childItems();
    const int size = chItems.size();
    for(int i=0; i<size; ++i)
        children << allChildren(chItems[i]);
    return children;
}

void ViewBoxBase::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    if(mBorder.style()!=Qt::NoPen)
    {
        painter->setPen(mBorder);
        painter->drawPath(shape());
    }
}

QRectF ViewBoxBase::screenGeometry() const
{
    // Return the screen geometry of the viewbox
    QGraphicsView* v = getViewWidget();
    if(v==nullptr)
        return QRectF();

    QRectF b = sceneBoundingRect();
    b = v->mapFromScene(b).boundingRect();
    QPointF p = v->mapFromGlobal(v->pos());
    return b.adjusted(p.x(), p.y(), p.x(), p.y());
}

Point ViewBoxBase::viewPixelSize() const
{
    // Return the (width, height) of a screen pixel in view coordinates.

    QPointF o = mapToView(0.0, 0.0);
    QVector<Point> pp = pixelVectors();
    Point px(mapToView(pp[0]) - o);
    Point py(mapToView(pp[1]) - o);
    return Point(px.length(), py.length());
}

void ViewBoxBase::linkView(const ViewBoxBase::Axis axis, ViewBoxBase *view)
{
    switch (axis)
    {
    case XAxis:
        setXLink(view);
        break;
    case YAxis:
        setYLink(view);
        break;
    default:
        break;
    }
}

void ViewBoxBase::setXLink(ViewBoxBase* view)
{
    ViewBoxBase* oldView = mLinkedViews[0].data();
    if(oldView!=nullptr)
    {
        disconnect(oldView, SIGNAL(sigXRangeChanged(Range)), this, SLOT(linkedXChanged()));
        disconnect(oldView, SIGNAL(sigResized()), this, SLOT(linkedXChanged()));
    }

    mLinkedViews[0] = view;
    if(view!=nullptr)
    {
        connect(view, SIGNAL(sigXRangeChanged(Range)), this, SLOT(linkedXChanged()));
        connect(view, SIGNAL(sigResized()), this, SLOT(linkedXChanged()));

        if(view->mAutoRangeEnabled[0])
        {
            enableAutoRange(XAxis, false);
            linkedXChanged();
        }
        else if(mAutoRangeEnabled[0]==false)
        {
            linkedXChanged();
        }
    }

    emit sigStateChanged(this);
}

void ViewBoxBase::setYLink(ViewBoxBase *view)
{
    ViewBoxBase* oldView = mLinkedViews[1].data();
    if(oldView!=nullptr)
    {
        disconnect(oldView, SIGNAL(sigYRangeChanged(Range)), this, SLOT(linkedYChanged()));
        disconnect(oldView, SIGNAL(sigResized()), this, SLOT(linkedYChanged()));
    }

    mLinkedViews[1] = view;
    if(view!=nullptr)
    {
        connect(view, SIGNAL(sigYRangeChanged(Range)), this, SLOT(linkedYChanged()));
        connect(view, SIGNAL(sigResized()), this, SLOT(linkedYChanged()));

        if(view->mAutoRangeEnabled[1])
        {
            enableAutoRange(YAxis, false);
            linkedYChanged();
        }
        else if(mAutoRangeEnabled[1]==false)
        {
            linkedYChanged();
        }
    }

    emit sigStateChanged(this);
}

ViewBoxBase* ViewBoxBase::linkedView(const ViewBoxBase::Axis axis) const
{
    switch (axis)
    {
    case XAxis:
        return mLinkedViews[0].data();
    case YAxis:
        return mLinkedViews[1].data();
    default:
        return nullptr;
    }
}

/*
## used to connect/disconnect signals between a pair of views
if axis == ViewBox.XAxis:
    signal = 'sigXRangeChanged'
    slot = self.linkedXChanged
else:
    signal = 'sigYRangeChanged'
    slot = self.linkedYChanged


oldLink = self.linkedView(axis)
if oldLink is not None:
    try:
        getattr(oldLink, signal).disconnect(slot)
        oldLink.sigResized.disconnect(slot)
    except (TypeError, RuntimeError):
        ## This can occur if the view has been deleted already
        pass


if view is None or isinstance(view, basestring):
    self.state['linkedViews'][axis] = view
else:
    self.state['linkedViews'][axis] = weakref.ref(view)
    getattr(view, signal).connect(slot)
    view.sigResized.connect(slot)
    if view.autoRangeEnabled()[axis] is not False:
        self.enableAutoRange(axis, False)
        slot()
    else:
        if self.autoRangeEnabled()[axis] is False:
            slot()


self.sigStateChanged.emit(self)
*/

void ViewBoxBase::wheelEvent(QGraphicsSceneWheelEvent* event)
{
    linkedWheelEvent(event, XYAxes);
}

void ViewBoxBase::keyPressEvent(QKeyEvent *event)
{
    event->accept();

    switch (event->key()) {
    case Qt::Key_Minus:
        scaleHistory(-1);
        break;
    case Qt::Key_Plus:
    case Qt::Key_Equal:
        scaleHistory(1);
        break;
    case Qt::Key_Backspace:
        scaleHistory(mAxHistory.size());
        break;
    default:
        event->ignore();
        break;
    }
}

void ViewBoxBase::updateScaleBox(const QPointF& p1, const QPointF& p2)
{
    QRectF r = mChildGroup->mapRectFromParent(QRectF(p1, p2));
    mRbScaleBox->resetTransform();
    mRbScaleBox->setRect(r);
    mRbScaleBox->show();
}

void ViewBoxBase::hideScaleBox()
{
    mRbScaleBox->hide();
}

void ViewBoxBase::addToHistory(const QRectF &r)
{
    mAxHistory.append(QRectF(r));
    mAxHistoryPointer += 1;
}

