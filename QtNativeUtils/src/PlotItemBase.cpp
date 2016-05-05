#include "PlotItemBase.h"

#include <QGraphicsGridLayout>

PlotItemBase::PlotItemBase(QGraphicsItem* parent, ViewBoxBase* viewBox, Qt::WindowFlags wFlags) :
    GraphicsWidget(parent, wFlags)
{
    Range::registerMetatype();
    if(viewBox==nullptr)
        viewBox = new ViewBoxBase();

    mViewBox = viewBox;
    mViewBox->setParentItem(this);

    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    QGraphicsGridLayout* gridLayout = new QGraphicsGridLayout();
    gridLayout->setContentsMargins(1,1,1,1);
    gridLayout->setHorizontalSpacing(0);
    gridLayout->setVerticalSpacing(0);
    setLayout(gridLayout);

    connect(mViewBox, SIGNAL(sigRangeChanged(Range,Range)), this, SIGNAL(sigRangeChanged(Range,Range)));
    connect(mViewBox, SIGNAL(sigXRangeChanged(Range)), this, SIGNAL(sigXRangeChanged(Range)));
    connect(mViewBox, SIGNAL(sigYRangeChanged(Range)), this, SIGNAL(sigYRangeChanged(Range)));

    //self.vb.sigStateChanged.connect(self.viewStateChanged)
    //self.setMenuEnabled(enableMenu, enableMenu) ## en/disable plotitem and viewbox menus
}

void PlotItemBase::setRange(const Range &xRange, const Range &yRange, const double padding, const bool disableAutoRange)
{
    mViewBox->setRange(xRange, yRange, padding, disableAutoRange);
}

void PlotItemBase::setRange(const QRectF &rect, const double padding, const bool disableAutoRange)
{
    mViewBox->setRange(rect, padding, disableAutoRange);
}

void PlotItemBase::setXRange(const double minR, const double maxR, const double padding)
{
    mViewBox->setXRange(minR, maxR, padding);
}

void PlotItemBase::setYRange(const double minR, const double maxR, const double padding)
{
    mViewBox->setYRange(minR, maxR, padding);
}

void PlotItemBase::setXLink(ViewBoxBase *view)
{
    mViewBox->setXLink(view);
}

void PlotItemBase::setYLink(ViewBoxBase *view)
{
    mViewBox->setYLink(view);
}

void PlotItemBase::setAutoPan(const bool x, const bool y)
{
    mViewBox->setAutoPan(x, y);
}

void PlotItemBase::setAutoVisible(const bool x, const bool y)
{
    mViewBox->setAutoVisible(x, y);
}

void PlotItemBase::autoRange(const double padding)
{
    mViewBox->autoRange(padding);
}

void PlotItemBase::autoRange(const QList<QGraphicsItem *> &items, const double padding)
{
    mViewBox->autoRange(items, padding);
}

void PlotItemBase::enableAutoRange(const ExtendedItem::Axis axis, const bool enable)
{
    mViewBox->enableAutoRange(axis, enable);
}

void PlotItemBase::disableAutoRange(const ExtendedItem::Axis ax)
{
    mViewBox->disableAutoRange(ax);
}

QRectF PlotItemBase::viewRect() const
{
    return mViewBox->viewRect();
}

const QList<Range>& PlotItemBase::viewRange() const
{
    return mViewBox->viewRange();
}

void PlotItemBase::setMouseEnabled(const bool enabledOnX, const bool enabledOnY)
{
    mViewBox->setMouseEnabled(enabledOnX, enabledOnY);
}

void PlotItemBase::setXLimits(const double xMin, const double xMax)
{
    mViewBox->setXLimits(xMin, xMax);
}

void PlotItemBase::setXLimits(const Range &rng)
{
    mViewBox->setXLimits(rng);
}

void PlotItemBase::setYLimits(const double yMin, const double yMax)
{
    mViewBox->setYLimits(yMin, yMax);
}

void PlotItemBase::setYLimits(const Range &rng)
{
    mViewBox->setYLimits(rng);
}

void PlotItemBase::setXRangeLimits(const double xMin, const double xMax)
{
    mViewBox->setXRangeLimits(xMin, xMax);
}

void PlotItemBase::setXRangeLimits(const Range &rng)
{
    mViewBox->setXRangeLimits(rng);
}

void PlotItemBase::setYRangeLimits(const double yMin, const double yMax)
{
    mViewBox->setYRangeLimits(yMin, yMax);
}

void PlotItemBase::setYRangeLimits(const Range &rng)
{
    mViewBox->setYRangeLimits(rng);
}

void PlotItemBase::setAspectLocked(const bool lock, const double ratio)
{
    mViewBox->setAspectLocked(lock, ratio);
}

void PlotItemBase::invertY(const bool b)
{
    mViewBox->invertY(b);
}

void PlotItemBase::invertX(const bool b)
{
    mViewBox->invertX(b);
}
