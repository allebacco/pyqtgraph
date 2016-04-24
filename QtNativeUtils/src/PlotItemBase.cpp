#include "PlotItemBase.h"

PlotItemBase::PlotItemBase(QGraphicsItem* parent, ViewBoxBase* viewBox, Qt::WindowFlags wFlags) :
    GraphicsWidget(parent, wFlags)
{
    Range::registerMetatype();
    if(viewBox==nullptr)
        viewBox = new ViewBoxBase();

    mViewBox = viewBox;
    mViewBox->setParentItem(this);
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
