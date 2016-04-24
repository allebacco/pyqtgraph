#ifndef PLOTITEM_H
#define PLOTITEM_H

#include <QList>

#include "Point.h"
#include "GraphicsWidget.h"
#include "ViewBoxBase.h"
#include "ItemDefines.h"


/*!
 * \brief Graphics item implementing a scalable ViewBox with plotting powers
 *
 * This class is one of the workhorses of pyqtgraph. It implements a graphics item with
 * plots, labels, and scales which can be viewed inside a QGraphicsScene. If you want
 * a widget that can be added to your GUI, see PlotWidget instead.
 * This class is very heavily featured:
 * - Automatically manages PlotCurveItems
 * - Fast display and update of plots
 * - Manages zoom/pan ViewBox, scale, and label elements
 * - Automatic scaling when data changes
 *
 * Plot graphics item that can be added to any graphics scene. Implements axes, titles, and interactive viewbox.
 * PlotItem also provides some basic analysis functionality that may be accessed from the context menu.
 * Use :func:`plot() <pyqtgraph.PlotItem.plot>` to create a new PlotDataItem and add it to the view.
 * Use :func:`addItem() <pyqtgraph.PlotItem.addItem>` to add any QGraphicsItem to the view
 *
 * This class wraps several methods from its internal ViewBox:
 * - ViewBox::setXRange
 * - ViewBox::setYRange
 * - ViewBox::setRange
 * - ViewBox::autoRange
 * - ViewBox::setXLink
 * - ViewBox::setYLink
 * - ViewBox::setAutoPan
 * - ViewBox::setAutoVisible
 * - ViewBox::setLimits
 * - ViewBox::viewRect
 * - ViewBox::viewRange
 * - ViewBox::setMouseEnabled
 * - ViewBox::enableAutoRange
 * - ViewBox::disableAutoRange
 * - ViewBox::setAspectLocked
 * - ViewBox::invertY
 * - ViewBox::invertX
 * - ViewBox::register
 * - ViewBox::unregister
 *
 * The ViewBox itself can be accessed by calling :func:`getViewBox() <pyqtgraph.PlotItem.getViewBox>`
 */
class PlotItemBase: public GraphicsWidget
{
    Q_OBJECT
public:
    PlotItemBase(QGraphicsItem* parent=nullptr, ViewBoxBase* viewBox=nullptr, Qt::WindowFlags wFlags=0);
    virtual ~PlotItemBase() {}

    enum { Type = CustomItemTypes::TypePlotItem };

    virtual int type() const
    {
        // Enable the use of qgraphicsitem_cast with this item.
        return Type;
    }

    virtual void forgetViewBox() {}

    /*!
     * \brief ViewBox inside the PlotItem
     * \return The ViewBox
     */
    virtual QObject* getViewBox() const { return mViewBox; }

    /*!
     * \brief ViewBox inside the PlotItem
     * \return The ViewBox
     */
    ViewBoxBase* getNativeViewBox() const;

    /*!
     * \brief Set the visible range of the ViewBox.
     * \param xRange The range that should be visible along the x-axis
     * \param yRange The range that should be visible along the y-axis
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     * \param disableAutoRange If True, auto-ranging is disabled. Otherwise, it is left unchanged
     */
    void setRange(const Range& xRange=Range(), const Range& yRange=Range(), const double padding=ViewBoxBase::AutoPadding, const bool disableAutoRange=true);

    /*!
     * \brief Set the visible range of the ViewBox.
     * \param rect The full range that should be visible in the view box.
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     * \param disableAutoRange If True, auto-ranging is disabled. Otherwise, it is left unchanged
     */
    void setRange(const QRectF& rect, const double padding=ViewBoxBase::AutoPadding, const bool disableAutoRange=true);

    /*!
     * \brief Set the visible X range of the view to [*min*, *max*].
     * \param minR Minimum value of the range
     * \param maxR Maximum value of the range
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     */
    void setXRange(const double minR, const double maxR, const double padding=ViewBoxBase::AutoPadding);

    /*!
     * \brief Set the visible Y range of the view to [*min*, *max*].
     * \param minR Minimum value of the range
     * \param maxR Maximum value of the range
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     */
    void setYRange(const double minR, const double maxR, const double padding=ViewBoxBase::AutoPadding);

signals:

    void sigYRangeChanged(const Range& range);
    void sigXRangeChanged(const Range& range);
    void sigRangeChanged(const Range& xRange, const Range& yRange);
};

#endif // PLOTITEM_H
