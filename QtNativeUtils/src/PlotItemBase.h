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
    ViewBoxBase* getNativeViewBox() const { return mViewBox; }

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

    /*!
     * \brief Link this view's X axis to another view.
     * If view is nullptr, the axis is left unlinked.
     * \param view Other view to link with
     */
    void setXLink(ViewBoxBase* view);

    /*!
     * \brief Link this view's Y axis to another view.
     * If view is nullptr, the axis is left unlinked.
     * \param view Other view to link with
     */
    void setYLink(ViewBoxBase* view);

    void setAutoPan(const bool x=false, const bool y=false);
    void setAutoVisible(const bool x=false, const bool y=false);

    /*!
     * \brief Set the range of the view box to make all children visible.
     *
     * Note that this is not the same as enableAutoRange, which causes the view to
     * automatically auto-range whenever its contents are changed.
     *
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     */
    void autoRange(const double padding=ViewBoxBase::AutoPadding);

    /*!
     * \brief Set the range of the view box to make all children visible.
     *
     * Note that this is not the same as enableAutoRange, which causes the view to
     * automatically auto-range whenever its contents are changed.
     *
     * \param items List of items to consider when determining the visible range.
     * \param padding Expand the view by a fraction of the requested range.
     *                By default (AutoPadding), this value is set between 0.02 and 0.1 depending on
     *                the size of the ViewBox.
     */
    void autoRange(const QList<QGraphicsItem*>& items, const double padding=ViewBoxBase::AutoPadding);

    /*!
     * \brief Enable (or disable) auto-range for axis.
     *
     * When enabled, the axis will automatically rescale when items are added/removed or change their shape.
     *
     * \param axis Axis, which may be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes for both.
     * \param enable Enabling state
     */
    void enableAutoRange(const Axis axis=XYAxes, const bool enable=true);
    void disableAutoRange(const Axis ax=XYAxes);

    /*!
     * \brief Bounding of the region visible within the ViewBox
     * \return The bounding of the region visible within the ViewBox
     */
    virtual QRectF viewRect() const;

    const QList<Range>& viewRange() const;

    /*!
     * \brief Set whether each axis is enabled for mouse interaction
     *
     * This allows the user to pan/scale one axis of the view while leaving the other axis unchanged.
     *
     * \param enabledOnX true to eneble mouse interacion on x axis
     * \param enabledOnY true to eneble mouse interacion on y axis
     */
    void setMouseEnabled(const bool enabledOnX=true, const bool enabledOnY=true);

    /*!
     * \brief Set the padding limits that constrain the possible view ranges.
     * \param xMin Minumum value in the x range
     * \param xMax Maximum value in the x range
     */
    void setXLimits(const double xMin, const double xMax);

    /*!
     * \brief Set the padding limits that constrain the possible view ranges.
     * \param rng Allowed X range
     */
    void setXLimits(const Range& rng);

    /*!
     * \brief Set the padding limits that constrain the possible view ranges.
     * \param yMin Minumum value in the y range
     * \param yMax Maximum value in the y range
     */
    void setYLimits(const double yMin, const double yMax);

    /*!
     * \brief Set the padding limits that constrain the possible view ranges.
     * \param rng Allowed Y range
     */
    void setYLimits(const Range& rng);

    /*!
     * \brief Set the scaling limits that constrain the possible view ranges.
     * \param xMin Minimum allowed left-to-right span across the view.
     * \param xMax Maximum allowed left-to-right span across the view.
     */
    void setXRangeLimits(const double xMin, const double xMax);

    /*!
     * \brief Set the scaling limits that constrain the possible view ranges.
     * \param rng Range allowed left-to-right span across the view.
     */
    void setXRangeLimits(const Range& rng);

    /*!
     * \brief Set the scaling limits that constrain the possible view ranges.
     * \param yMin Minimum allowed top-to-bottom span across the view.
     * \param yMax Maximum allowed top-to-bottom span across the view.
     */
    void setYRangeLimits(const double yMin, const double yMax);

    /*!
     * \brief Set the scaling limits that constrain the possible view ranges.
     * \param rng Range allowed left-to-right span across the view.
     */
    void setYRangeLimits(const Range& rng);

    /*!
     * \brief Lock teh aspect ratio.
     * If the aspect ratio is locked, view scaling must always preserve the aspect ratio.
     * By default, the ratio is set to 1; x and y both have the same scaling.
     * This ratio can be overridden (xScale/yScale), or use 0.0 to lock in the current ratio.
     * \param lock true for locking the aspect
     * \param ratio New aspect ratio. 0.0 for setting the current aspect ratio.
     */
    void setAspectLocked(const bool lock=true, const double ratio=1.0);

    void invertY(const bool b=true);
    void invertX(const bool b=true);

signals:

    void sigYRangeChanged(const Range& range);
    void sigXRangeChanged(const Range& range);
    void sigRangeChanged(const Range& xRange, const Range& yRange);
};

#endif // PLOTITEM_H
