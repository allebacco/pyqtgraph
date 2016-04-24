

class PlotItemBase: public GraphicsWidget
{

%TypeHeaderCode
#include "PlotItemBase.h"
%End

public:
    PlotItemBase(QGraphicsItem* parent=nullptr, ViewBoxBase* viewBox=nullptr, Qt::WindowFlags wFlags=0);
    virtual ~PlotItemBase();

    virtual int type() const /HoldGIL/;

    virtual void forgetViewBox();
    virtual QObject* getViewBox() const;

    void setRange() /NoArgParser/;
    %MethodCode
        // rect=None, xRange=None, yRange=None, padding=None, update=True, disableAutoRange=True
        // PyObject *sipSelf, PyObject *sipArgs, PyObject *sipKwds
        char* keywords[] = {"rect", "xRange", "yRange", "padding", "update", "disableAutoRange", nullptr};
        PyObject* pyRect = Py_None;
        PyObject* pyXRange = Py_None;
        PyObject* pyYRange = Py_None;
        PyObject* pyPadding = Py_None;
        PyObject* pyUpdate = Py_True;
        PyObject* pyDisableAutoRange = Py_True;

        bool sipSelfWasArg = (!sipSelf || sipIsDerived((sipSimpleWrapper *)sipSelf));

        if(!sipSelfWasArg)
        {
            if(!PyArg_ParseTupleAndKeywords(sipArgs, sipKwds, "O|OOOOOO", keywords, &sipSelf, &pyRect, &pyXRange, &pyYRange, &pyPadding, &pyUpdate, &pyDisableAutoRange))
            {
                return NULL;
            }
        }
        else
        {
            if(!PyArg_ParseTupleAndKeywords(sipArgs, sipKwds, "|OOOOOO", keywords, &pyRect, &pyXRange, &pyYRange, &pyPadding, &pyUpdate, &pyDisableAutoRange))
            {
                return NULL;
            }
        }

        if(!sipCanConvertToType(sipSelf, sipType_PlotItemBase, SIP_NOT_NONE))
        {
            return NULL;
        }

        int vbstate = 0, sipIsErr = 0;
        PlotItemBase* plotItem = reinterpret_cast<PlotItemBase*>(sipConvertToType(sipSelf, sipType_PlotItemBase, 0, 0, &vbstate, &sipIsErr));

        double padding = ViewBoxBase::AutoPadding;
        if(pyPadding!=Py_None)
            padding = PyFloat_AS_DOUBLE(pyPadding);
        bool disableAutoRange = PyObject_IsTrue(pyDisableAutoRange);

        // try to call with rect
        if(sipCanConvertToType(pyRect, sipType_QRectF, SIP_NOT_NONE))
        {
            int state = 0;
            QRectF* rect = reinterpret_cast<QRectF*>(sipConvertToType(pyRect, sipType_QRectF, 0, SIP_NOT_NONE, &state, &sipIsErr));

            plotItem->setRange(*rect, padding, disableAutoRange);

            sipReleaseType(rect, sipType_QRectF, state);
            sipReleaseType(plotItem, sipType_PlotItemBase, vbstate);
            Py_RETURN_NONE;
        }

        Range xRange;
        Range yRange;

        if(sipCanConvertToType(pyXRange, sipType_Range, SIP_NOT_NONE))
        {
            int xstate = 0;
            Range* r = reinterpret_cast<Range*>(sipConvertToType(pyXRange, sipType_Range, 0, SIP_NOT_NONE, &xstate, &sipIsErr));
            xRange.setRange(r->min(), r->max());
            sipReleaseType(r, sipType_Range, xstate);
        }

        if(sipCanConvertToType(pyYRange, sipType_Range, SIP_NOT_NONE))
        {
            int ystate = 0;
            Range* r = reinterpret_cast<Range*>(sipConvertToType(pyYRange, sipType_Range, 0, SIP_NOT_NONE, &ystate, &sipIsErr));
            yRange.setRange(r->min(), r->max());
            sipReleaseType(r, sipType_Range, ystate);
        }

        plotItem->setRange(xRange, yRange, padding, disableAutoRange);

        sipReleaseType(plotItem, sipType_PlotItemBase, vbstate);

        Py_RETURN_NONE;

    %End

    void setXRange(const double minR, const double maxR, const double padding=ViewBoxBase::AutoPadding);
    void setYRange(const double minR, const double maxR, const double padding=ViewBoxBase::AutoPadding);

    void setXLink(ViewBoxBase* view);
    void setYLink(ViewBoxBase* view);

    void setAutoPan(const bool x=false, const bool y=false);
    void setAutoVisible(const bool x=false, const bool y=false);

    void autoRange(const double padding=ViewBoxBase::AutoPadding);
    void autoRange(const QList<QGraphicsItem*>& items, const double padding=ViewBoxBase::AutoPadding);

    void enableAutoRange(const Axis axis=XYAxes, const bool enable=true);
    void disableAutoRange(const Axis ax=XYAxes);

    virtual QRectF viewRect() const;
    const QList<Range>& viewRange() const;

    void setMouseEnabled(const bool enabledOnX=true, const bool enabledOnY=true);

    void setXLimits(const Range& rng);
    void setYLimits(const Range& rng);
    void setXRangeLimits(const Range& rng);
    void setYRangeLimits(const Range& rng);

    void setLimits() /NoArgParser/;
    %MethodCode
        // "xMin", "xMax", "yMin", "yMax", "minXRange", "maxXRange", "minYRange", "maxYRange"
        // PyObject *sipSelf, PyObject *sipArgs, PyObject *sipKwds

        bool sipSelfWasArg = (!sipSelf || sipIsDerived((sipSimpleWrapper *)sipSelf));

        if(!sipSelfWasArg)
        {
            if(!PyTuple_Size(sipArgs)==1)
                return NULL;

            sipSelf = PyTuple_GetItem(sipArgs, 0);
        }

        if(!sipCanConvertToType(sipSelf, sipType_PlotItemBase, SIP_NOT_NONE))
        {
            return NULL;
        }

        int vbstate = 0, sipIsErr = 0;
        PlotItemBase* plotitem = reinterpret_cast<PlotItemBase*>(sipConvertToType(sipSelf, sipType_PlotItemBase, 0, 0, &vbstate, &sipIsErr));

        ViewBoxBase* viewBox = plotitem->getNativeViewBox();
        Range rng = viewBox->xLimits();
        PyObject* val = PyDict_GetItemString(sipKwds, "xMin");
        if(val!=nullptr)
            rng.setMin((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        val = PyDict_GetItemString(sipKwds, "xMax");
        if(val!=nullptr)
            rng.setMax((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        viewBox->setXLimits(rng);

        rng = viewBox->yLimits();
        val = PyDict_GetItemString(sipKwds, "yMin");
        if(val!=nullptr)
            rng.setMin((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        val = PyDict_GetItemString(sipKwds, "yMax");
        if(val!=nullptr)
            rng.setMax((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        viewBox->setYLimits(rng);

        rng = viewBox->xRangeLimits();
        val = PyDict_GetItemString(sipKwds, "minXRange");
        if(val!=nullptr)
            rng.setMin((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        val = PyDict_GetItemString(sipKwds, "maxXRange");
        if(val!=nullptr)
            rng.setMax((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        viewBox->setXRangeLimits(rng);

        rng = viewBox->yRangeLimits();
        val = PyDict_GetItemString(sipKwds, "minYRange");
        if(val!=nullptr)
            rng.setMin((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        val = PyDict_GetItemString(sipKwds, "maxYRange");
        if(val!=nullptr)
            rng.setMax((val==Py_None) ? Range::NO_LIMIT : pyAsDouble(val));
        viewBox->setYRangeLimits(rng);

        viewBox->updateViewRange();

        sipReleaseType(plotitem, sipType_PlotItemBase, vbstate);

        Py_RETURN_NONE;
    %End

    void setAspectLocked(const bool lock=true, const double ratio=1.0);

    void invertY(const bool b=true);
    void invertX(const bool b=true);

signals:

    void sigYRangeChanged(const Range& range);
    void sigXRangeChanged(const Range& range);
    void sigRangeChanged(const Range& xRange, const Range& yRange);
};
