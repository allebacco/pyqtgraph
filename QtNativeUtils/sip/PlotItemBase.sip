

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

signals:

    void sigYRangeChanged(const Range& range);
    void sigXRangeChanged(const Range& range);
    void sigRangeChanged(const Range& xRange, const Range& yRange);
};
