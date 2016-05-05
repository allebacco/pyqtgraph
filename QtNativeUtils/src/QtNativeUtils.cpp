#include "QtNativeUtils.h"


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL numpy_ARRAY_API
#include <numpy/arrayobject.h>

#include "Range.h"
#include "mouseevents/MouseEvent.h"


void init_package()
{
    Range::registerMetatype();
    int type = QEvent::registerEventType(MouseEvent::EvType);
    if(type!=MouseEvent::EvType)
        qWarning("Error in registering MouseEvent event");

    import_array();
}

