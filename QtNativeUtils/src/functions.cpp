#include "functions.h"
#include <cmath>
#include <QDebug>

static void arrayToQPathAll(const double* x, const double* y, const size_t size, QPainterPath& path)
{
    path.moveTo(x[0], y[0]);
    for(size_t i=1; i<size; ++i)
        path.lineTo(x[i], y[i]);
}

static void arrayToQPathPairs(const double* x, const double* y, const size_t size, QPainterPath& path)
{
    for(size_t i=0; i<size; i+=2)
    {
        path.moveTo(x[i], y[i]);
        path.lineTo(x[i+1], y[i+1]);
    }
}


static void arrayToQPathFinite(const double* x, const double* y, const size_t size, QPainterPath& path)
{
    path.moveTo(x[0], y[0]);
    bool skip = true;
    for(size_t i=1; i<size; ++i)
    {
        if(std::isfinite(x[i]) && std::isfinite(y[i]))
        {
            if(skip)
                path.moveTo(x[i], y[i]);
            else
                path.lineTo(x[i], y[i]);
            skip = false;
        }
        else
            skip = true;
    }
}

QPainterPath arrayToQPath(const double* x, const double* y, const size_t size, const QString &connect=QString())
{
    QPainterPath path;
    if(size>0)
    {
        if(connect.isEmpty() || connect=="all")
            arrayToQPathAll(x, y, size, path);
        else if(connect=="finite")
            arrayToQPathFinite(x, y, size, path);
        else if(connect=="pairs")
            arrayToQPathPairs(x, y, size, path);
    }
    return path;
}
