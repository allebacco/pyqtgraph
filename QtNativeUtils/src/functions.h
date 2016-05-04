#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <QPainterPath>
#include <QString>
#include <QTransform>

#include "internal/functions_impl.h"

//QPainterPath arrayToQPath(const double* x, const double* y, const size_t size, const QString& connect);

template<typename _Tp1, typename _Tp2>
static QPainterPath arrayToQPath(const _Tp1* x, const _Tp2* y, const size_t size, const QString &connect=QString())
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

QTransform inverted(const QTransform& tr, bool *invertible=nullptr);

#endif // FUNCTIONS_H
