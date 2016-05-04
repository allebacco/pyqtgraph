#ifndef FUNCTIONS_IMPL_H
#define FUNCTIONS_IMPL_H

#include <QPainterPath>
#include <QString>

template<typename _Tp1, typename _Tp2>
static void arrayToQPathAll(const _Tp1* x, const _Tp2* y, const size_t size, QPainterPath& path)
{
    path.moveTo(x[0], y[0]);
    for(size_t i=1; i<size; ++i)
        path.lineTo(x[i], y[i]);
}

template<typename _Tp1, typename _Tp2>
static void arrayToQPathPairs(const _Tp1* x, const _Tp2* y, const size_t size, QPainterPath& path)
{
    for(size_t i=0; i<size; i+=2)
    {
        path.moveTo(x[i], y[i]);
        path.lineTo(x[i+1], y[i+1]);
    }
}


template<typename _Tp1, typename _Tp2>
static void arrayToQPathFinite(const _Tp1* x, const _Tp2* y, const size_t size, QPainterPath& path,
                               typename std::enable_if<std::is_floating_point<_Tp2>::value >::type* = 0)
{
    path.moveTo(x[0], y[0]);
    bool skip = true;
    for(size_t i=1; i<size; ++i)
    {
        //if(std::isfinite(x[i]) && std::isfinite(y[i]))
        if(std::isfinite(y[i]))
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

template<typename _Tp1, typename _Tp2>
static void arrayToQPathFinite(const _Tp1* x, const _Tp2* y, const size_t size, QPainterPath& path,
                               typename std::enable_if<std::is_integral<_Tp2>::value >::type* = 0)
{
    // Integers are always finite
    arrayToQPathAll(x, y, size, path);
}

/*
template<typename _Tp>
static QPainterPath arrayToQPath(const _Tp* x, const _Tp* y, const size_t size, const QString &connect=QString())
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
*/

#endif // FUNCTIONS_IMPL_H
