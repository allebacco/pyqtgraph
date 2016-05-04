#include "functions.h"
#include <cmath>
#include <QDebug>
/*
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
*/

QTransform inverted(const QTransform& tr, bool* invertible)
{
    if(invertible!=nullptr)
        *invertible = false;

    const double in_m[3][3] { {tr.m11(), tr.m12(), tr.m13()},
                              {tr.m21(), tr.m22(), tr.m23()},
                              {tr.m31(), tr.m32(), tr.m33()} };

    // Calculate the determinant of in_m (det)
    const double a = in_m[0][0]*(in_m[1][1]*in_m[2][2] - in_m[2][1]*in_m[1][2]);
    const double b = in_m[0][1]*(in_m[1][0]*in_m[2][2] - in_m[2][0]*in_m[1][2]);
    const double c = in_m[0][2]*(in_m[1][0]*in_m[2][1] - in_m[2][0]*in_m[1][1]);
    const double det = a - b + c;
    if(det==0.0)
        return QTransform();

    if(invertible!=nullptr)
        *invertible = true;

    // Calculate the adjoint matrix (out_m) of A
    double out_m[3][3];
    out_m[0][0] =   in_m[1][1]*in_m[2][2] - in_m[1][2]*in_m[2][1];
    out_m[0][1] = -(in_m[0][1]*in_m[2][2] - in_m[0][2]*in_m[2][1]);
    out_m[0][2] =   in_m[0][1]*in_m[1][2] - in_m[0][2]*in_m[1][1];
    out_m[1][0] = -(in_m[1][0]*in_m[2][2] - in_m[1][2]*in_m[2][0]);
    out_m[1][1] =   in_m[0][0]*in_m[2][2] - in_m[0][2]*in_m[2][0];
    out_m[1][2] = -(in_m[0][0]*in_m[1][2] - in_m[0][2]*in_m[1][0]);
    out_m[2][0] =   in_m[1][0]*in_m[2][1] - in_m[1][1]*in_m[2][0];
    out_m[2][1] = -(in_m[0][0]*in_m[2][1] - in_m[0][1]*in_m[2][1]);
    out_m[2][2] =   in_m[0][0]*in_m[1][1] - in_m[0][1]*in_m[1][0];

    // Calculate the inverse matrix of in_m (adj(in_m)/det)
    const double det_inv = 1.0/det;
    for(int i=0 ; i<3 ; i++)
        for(int j=0 ; j<3 ; j++)
            out_m[i][j] *= det_inv;

    return QTransform(out_m[0][0], out_m[0][1], out_m[0][2],
                      out_m[1][0], out_m[1][1], out_m[1][2],
                      out_m[2][0], out_m[2][1], out_m[2][2]);
}
