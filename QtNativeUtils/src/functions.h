#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <QPainterPath>
#include <QString>


QPainterPath arrayToQPath(const double* x, const double* y, const size_t size, const QString& connect);

#endif // FUNCTIONS_H
