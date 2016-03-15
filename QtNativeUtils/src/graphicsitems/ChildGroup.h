#ifndef CHILDGROUP_H
#define CHILDGROUP_H

#include <QVector>
#include <QVariant>

#include "ItemGroup.h"
#include "Interfaces.h"

class ChildGroup : public ItemGroup
{
    Q_OBJECT
public:
    explicit ChildGroup(QGraphicsItem *parent = 0);
    virtual ~ChildGroup();

    void addListener(ItemChangedListener* listener);

protected:

    virtual QVariant itemChange(GraphicsItemChange change, const QVariant& value);

protected:

    QVector<ItemChangedListener*> mListeners;

};

#endif // CHILDGROUP_H
