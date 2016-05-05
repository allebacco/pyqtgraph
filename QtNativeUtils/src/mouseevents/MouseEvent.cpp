#include "MouseEvent.h"

MouseEvent::MouseEvent(): QEvent(EvType)
{
    mCurrentItem = nullptr;
}
