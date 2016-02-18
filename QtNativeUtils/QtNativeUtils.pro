#-------------------------------------------------
#
# Project created by QtCreator 2016-02-16T20:55:07
#
#-------------------------------------------------

TARGET = QtNativeUtils
TEMPLATE = lib
CONFIG += static

QMAKE_CXXFLAGS += -std=c++11

INCLUDEPATH += src \
               src/mouseevents

SOURCES += src/QtNativeUtils.cpp \
           src/mouseevents/MouseClickEvent.cpp \
           src/mouseevents/HoverEvent.cpp \
           src/mouseevents/MouseDragEvent.cpp \
           src/mouseevents/MouseEvent.cpp

HEADERS += src/QtNativeUtils.h \
           src/mouseevents/MouseEvent.h \
           src/mouseevents/HoverEvent.h \
           src/mouseevents/MouseDragEvent.h \
           src/mouseevents/MouseClickEvent.h

DISTFILES += sip/Exceptions.sip \
             sip/QtNativeUtils.sip

#NUMPY_INCLUDE = /usr/local/lib/python2.7/dist-packages/numpy/core/include
NUMPY_INCLUDE = /usr/lib/python2.7/dist-packages/numpy/core/include
PYTHON_INCLUDE = /usr/include/python2.7

LIBS += python2.7

INCLUDEPATH += $$NUMPY_INCLUDE
INCLUDEPATH += $$PYTHON_INCLUDE

OTHER_FILES += \
    sip/MouseEvent.sip \
    sip/MouseClickEvent.sip


