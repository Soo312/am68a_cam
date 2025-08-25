QT       += core gui widgets
CONFIG   += c++17
TEMPLATE = app
TARGET = CameraWorker


SOURCES += main.cpp \
           ImageRenderHelper.cpp \
           cameraworker.cpp

HEADERS += cameraworker.h \
           ImageRenderHelper.h

FORMS   += cameraworker.ui

SDK = /home/ArenaSDK/ArenaSDK_Linux_ARM64

SYSROOT = /home/vmware/ti-processor-sdk-linux-edgeai-j721s2-evm-11_00_00_08/linux-devkit/sysroots/aarch64-oe-linux


INCLUDEPATH += \
    $$SDK/include \
    $$SDK/GenICam/library/CPP/include \
    $$SYSROOT/usr/include/opencv4

# 링크 디렉터리
LIBS += -L$$SDK/lib
LIBS += -L$$SDK/GenICam/library/lib/Linux64_ARM
LIBS += -L$$SDK/GenTL
LIBS += -L$$SYSROOT/usr/lib \
        -lopencv_core -lopencv_imgproc

QMAKE_LIBDIR += \
    $$SDK/lib \
    $$SDK/GenICam/library/lib/Linux64_ARM \
    $$SDK/GenTL \
    $$SYSROOT/usr/lib

QMAKE_CXXFLAGS += --sysroot=$$SYSROOT
QMAKE_LFLAGS   += --sysroot=$$SYSROOT

QMAKE_LFLAGS   += -Wl,-rpath-link,$$SYSROOT/usr/lib

LIBS += -ltbb

# 의존하는 순서대로!  arena -> GenTL -> GenApi -> GCBase
QMAKE_LFLAGS += -Wl,--no-as-needed
LIBS += -larena \
        -llucidlog \
        -lgentl \
        -lGenApi_gcc54_v3_3_LUCID \
        -lGCBase_gcc54_v3_3_LUCID

QMAKE_LFLAGS += -Wl,-rpath,/usr/lib

# (디버깅용) qmake가 먹인 경로 확인
message(SYSROOT=$$SYSROOT)
message(INCLUDEPATH=$$INCLUDEPATH)
message(QMAKE_LIBDIR=$$QMAKE_LIBDIR)
message(LFLAGS=$$QMAKE_LFLAGS)
message(LIBS=$$LIBS)
