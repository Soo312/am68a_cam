QT       += core gui widgets
CONFIG   += c++17
TEMPLATE = app
TARGET = CameraWorker


SOURCES += main.cpp \
           ImageRenderHelper.cpp \
           PoseWorker.cpp \
           TerminalInput.cpp \
           cameraworker.cpp \
           person_detect_onnx.cpp \
           pose_estimate_onxx.cpp \
           yolo_pose_ipc.cpp \
           yolox_cpu_min.cpp

HEADERS += cameraworker.h \
           CaptureWorker.h \
           ImageRenderHelper.h \
           PoseWorker.h \
           TerminalInput.h \
           depth_lift.h \
           person_detect_onnx.h \
           pose_estimate_onnx.h \
           pose_ipc.hpp \
           skeleton_draw.hpp \
           utilHeader.h \
           yolo_pose_ipc.h

FORMS   += cameraworker.ui

SDK = /home/ArenaSDK/ArenaSDK_Linux_ARM64

SYSROOT = /home/vmware/ti-processor-sdk-linux-edgeai-j721s2-evm-10_01_00_04/linux-devkit/sysroots/aarch64-oe-linux


INCLUDEPATH += \
    $$SDK/include \
    $$SDK/GenICam/library/CPP/include \
    $$SYSROOT/usr/include/opencv4

# 헤더: vendor만 사용
INCLUDEPATH += $$PWD/third_party/onnxruntime-1.15.0/include
QMAKE_CXXFLAGS += -isystem $$PWD/third_party/onnxruntime-1.15.0/include

# 링크 디렉터리
LIBS += -L$$SDK/lib
LIBS += -L$$SDK/GenICam/library/lib/Linux64_ARM
LIBS += -L$$SDK/GenTL
LIBS += -L$$SYSROOT/usr/lib \
        -lopencv_core -lopencv_imgproc
s
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

DISTFILES += \
    yolo_pose_7060_srv.py
