#include "cameraworker.h"
#include <QApplication>
#include <QPushButton>   // ✅ QPushButton 선언 필요
#include <QMetaObject>   // ✅ invokeMethod 쓸 때 필요
#include <QtCore/Qt>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    CameraWorker w;
    w.resize(960, 600);
    w.show();

    QMetaObject::invokeMethod(w.findChild<QPushButton*>("btnStart"), "click",
                             Qt::QueuedConnection);
    return app.exec();
}
