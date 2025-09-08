#include "cameraworker.h"
#include <QApplication>
#include <QPushButton>   // ✅ QPushButton 선언 필요
#include <QMetaObject>   // ✅ invokeMethod 쓸 때 필요
#include <QtCore/Qt>
#include <TerminalInput.h>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    CameraWorker w;
    w.resize(960, 600);
    w.show();

    //터미널 입력
    auto* term = new TerminalInput(&w);
    QObject::connect(term, &TerminalInput::keyChar,
                     &w, &CameraWorker::handleTermKey);

    QMetaObject::invokeMethod(w.findChild<QPushButton*>("btnStart"), "click",
                             Qt::QueuedConnection);
    return app.exec();
}
