#include "TerminalInput.h"
#include <QDebug>
#include <unistd.h>

TerminalRawGuard::TerminalRawGuard()
{
    tcgetattr(STDIN_FILENO, &orig_);
    termios raw = orig_;
    cfmakeraw(&raw);
    raw.c_lflag |= ISIG;  // Ctrl+C 허용
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
}

// 종료 시 복구
TerminalRawGuard::~TerminalRawGuard()
{
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_);
}

// 생성자: stdin 감시 시작
TerminalInput::TerminalInput(QObject* parent)
    : QObject(parent),
      guard_(new TerminalRawGuard()),
      notifier_(new QSocketNotifier(STDIN_FILENO, QSocketNotifier::Read, this))
{
    connect(notifier_, &QSocketNotifier::activated, this, &TerminalInput::onReadyRead);
}

// 읽을 때마다 키 emit
void TerminalInput::onReadyRead()
{
    char ch;
    const ssize_t n = ::read(STDIN_FILENO, &ch, 1);
    if (n > 0)
    {
        emit keyChar(ch);
    }
}
