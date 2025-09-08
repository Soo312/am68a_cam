#ifndef TERMINALINPUT_H
#define TERMINALINPUT_H

#pragma once
#include <QObject>
#include <QSocketNotifier>
#include <termios.h>
#include <unistd.h>

#include <bits/unique_ptr.h>

class TerminalRawGuard
{
private:
    termios orig_ {};

public:
    TerminalRawGuard();
    ~TerminalRawGuard();


};

class TerminalInput : public QObject
{
    Q_OBJECT
public:
    explicit TerminalInput(QObject* parent = nullptr);

signals:
    void keyChar(char ch);

private slots:
    void onReadyRead();


private:
    std::unique_ptr<TerminalRawGuard> guard_;
    QSocketNotifier* notifier_;

};



#endif // TERMINALINPUT_H
