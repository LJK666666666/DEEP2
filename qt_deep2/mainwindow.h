#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <iostream>
#include <fstream>
#include <QMessageBox>
#include <QDialog>
#include <QFileDialog>
#include <QDebug>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    QString filename;                 //存放读取图片的路径
    QProcess * detect_exe;          //创建外部程序
    bool start_or_stop = false;     //false:停止状态 true:开始状态

private slots:
    //槽函数

    void show_image();
    void show_image2();//显示函数
    void on_startorend_clicked();//检测开关
    void on_file_name_button_clicked();//读取文件
    char* ReadCommunication(char* path);//读取通信文件
    void WriteCommunication(const char* path, QString str, int len);//写通信文件


private:
    Ui::MainWindow *ui;

signals:
    //信号函数
    void showImageSignal();
    void showImageSignal2();
};

#endif // MAINWINDOW_H
