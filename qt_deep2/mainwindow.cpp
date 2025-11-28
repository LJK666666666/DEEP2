#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //设置界面名称
    setWindowTitle("调用外部程序");

    //将信号函数与显示图片的槽函数进行连接
    connect(this,SIGNAL(showImageSignal()), this, SLOT(show_image()));
    connect(this,SIGNAL(showImageSignal2()), this, SLOT(show_image2()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

//显示图片
void MainWindow::show_image()
{

    //读取一帧识别结果图片
    QImage *img_mainicon;//主图标显示在右上角lable中
    img_mainicon =new QImage;//新建一个image对象
    QString file_name = ui->file_name_Edit->text();
    img_mainicon->load(file_name); //载入图片到img对象中
    img_mainicon->scaled(ui->image_show->size(),Qt::KeepAspectRatio);//调整图片的大小
    ui->image_show->setScaledContents(true);
    ui->image_show->setPixmap(QPixmap::fromImage(*img_mainicon)); //将图片放入label，使用setPixmap,注意指针*img   
    //在label上显示
    ui->image_show->show();
}

void MainWindow::show_image2()
{
    //读取一帧识别结果图片
    QImage *img_mainicon;//主图标显示在右上角lable中
    img_mainicon =new QImage;//新建一个image对象
    //QString c = ReadCommunication("/home/b401-25/SSD-Resnet-Attention-FeatureFusion-master/file_name.txt");
    //QString file_name = "/home/b401-25/SSD-Resnet-Attention-FeatureFusion-master/demo/result/" + c;
    QString file_name="/home/b401-25/shiyan2/SSD-Resnet-Attention-FeatureFusion-master/demo/result/result.jpg";
    img_mainicon->load(file_name); //载入图片到img对象中
    img_mainicon->scaled(ui->label->size(),Qt::KeepAspectRatio);//调整图片的大小
    ui->label->setScaledContents(true);
    ui->label->setPixmap(QPixmap::fromImage(*img_mainicon)); //将图片放入label，使用setPixmap,注意指针*img
    //在label上显示
    ui->label->show();
}

//开始检测或停止检测
void MainWindow::on_startorend_clicked()
{
    if(!start_or_stop)
    {
        ui->state_Edit->setText("开始检测！");
        //system("cd /home/b401-13/SSD-Resnet-Attention-FeatureFusion-master/ && python3 demo.py --config-file configs/resnet50_ssd300_voc0712.yaml");
        system("cd /home/b401-25/shiyan2/SSD-Resnet-Attention-FeatureFusion-master && python3 demo1.py");
        showImageSignal();
        showImageSignal2();
    }
    else//再次点击，结束
    {
     /*重新设定按钮属性*/
     ui->state_Edit->setText("停止检测！");
     ui->image_show->clear();
     ui->label->clear();
     ui->image_show->setText("original");
     ui->label->setText("result");
     //ui->picture->setEnabled(true);
    }
    //标志位反转
    start_or_stop = !start_or_stop;
}

/*
 * 函数名称：ReadCommunication
 * 函数功能：读取通信文件内容
 * 输入参数：通信文件的路径
 * 输出参数：无
 * 返 回 值：读取到的内容
 */
char* MainWindow::ReadCommunication(char* path)
{
    static char temp[100];

    /*创建文件流，进行文件读操作*/
    ifstream communication_file;
    communication_file.open(path);
    communication_file.getline(temp,sizeof(temp));
    communication_file.close();

    //返回读到的字符串
    return temp;
}

//获取读取图片的路径,并将其输入到通信文件中
void MainWindow::on_file_name_button_clicked()
{
    QString s=QFileDialog::getOpenFileName(\
                this, "选择YUV文件",
                        "/",//初始目录
                 "图片文件((*.png *jpg *bmp);;");
    if (!s.isEmpty())
    {
        //s.replace("/","\\");
        ui->file_name_Edit->setText(s);
        filename=s;
    }
    //写要检测图片的路径   
    //QString process="cp ";
    //QString a = " /home/b401-25/SSD-Resnet-Attention-FeatureFusion-master/demo/";
    //process = process+filename+a;
    //QProcess::execute(process);
    //QStringList list = filename.split("/");
    //QString b = list[list.size()-1];
    WriteCommunication("/home/b401-25/shiyan2/file_name.txt",filename,filename.size());
}

/*
 * 函数名称：WriteCommunication
 * 函数功能：写入通信文件
 * 输入参数：通信文件的路径, 写入的内容，写入的字节长度
 * 输出参数：无
 * 返 回 值：无
 */
void MainWindow::WriteCommunication(const char* path, QString str, int len)
{
    //将QString类型转换为const char *
    char*  ch;
    QByteArray ba = str.toLatin1();
    ch=ba.data();

    /*创建文件流，进行文件写操作*/
    ofstream communication_file;
    communication_file.open(path);
    communication_file.write(ch, len);
    communication_file.close();
}
