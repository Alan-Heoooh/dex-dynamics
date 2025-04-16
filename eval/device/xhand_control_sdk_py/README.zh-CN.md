# XHandControl Python SDK 使用说明

<p align="center">
  <a target="_blank" href="./README.md">国际版</a>
</p>

## 介绍
XHandControl Python SDK 是一套软件开发工具包，用于与 XHand 机械手进行通信，控制机械手的关节和读取传感器数据。

## 安装
```bash
sudo apt install libcrypto++-dev
```

## 构建
```bash
sudo python3 setup.py install sdist bdist_wheell
sudo ldconfig
```

## 安装
```bash
sudo pip3 install .

```

## 运行
```bash
python3 xhand_control_example.py
```