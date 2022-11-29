<center><font size='60'>Docker Command</font></center>

## 1.常用命令

1. **docker exec 进入容器**

   docker exec -it master /bin/bash

   

2. **docker cp 拷贝文件**

   #从本地机器拷贝到容器

   docker cp test.txt  master:/opt/

   #从容器拷贝到本地容器

   docker cp master:/opt/test.txt /home/