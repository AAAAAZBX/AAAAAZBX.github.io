---
title: "Linux相关语法"
date: "2025-12-01"
description: "Linux常用命令和语法总结，包括文件管理、权限控制等基础操作"
tags: ["linux", "命令行", "系统"]
---

## 常用文件管理命令

### 一、命令介绍

ㅤㅤ1. `Ctrl c`：取消命令，并且换行

ㅤㅤ2. `Ctrl u`：清空本行命令

ㅤㅤ3. `tab键`：可以补全命令和文件名，如果补全不了快速按两下tab键，可以显示备选选项

ㅤㅤ4. `ls`：列出当前目录下所有文件，蓝色的是文件夹，白色的是普通文件，绿色的是可执行文件

ㅤㅤㅤ●ㅤ`ls -a`：查看所有文件包括隐藏文件（以.开头的文件就是隐藏文件）

ㅤㅤㅤ●ㅤ`ls -l`：查看当前路径下文件的读、写、执行权限

ㅤㅤㅤ●ㅤ`ls | wc -l`：查看ls下有多少个文件

ㅤㅤ5. `pwd`：显示当前路径

ㅤㅤ6. `cd XXX`：进入`XXX`目录下，`cd ..`返回上层目录

ㅤㅤㅤ●ㅤ`.`：当前目录`..`：上级目录

ㅤㅤㅤ●ㅤ`~`：家目录，回回到路径`/home/acs`下

ㅤㅤㅤ●ㅤ`cd -`：返回改变路径前的路径，比如当前在`/home/acs/homework`然后`cd** **/`这个时候就处于/目录下，然后`cd -`就会回到改变路径前的路径也就是`/home/acs/homework`

ㅤㅤ7. `cp XXX YYY`：将`XXX`文件复制成`YYY`，`XXX`和`YYY`可以是同一个路径，比如`../dir_c/a.txt`，表示上层目录下的`dir_c`文件夹下的文件`a.txt`

ㅤㅤㅤ●ㅤ`cp XXX YYY -r`将`XXX`目录（文件夹）复制到`YYY`下

ㅤㅤㅤ●ㅤ非当前路径重命名方法：`cp a.txt ../b.txt`

ㅤㅤ8. `mkdir XXX`：创建目录（文件夹）`XXX`

ㅤㅤㅤ●ㅤ`mkdir -p：-p`：如果文件夹不存在，则创建`

ㅤㅤ9. `rm XXX`：删除普通文件； `rm XXX -r`：删除文件夹

ㅤㅤㅤ●ㅤ支持正则表达式，删除所有`.txt`类型文件：`rm *.txt`

ㅤㅤㅤ●ㅤ删除所有文件（不包括文件夹）：`rm *`

ㅤㅤㅤ●ㅤ正则表达式删除所有文件夹：`rm * -r`即可

ㅤㅤ10. `mv XXX YYY`：将`XXX`文件移动到`YYY`下，和`cp`命令一样，`XXX`和`YYY`可以是同一个路径；重命名也是用这个命令

ㅤㅤㅤ●ㅤ非当前路径移动方法：`mv a.txt ../b.txt`

ㅤㅤ11. `touch XXX`：创建一个文件

ㅤㅤ12. `cat XXX`：展示文件`XXX`中的内容

ㅤㅤ13. 复制文本：`windows/Linux`下：`Ctrl + insert`，Mac下：`command + c`

ㅤㅤ14. 粘贴文本：`windows/Linux`下：`Shift + insert`，Mac下：`command + v`

ㅤㅤ15. `history`：查看历史输入指令

ㅤㅤ16. `tree`：以树形显示文件目录结构

## tmux相关操作

### 功能：

​    (1) 分屏。
​    (2) 允许断开`Terminal`连接后，继续运行进程。
结构：
​    一个`tmux`可以包含多个`session`，一个`session`可以包含多个`window`，一个`window`可以包含多个`pane`。
​    实例：

```
        tmux:
            session 0:
                window 0:
                    pane 0
                    pane 1
                    pane 2
                    ...
                window 1
                window 2
                ...
            session 1
            session 2
            ...
```

### 操作：

​    (1) `tmux`：新建一个`session`，其中包含一个`window`，`window`中包含一个`pane`，pane里打开了一个`shell`对话框。
​    (2) 按下`Ctrl + a`后手指松开，然后按%：将当前`pane`左右平分成两个`pane`。
​    (3) 按下`Ctrl + a`后手指松开，然后按"（注意是双引号"）：将当前`pane`上下平分成两个`pane`。
​    (4) `Ctrl + d`：关闭当前`pane`；如果当前`window`的所有`pane`均已关闭，则自动关闭`window`；如果当前`session`的所有`window`均已关闭，则自动关闭`session`。
​    (5) 鼠标点击可以选`pane`。
​    (6) 按下`ctrl + a`后手指松开，然后按方向键：选择相邻的`pane`。
​    (7) 鼠标拖动`pane`之间的分割线，可以调整分割线的位置。
​    (8) 按住`ctrl + a`的同时按方向键，可以调整`pane`之间分割线的位置。
​    (9) 按下`ctrl + a`后手指松开，然后按z：将当前`pane`全屏/取消全屏。
​    (10) 按下`ctrl + a`后手指松开，然后按d：挂起当前`session`。
​    (11) `tmux a`：打开之前挂起的`session`。
​    (12) 按下`ctrl + a`后手指松开，然后按s：选择其它`session`。
​        方向键 —— 上：选择上一项 `session/window/pane`
​        方向键 —— 下：选择下一项 `session/window/pane`
​        方向键 —— 右：展开当前项 `session/window`
​        方向键 —— 左：闭合当前项 `session/window`
​    (13) 按下`Ctrl + a`后手指松开，然后按`c`：在当前`session`中创建一个新的window。
​    (14) 按下`Ctrl + a`后手指松开，然后按`w`：选择其他`window`，操作方法与(12)完全相同。
​    (15) 按下`Ctrl + a`后手指松开，然后按`PageUp`：翻阅当前`pane`内的内容。
​    (16) 鼠标滚轮：翻阅当前`pane`内的内容。
​    (17) 在`tmux`中选中文本时，需要按住`shift`键。（仅支持`Windows和Linux`，不支持`Mac`，不过该操作并不是必须的，因此影响不大）
​    (18) `tmux`中复制/粘贴文本的通用方式：
​        (1) 按下`Ctrl + a`后松开手指，然后按[
​        (2) 用鼠标选中文本，被选中的文本会被自动复制到`tmux`的剪贴板
​        (3) 按下`Ctrl + a`后松开手指，然后按]，会将剪贴板中的内容粘贴到光标处



## vim相关操作

### 功能：

​    (1) 命令行模式下的文本编辑器。
​    (2) 根据文件扩展名自动判别编程语言。支持代码缩进、代码高亮等功能。
​    (3) 使用方式：`vim filename`
​        如果已有该文件，则打开它。
​        如果没有该文件，则打开个一个新的文件，并命名为`filename`

### 模式：

​    (1) 一般命令模式
​        默认模式。命令输入方式：类似于打游戏放技能，按不同字符，即可进行不同操作。可以复制、粘贴、删除文本等。
​    (2) 编辑模式
​        在一般命令模式里按下i，会进入编辑模式。
​        按下`ESC`会退出编辑模式，返回到一般命令模式。
​    (3) 命令行模式
​        在一般命令模式里按下`:/?`三个字母中的任意一个，会进入命令行模式。命令行在最下面。
​        可以查找、替换、保存、退出、配置编辑器等。

### 操作：

​    (1) `i`：进入编辑模式
​    (2) `ESC`：进入一般命令模式
​    (3) `h` 或 左箭头键：光标向左移动一个字符
​    (4) `j` 或 向下箭头：光标向下移动一个字符
​    (5) `k` 或 向上箭头：光标向上移动一个字符
​    (6) `l `或 向右箭头：光标向右移动一个字符
​    (7) `n<Space>`：n表示数字，按下数字后再按空格，光标会向右移动这一行的n个字符
​    (8) `0` 或 功能键`[Home]`：光标移动到本行开头
​    (9) `$` 或 功能键`[End]`：光标移动到本行末尾
​    (10)` G`：光标移动到最后一行
​    (11) :`n 或 nG`：n为数字，光标移动到第n行
​    (12) `gg`：光标移动到第一行，相当于1G
​    (13) `n<Enter>`：n为数字，光标向下移动n行
​    (14) `/word`：向光标之下寻找第一个值为word的字符串。
​    (15) `?word`：向光标之上寻找第一个值为word的字符串。
​    (16) `n`：重复前一个查找操作
​    (17) `N`：反向重复前一个查找操作
​    (18) `:n1,n2s/word1/word2/g`：n1与n2为数字，在第n1行与n2行之间寻找word1这个字符串，并将该字符串替换为word2
​    (19) :`1,$s/word1/word2/g`：将全文的word1替换为word2
​    (20) :`1,$s/word1/word2/gc`：将全文的word1替换为word2，且在替换前要求用户确认。
​    (21) `v`：选中文本
​    (22) `d`：删除选中的文本
​    (23) `dd`: 删除当前行
​    (24) `y`：复制选中的文本
​    (25) `yy`: 复制当前行
​    (26) `p`: 将复制的数据在光标的下一行/下一个位置粘贴
​    (27) `u`：撤销
​    (28) `Ctrl + r`：取消撤销
​    (29) 大于号` >`：将选中的文本整体向右缩进一次
​    (30) 小于号`<`：将选中的文本整体向左缩进一次
​    (31) `:w` 保存
​    (32) `:w!` 强制保存
​    (33) `:q` 退出
​    (34) `:q!` 强制退出
​    (35) `:wq` 保存并退出
​    (36) `:set paste` 设置成粘贴模式，取消代码自动缩进
​    (37) `:set nopaste` 取消粘贴模式，开启代码自动缩进
​    (38) `:set nu` 显示行号
​    (39) `:set nonu` 隐藏行号
​    (40) `gg=G`将全文代码格式化
​    (41) `:noh` 关闭查找关键词高亮
​    (42) `Ctrl + q`：当vim卡死时，可以取消当前正在执行的命令

### 异常处理：

​    每次用vim编辑文件时，会自动创建一个`.filename.swp`的临时文件。
​    如果打开某个文件时，该文件的`swp`文件已存在，则会报错。此时解决办法有两种：
​        (1) 找到正在打开该文件的程序，并退出
​        (2) 直接删掉该`swp`文件即可



## shell语法


#### 3.1 概论

`shell`是我们通过命令与操作系统沟通的语言

`shell`脚本可以直接在命令行中执行，也可以将一套逻辑组织成一个文件，方便复用

`Ac Terminal`中的命令行可以看成是一个`shell`脚本在逐行执行

`Linux`中常见的`shell`脚本有很多种，常见的有：

ㅤㅤ●ㅤ`Bourne Shell(/usr/bin/sh或/bin/sh)`

ㅤㅤ●ㅤ`Bourne Again Shell(/bin/bash)`

ㅤㅤ●ㅤ`C Shell(/usr/bin/csh)`

ㅤㅤ●ㅤ`K Shell(/usr/bin/ksh)`

ㅤㅤ●ㅤ`zsh`

ㅤㅤ●ㅤ`...`

`Linux`系统中一般默认使用`bash`，所以接下来讲解`bash`中的语法

文件开头需要写`#! /bin/bash`，指明`bash`为脚本解释器

##### 3.1.1 学习技巧

ㅤㅤ不要死记硬背，遇到含糊不清的地方，可以在终端里实际运行一遍。

##### 3.1.2 脚本示例

新建一个`test.sh`文件，内 c容如下：

```shell
#! /bin/bash

echo "Hello World!"
```

##### 3.1.3 运行方式

###### 作为可执行文件

```
chmod +x test.sh # 使脚本具有可执行权限

./test.sh # 当前路径下执行
Hello World! # 脚本输出

/home/acs/test.sh # 绝对路径下执行
Hello World! # 脚本输出

~/test.sh # 家目录路径下执行
Hello World! # 脚本输出
```

###### 用解释器执行

```shell
bash test.sh
Hello World! # 脚本输出
```

#### 3.2 注释

##### 3.2.1 单行注释

每行中#之后的内容均是注释

```shell
# 这是一行注释
echo "Hello World!" # 这也是注释
```

##### 3.2.2 多行注释

格式：

```shell
:<<EOF
注释1
注释2
注释3
EOF
其中EOF可以替换成其它任意字符串 例如：

:<<abc
注释4
注释5
注释6
abc

:<<!
注释7
注释8
注释9
!
```





#### 3.3 变量

##### 3.3.1 定义变量

定义变量，不需要加$符号， 例如：

```shell
name1='zst' # 单引号定义字符串
name2="zst" # 双引号定义字符串
name3=zst # 也可以不加引号， 同样表示字符串
```

##### 3.3.2 使用变量

使用变量，需要加上$符号，或者${}符号 花括号是可选的， 主要是为了帮助解释器识别变量边界

```shell
name=zst
echo $name # 输出 zst
echo ${name} # 输出 zst
echo ${name}acwing # 输出 zstacwing
```

##### 3.3.3 只读变量

使用`readonly`或者`declare`可以将变量变为只读

```shell
name=zst
readonly name
declare -r name # 两种写法均可

name=abc # 会报错，因为此时name是只读变量
```

##### 3.3.4 删除变量

unset可以删除变量

```shell
name=zst
unset name
echo $name # 输出空行
```

##### 3.3.5 变量类型

1. 自定义变量（局部变量）子进程不能访问的变量

2. 环境变量（全局变量）子进程可以访问的变量

###### 自定义变量改成环境变量：

```shell
name=zst # 定义自定义变量
export name # 第一种方法改环境变量
declare -x name # 第二种方法改环境变量
```

###### 环境变量改为自定义变量：

```shell
export name=zst # 定义环境变量
declare +x name # 改为自定义变量
```

目前tmux中就是一个bash

```
name=zst # 定义自定义变量
export name # 将自定变量修改为环境变量
# 或者是 declare -x name

bash # 进入一个新的子进程
echo $name # 输出 zst
exit # 退出当前子进程

declare +x name # 将环境变量修改为自定义变量
bash # 进入一个新的子进程
echo $name # 输出 空行
exit # 退出当前子进程
```

##### 3.3.6 字符串

字符串可以用单引号，也可以用双引号，也可以不用引号

单引号和双引号的区别：

单引号总的内容会原样输出，不会执行、不会取变量;

双引号中的内容可以用执行、可以取变量;

```shell
name=zst # 不用引号
echo 'hello， $name \"hh\"' # 单引号字符串，输出 hello， $name \"hh\"
echo "hello， $name \"hh\"" # 双引号字符串， 输出 hello， zst "hh"
```

获取字符串长度

```shell
name="zst"
echo ${#name} # 输出 3
```

提取子串

```shell
name="hello， zst"
echo ${name:0:5} # 提取从 0 开始的 5 个字符
```



## ssh相关操作

### 4.1 ssh 登录

#### 4.1.1 基本用法

远程登录服务器：

`ssh user@hostname`

ㅤㅤ●ㅤ`user`：用户名

ㅤㅤ●ㅤ`hostname`：IP地址或域名

第一次登录时会提示：

```markdown
The authenticity of host '123.57.47.211 (123.57.47.211)' can't be established.
ECDSA key fingerprint is SHA256:iy237yysfCe013/l+kpDGfEG9xxHxm0dnxnAbJTPpG8.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入`yes`，然后回车即可

这样会将该服务器的信息记录在`~/.ssh/known_hosts`文件中

然后输入密码即可登录到远程服务器中

`logout`：退出当前服务器

默认登录端口号为22，如果想登录某一特定端口：

`ssh user@hostname -p 22`

#### 4.1.2 配置文件

创建文件`~/.ssh/config`

然后在文件中输入：

~~~
Host myserver1
    HostName IP地址或域名
    User 用户名

Host myserver2
    HostName IP地址或域名
    User 用户名
~~~

之后再使用服务器时，可以直接使用别名`myserver1`、`myserver2`

#### 4.1.3 密钥登录

创建密钥：

`ssh-keygen`

然后一直回车即可

执行结束后，`~/.ssh/`目录下会多两个文件：

ㅤㅤ●ㅤ`id_rsa`：私钥

ㅤㅤ●ㅤ`id_rsa.pub`：公钥

ㅤㅤ之后想免密码登录哪个服务器，就将公钥传给哪个服务器即可。

ㅤㅤ例如，想免密登录`myserver`服务器。则将公钥中的内容，复制到`myserver`中的`~/.ssh/authorized_keys`文件里即可。

ㅤㅤ也可以使用如下命令一键添加公钥：

ㅤㅤ`ssh-copy-id myserver`

#### 4.1.4 执行命令

命令格式：

`ssh user@hostname command`

例如：

`ssh user@hostname ls -a`

或者

~~~
# 单引号中的$i可以求值
ssh myserver 'for ((i = 0; i < 10; i ++ )) do echo $i; done'
~~~

或者

```
# 双引号中的$i不可以求值
ssh myserver "for ((i = 0; i < 10; i ++ )) do echo $i; done"
```

### 4.2 $scp$ 传文件

#### 4.2.1 基本用法

##### 命令格式：

`scp source destination`

将`source`路径下的文件复制到`destination`中

一次复制多个文件：

`scp source1 source2 destination`

##### 复制文件夹：

`scp -r ~/tmp myserver:/home/acs/`

将本地家目录中的`tmp`文件夹复制到`myserver`服务器中的`/home/acs/`目录下。

`scp -r ~/tmp myserver:homework/`

将本地家目录中的`tmp`文件夹复制到`myserver`服务器中的`~/homework/`目录下。

`scp -r myserver:homework` .

将`myserver`服务器中的`~/homework/`文件夹复制到本地的当前路径下。

##### 指定服务器的端口号：

`scp -P 22 source1 source2 destination`

注意： `scp`的`-r -P`等参数尽量加在`source`和`destination`之前。

使用`scp`配置其他服务器的`vim`和`tmux`

`scp ~/.vimrc ~/.tmux.conf myserver:`

(将$Acwing$本地的$tmux$和$vim$配置安装到自己的服务器上，可以使自己的服务器支持鼠标，操作也更加方便)

## $Git$相关操作

[$Acwing$代码托管平台](https://git.acwing.com/)(基于$GitLab$)

### 1.1 $Git$基本概念

- 工作区：仓库的目录。工作区是独立于各个分支的。
- 暂存区：数据暂时存放的区域，类似于工作区写入版本库前的缓存区。暂存区是独立于各个分支的。
- 版本库：存放所有已经提交到本地仓库的代码版本
- 版本结构：树结构，树中每个节点代表一个代码版本。

### 1.2 $Git$常用命令

#### 1.2.1 全局设置

`git config --global user.name xxx`：设置全局用户名，信息记录在`~/.gitconfig`文件中

`git config --global user.email xxx@xxx.com`：设置全局邮箱地址，信息记录在`~/.gitconfig`文件中

`git init`：将当前目录配置成git仓库，信息记录在隐藏的`.git`文件夹中

####  1.2.2 常用命令

`git add XX` ：将XX文件添加到暂存区

`git add .`:将所有修改的文件加入暂存区

`git commit -m` "给自己看的备注信息"：将暂存区的内容提交到当前分支

`git status`：查看仓库状态

`git log`：查看当前分支的所有版本

`git push -u` (第一次需要`-u`以后不需要) ：将当前分支推送到远程仓库

`git clone git@git.acwing.com:xxx/XXX.git`：将远程仓库`XXX`下载到当前目录下

`git branch`：查看所有分支和当前所处分支

#### 1.2.3 查看命令

`git diff XX`：查看XX文件相对于暂存区修改了哪些内容
`git status`：查看仓库状态
`git log`：查看当前分支的所有版本
`git log --pretty=oneline`：用一行来显示
`git reflog`：查看HEAD指针的移动历史（包括被回滚的版本）
`git branch`：查看所有分支和当前所处分支
`git pull`：将远程仓库的当前分支与本地仓库的当前分支合并

#### 1.2.4 删除命令

`git rm --cached XX`：将文件从仓库索引目录中删掉，不希望管理这个文件
`git restore --staged xx`：==将xx从暂存区里移除==,并移动到工作区
`git checkout — XX或git restore XX`：==将XX文件尚未加入暂存区的修改全部撤销==

`git restore`:不指明任何文件，则直接从工作区最新修改回滚到暂存区的版本，如果暂存区，没有东西，则回滚到$HEAD$指向的这个版本

#### 1.2.5 代码回滚

`git reset --hard HEAD^ 或git reset --hard HEAD~` ：将代码库回滚到上一个版本
`git reset --hard HEAD^^`：往上回滚两次，以此类推
`git reset --hard HEAD~100`：往上回滚100个版本

`git reset --hard 版本号`：回滚到某一特定版本

#### 1.2.6 远程仓库

`git remote add origin git@git.acwing.com:xxx/XXX.git`：将本地仓库关联到远程仓库
`git push -u` (第一次需要-u以后不需要) ：将当前分支推送到远程仓库
`git push origin branch_name`：将本地的某个分支推送到远程仓库
`git clone git@git.acwing.com:xxx/XXX.git`：将远程仓库`XXX`下载到当前目录下
`git push --set-upstream origin branch_name`：设置本地的`branch_name`分支对应远程仓库的`branch_name`分支
`git push -d origin branch_name`：删除远程仓库的`branch_name`分支
`git checkout -t origin/branch_name` 将远程的`branch_name`分支拉取到本地
`git pull` ：将远程仓库的当前分支与本地仓库的当前分支合并
`git pull origin branch_name`：将远程仓库的`branch_name`分支与本地仓库的当前分支合并
`git branch --set-upstream-to=origin/branch_name1 branch_name2`：将远程的`branch_name1`分支与本地的`branch_name2`分支对应

#### 1.2.7 分支命令

`git branch branch_name`：创建新分支
`git branch`：查看所有分支和当前所处分支
`git checkout -b branch_name`：创建并切换到branch_name这个分支
`git checkout branch_name`：切换到branch_name这个分支
`git merge branch_name`：将分支branch_name合并到当前分支上
`git branch -d branch_name`：删除本地仓库的branch_name分支
`git push --set-upstream origin branch_name`：设置本地的branch_name分支对应远程仓库的branch_name分支
`git push -d origin branch_name`：删除远程仓库的branch_name分支
`git checkout -t origin/branch_name` 将远程的branch_name分支拉取到本地
`git pull` ：将远程仓库的当前分支与本地仓库的当前分支合并
`git pull origin branch_name`：将远程仓库的branch_name分支与本地仓库的当前分支合并
`git branch --set-upstream-to=origin/branch_name1 branch_name2`：将远程的`branch_name1`分支与本地的`branch_name2`分支对应

#### 1.2.8 stash暂存

`git stash`：将工作区和暂存区中尚未提交的修改存入栈中
`git stash apply`：将栈顶存储的修改恢复到当前分支，但不删除栈顶元素
`git stash drop`：删除栈顶存储的修改
`git stash pop`：将栈顶存储的修改恢复到当前分支，同时删除栈顶元素
`git stash list`：查看栈中所有元素

## Thrift

$thrift$作用:服务器之间的交互，不同服务器之间可以使用不同语言，根据对应的结构定义接口，提供$server$，以及请求$(client)$,抽象的说，在两个服务器之间建立一条有向边，$Thrift$也被称为$RPC$框架。

### 具体使用

参照[官方文档](https://thrift.apache.org/)

## 管道

### 概念

管道类似于文件重定向（但是和文件重定向不一样），可以将前一个命令的`stdout`重定向到下一个命令的`stdin`。

### 要点

管道命令仅处理`stdout`，会忽略`stderr`。
管道右边的命令必须能接受`stdin`。
多个管道命令可以串联。
与文件重定向的区别
文件重定向左边为命令，右边为文件。
管道左右两边均为命令，左边有`stdout`，右边有`stdin`。

### 举例

统计当前目录下所有`python`文件的总行数，其中`find`、`xargs`、`wc`等命令可以参考常用命令这一节内容。

`find . -name '*.py' | xargs cat | wc -l`

## 环境变量

### 概念

$Linux$系统中会用很多环境变量来记录配置信息。
环境变量类似于全局变量，可以被各个进程访问到。我们可以通过修改环境变量来方便地修改系统配置。

### 查看

列出当前环境下的所有环境变量：

`env`  # 显示当前用户的变量
`set`  # 显示当前shell的变量，包括当前用户的变量;
`export`  # 显示当前导出成用户变量的shell变量
输出某个环境变量的值：

`echo $PATH`

### 修改

环境变量的定义、修改、删除操作可以参考`3. shell语法——变量`这一节的内容。

为了将对环境变量的修改应用到未来所有环境下，可以将修改命令放到`~/.bashrc`文件中。
修改完`~/.bashrc`文件后，记得执行`source ~/.bashrc`，来将修改应用到当前的bash环境下。

为何将修改命令放到`~/.bashrc`，就可以确保修改会影响未来所有的环境呢？

每次启动`bash`，都会先执行`~/.bashrc`。
每次`ssh`登陆远程服务器，都会启动一个`bash`命令行给我们。
每次`tmux`新开一个`pane`，都会启动一个`bash`命令行给我们。
所以未来所有新开的环境都会加载我们修改的内容。

### 常见环境变量

`HOME`：用户的家目录。
`PATH`：可执行文件（命令）的存储路径。路径与路径之间用:分隔。当某个可执行文件同时出现在多个路径中时，会选择从左到右数第一个路径中的执行。下列所有存储路径的环境变量，均采用从左到右的优先顺序。
`LD_LIBRARY_PATH`：用于指定动态链接库(.so文件)的路径，其内容是以冒号分隔的路径列表。
`C_INCLUDE_PATH`：C语言的头文件路径，内容是以冒号分隔的路径列表。
`CPLUS_INCLUDE_PATH`：CPP的头文件路径，内容是以冒号分隔的路径列表。
`PYTHONPATH`：Python导入包的路径，内容是以冒号分隔的路径列表。
`JAVA_HOME`：jdk的安装目录。
`CLASSPATH`：存放Java导入类的路径，内容是以冒号分隔的路径列表。

## 常用命令

### 系统状况

`top`：查看所有进程的信息（Linux的任务管理器）
打开后，输入`M`：按使用内存排序
打开后，输入`P`：按使用CPU排序
打开后，输入`q`：退出
`df -h`：查看硬盘使用情况
`free -h`：查看内存使用情况
`du -sh`：查看当前目录占用的硬盘空间
`ps aux`：查看所有进程
`kill -9 pid`：杀死编号为`pid`的进程
传递某个具体的信号：`kill -s SIGTERM pid`
`netstat -nt`：查看所有网络连接
`w`：列出当前登陆的用户
`ping www.baidu.com`：检查是否连网

### 文件权限

`chmod`：修改文件权限
`chmod +x xxx`：给`xxx`添加可执行权限
`chmod -x xxx`：去掉`xxx`的可执行权限
`chmod 777 xxx`：将`xxx`的权限改成777
`chmod 777 xxx -R`：递归修改整个文件夹的权限

### 文件检索

`find /path/to/directory/ -name '*.py'`：搜索某个文件路径下的所有`*.py`文件
`grep xxx`：从`stdin`中读入若干行数据，如果某行中包含`xxx`，则输出该行；否则忽略该行。
`wc`：统计行数、单词数、字节数
既可以从`stdin`中直接读入内容；也可以在命令行参数中传入文件名列表；
`wc -l`：统计行数
`wc -w`：统计单词数
`wc -c`：统计字节数
`tree`：展示当前目录的文件结构
`tree /path/to/directory/`：展示某个目录的文件结构
`tree -a`：展示隐藏文件
`ag xxx`：搜索当前目录下的所有文件，检索`xxx`字符串
`cut`：分割一行内容
从`stdin`中读入多行数据
`echo $PATH | cut -d ':' -f 3,5`：输出`PATH`用:分割后第`3、5`列数据
`echo $PATH | cut -d ':' -f 3-5`：输出`PATH`用:分割后第`3-5`列数据
`echo $PATH | cut -c 3,5`：输出`PATH`的第`3、5`个字符

`echo $PATH | cut -c 3-5`：输出`PATH`的第`3-5`个字符
`sort`：将每行内容按字典序排序
可以从`stdin`中读取多行数据
可以从命令行参数中读取文件名列表
`xargs`：将`stdin`中的数据用空格或回车分割成命令行参数
`find . -name '*.py' | xargs cat | wc -l`：统计当前目录下所有`python`文件的总行数

### 查看文件内容(不如Vim)

`more`：浏览文件内容
回车：下一行
空格：下一页
`b`：上一页
`q`：退出
`less`：与`more`类似，功能更全
回车：下一行
`y`：上一行
`Page Down`：下一页
`Page Up`：上一页
`q`：退出
`head -3 xxx`：展示`xxx`的前3行内容
同时支持从`stdin`读入内容
`tail -3 xxx`：展示`xxx`末尾3行内容
同时支持从`stdin`读入内容

### 用户相关

`history`：展示当前用户的历史操作。内容存放在`~/.bash_history`中

### 工具

`md5sum`：计算`md5`哈希值
可以从`stdin`读入内容
也可以在命令行参数中传入文件名列表；
`time command`：统计`command`命令的执行时间
`ipython3`：交互式`python3`环境。可以当做计算器，或者批量管理文件。
`! echo "Hello World"`：!表示执行shell脚本
`watch -n 0.1 command`：每0.1秒执行一次`command`命令
`tar`：压缩文件
`tar -zcvf xxx.tar.gz /path/to/file/*`：压缩
`tar -zxvf xxx.tar.gz`：解压缩
`diff xxx yyy`：查找文件`xxx`与`yyy`的不同点

### 安装软件

`sudo command`：以`root`身份执行`command`命令
`apt-get install xxx`：安装软件
`pip install xxx --user --upgrade`：安装`python`包

