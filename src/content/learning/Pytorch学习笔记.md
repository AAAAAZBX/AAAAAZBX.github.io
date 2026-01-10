---
title: "Pytorch入门"
id: "2025-12-06-01"
date: "2025-12-06"
description: "Pytorch入门学习笔记，包括开发环境配置、基础语法和常用操作"
tags: ["pytorch"]
---

# 一、Pycharm 及 Jupyter notebook使用及对比

例如以下几行Python代码:

~~~python
print("Start!")
a="hello world"
b=2019
c=a+b
print(c)
~~~

- 如果代码是以块为一个整体运行的话，那么，**Python文件**的块是所有行的代码。

**优点**：通用，传播方便，适用于大型项目。

**缺点**：需要从头运行。

- 在**Python控制台**中，代码以行为块运行，（按shift+enter也可以多行运行），一般用于单行调试。

**优点**：可以显示每个变量属性。

**缺点**：不利于代码阅读及修改。

- 在**Jupyter notebook**中可以任意切分代码块。

**优点**：利于代码的阅读和修改。

**缺点**：环境需要再使用前配置。


# 二、使用Python加载数据

## 1.数据相关类

### （1）Dataset

介绍：如果说data是一堆垃圾，那么dataset就提供了一种获取数据及其label的方法。

功能：如何获取一个数据及其label,并且告诉我们共有多少个数据

### （2）Dataloader

为后面的网络提供不同的数据形式

## 2.Dataset类代码实战

### （1）工具包——Dataset介绍


```python
from torch.utils.data import Dataset
```


```python
help(Dataset)
```

    Help on class Dataset in module torch.utils.data.dataset:
    
    class Dataset(typing.Generic)
     |  An abstract class representing a :class:`Dataset`.
     |  
     |  All datasets that represent a map from keys to data samples should subclass
     |  it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
     |  data sample for a given key. Subclasses could also optionally overwrite
     |  :meth:`__len__`, which is expected to return the size of the dataset by many
     |  :class:`~torch.utils.data.Sampler` implementations and the default options
     |  of :class:`~torch.utils.data.DataLoader`.
     |  
     |  .. note::
     |    :class:`~torch.utils.data.DataLoader` by default constructs a index
     |    sampler that yields integral indices.  To make it work with a map-style
     |    dataset with non-integral indices/keys, a custom sampler must be provided.
     |  
     |  Method resolution order:
     |      Dataset
     |      typing.Generic
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]'
     |  
     |  __getitem__(self, index) -> +T_co
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __orig_bases__ = (typing.Generic[+T_co],)
     |  
     |  __parameters__ = (+T_co,)
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from typing.Generic:
     |  
     |  __class_getitem__(params) from builtins.type
     |  
     |  __init_subclass__(*args, **kwargs) from builtins.type
     |      This method is called when a class is subclassed.
     |      
     |      The default implementation does nothing. It may be
     |      overridden to extend subclasses.


​    


```python
Dataset??
```

### （2）数据的读取


```python
from PIL import Image
img_path = "D:\\Micro_Climate_Summer_Task\\Pytorch_For_Deep_Learning\\dataset\\train\\daisy\\5547758_eea9edfd54_n.jpg"
img = Image.open(img_path)
img.size
```




    (320, 232)




```python
print(img)
```

    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=320x232 at 0x2AC44FF85B0>



```python
img.show()
```

- 控制台中执行下面几行代码会使其中一个文件变为一个list
~~~python
dir_path = "dataset/train/daisy"
import os
img_path_list = os.listdir(dir_path)
~~~
这时候img_path_list[0]就是第一个图片的名称，img_path_list[1]就是第二个图片的名称


```python
<img src="http://localhost:8888/view/Desktop/Pytorch_For_Deep_Learning/dir_to_list.png", width=320>
```


      File "C:\Users\LENOVO\AppData\Local\Temp\ipykernel_13452\4213775797.py", line 1
        <img src="http://localhost:8888/view/Desktop/Pytorch_For_Deep_Learning/dir_to_list.png", width=320>
        ^
    SyntaxError: invalid syntax



- 控制台中执行下面的命令会得到其中一个数据的地址及标签
~~~python
root_dir = "dataset/train"
label_dir = "daisy"
path = os.path.join(root_dir , label_dir)
img_path = os.listdir(path)
~~~
这时候img_path也变成了一个地址数组，比上面的数组多了一个标签

### （3）将数据打包成数据集

- 下面的代码演示了自定义数据集的几个重要函数，初始化（将标签与图片地址合并），查找，询问长度，并且以daisy为例得到了daisy的训练集

~~~python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self ,root_dir, label_dir): #root_dir即为根目录，label_dir即为标签
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir , self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self ,idx): #使用listdir获得文件列表后，函数返回第i张图片相关信息
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
daisy_label_dir = "daisy"
daisy_dataset = MyData(root_dir, daisy_label_dir)

~~~

- 运行完上面的代码后，输入daisy_data[0]，便会返回数据信息及其标签，如下所示
~~~
Out[3]: (<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=320x263>, 'daisy')
~~~

- 通过下面的代码可以查看数据集中对应位置的图片
~~~python
img, label=daisy_dataset[idx]
img.show()
~~~

- 由此，可以自己将五个数据集合并得到训练集，总代码如下：

~~~python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self ,root_dir, label_dir): #root_dir即为根目录，label_dir即为标签
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir , self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self ,idx): #使用listdir获得文件列表后，函数返回第i张图片相关信息
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
daisy_label_dir = "daisy"
dandelion_label_dir = "dandelion"
roses_label_dir = "roses"
sunflowers_label_dir = "sunflowers"
tulips_label_dir = "tulips"

daisy_dataset = MyData(root_dir, daisy_label_dir)
dandelion_dataset = MyData(root_dir, dandelion_label_dir)
roses_dataset = MyData(root_dir, roses_label_dir)
sunflowers_dataset = MyData(root_dir, sunflowers_label_dir)
tulips_dataset = MyData(root_dir, tulips_label_dir)

train_dataset = daisy_dataset + dandelion_dataset + roses_dataset + sunflowers_dataset + tulips_dataset


~~~

- 经过检验，训练集中的图片数量正好为五个训练集数据量之和

# 三、数据可视化工具——Tensorboard的使用

## 1.SummaryWriter的使用


```python
from torch.utils.tensorboard import SummaryWriter
```


```python
help(SummaryWriter)
```

    Help on class SummaryWriter in module torch.utils.tensorboard.writer:
    
    class SummaryWriter(builtins.object)
     |  SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')
     |  
     |  Writes entries directly to event files in the log_dir to be
     |  consumed by TensorBoard.
     |  
     |  The `SummaryWriter` class provides a high-level API to create an event file
     |  in a given directory and add summaries and events to it. The class updates the
     |  file contents asynchronously. This allows a training program to call methods
     |  to add data to the file directly from the training loop, without slowing down
     |  training.
     |  
     |  Methods defined here:
     |  
     |  __enter__(self)
     |  
     |  __exit__(self, exc_type, exc_val, exc_tb)
     |  
     |  __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')
     |      Creates a `SummaryWriter` that will write out events and summaries
     |      to the event file.
     |      
     |      Args:
     |          log_dir (string): Save directory location. Default is
     |            runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
     |            Use hierarchical folder structure to compare
     |            between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
     |            for each new experiment to compare across them.
     |          comment (string): Comment log_dir suffix appended to the default
     |            ``log_dir``. If ``log_dir`` is assigned, this argument has no effect.
     |          purge_step (int):
     |            When logging crashes at step :math:`T+X` and restarts at step :math:`T`,
     |            any events whose global_step larger or equal to :math:`T` will be
     |            purged and hidden from TensorBoard.
     |            Note that crashed and resumed experiments should have the same ``log_dir``.
     |          max_queue (int): Size of the queue for pending events and
     |            summaries before one of the 'add' calls forces a flush to disk.
     |            Default is ten items.
     |          flush_secs (int): How often, in seconds, to flush the
     |            pending events and summaries to disk. Default is every two minutes.
     |          filename_suffix (string): Suffix added to all event filenames in
     |            the log_dir directory. More details on filename construction in
     |            tensorboard.summary.writer.event_file_writer.EventFileWriter.
     |      
     |      Examples::
     |      
     |          from torch.utils.tensorboard import SummaryWriter
     |      
     |          # create a summary writer with automatically generated folder name.
     |          writer = SummaryWriter()
     |          # folder location: runs/May04_22-14-54_s-MacBook-Pro.local/
     |      
     |          # create a summary writer using the specified folder name.
     |          writer = SummaryWriter("my_experiment")
     |          # folder location: my_experiment
     |      
     |          # create a summary writer with comment appended.
     |          writer = SummaryWriter(comment="LR_0.1_BATCH_16")
     |          # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/
     |  
     |  add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None)
     |      Add audio data to summary.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          snd_tensor (torch.Tensor): Sound data
     |          global_step (int): Global step value to record
     |          sample_rate (int): sample rate in Hz
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |      Shape:
     |          snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
     |  
     |  add_custom_scalars(self, layout)
     |      Create special chart by collecting charts tags in 'scalars'. Note that this function can only be called once
     |      for each SummaryWriter() object. Because it only provides metadata to tensorboard, the function can be called
     |      before or after the training loop.
     |      
     |      Args:
     |          layout (dict): {categoryName: *charts*}, where *charts* is also a dictionary
     |            {chartName: *ListOfProperties*}. The first element in *ListOfProperties* is the chart's type
     |            (one of **Multiline** or **Margin**) and the second element should be a list containing the tags
     |            you have used in add_scalar function, which will be collected into the new chart.
     |      
     |      Examples::
     |      
     |          layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
     |                       'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
     |                            'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}
     |      
     |          writer.add_custom_scalars(layout)
     |  
     |  add_custom_scalars_marginchart(self, tags, category='default', title='untitled')
     |      Shorthand for creating marginchart. Similar to ``add_custom_scalars()``, but the only necessary argument
     |      is *tags*, which should have exactly 3 elements.
     |      
     |      Args:
     |          tags (list): list of tags that have been used in ``add_scalar()``
     |      
     |      Examples::
     |      
     |          writer.add_custom_scalars_marginchart(['twse/0050', 'twse/2330', 'twse/2006'])
     |  
     |  add_custom_scalars_multilinechart(self, tags, category='default', title='untitled')
     |      Shorthand for creating multilinechart. Similar to ``add_custom_scalars()``, but the only necessary argument
     |      is *tags*.
     |      
     |      Args:
     |          tags (list): list of tags that have been used in ``add_scalar()``
     |      
     |      Examples::
     |      
     |          writer.add_custom_scalars_multilinechart(['twse/0050', 'twse/2330'])
     |  
     |  add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)
     |      Add embedding projector data to summary.
     |      
     |      Args:
     |          mat (torch.Tensor or numpy.array): A matrix which each row is the feature vector of the data point
     |          metadata (list): A list of labels, each element will be convert to string
     |          label_img (torch.Tensor): Images correspond to each data point
     |          global_step (int): Global step value to record
     |          tag (string): Name for the embedding
     |      Shape:
     |          mat: :math:`(N, D)`, where N is number of data and D is feature dimension
     |      
     |          label_img: :math:`(N, C, H, W)`
     |      
     |      Examples::
     |      
     |          import keyword
     |          import torch
     |          meta = []
     |          while len(meta)<100:
     |              meta = meta+keyword.kwlist # get some strings
     |          meta = meta[:100]
     |      
     |          for i, v in enumerate(meta):
     |              meta[i] = v+str(i)
     |      
     |          label_img = torch.rand(100, 3, 10, 32)
     |          for i in range(100):
     |              label_img[i]*=i/100.0
     |      
     |          writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
     |          writer.add_embedding(torch.randn(100, 5), label_img=label_img)
     |          writer.add_embedding(torch.randn(100, 5), metadata=meta)
     |  
     |  add_figure(self, tag, figure, global_step=None, close=True, walltime=None)
     |      Render matplotlib figure into an image and add it to summary.
     |      
     |      Note that this requires the ``matplotlib`` package.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          figure (matplotlib.pyplot.figure) or list of figures: Figure or a list of figures
     |          global_step (int): Global step value to record
     |          close (bool): Flag to automatically close the figure
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |  
     |  add_graph(self, model, input_to_model=None, verbose=False, use_strict_trace=True)
     |      Add graph data to summary.
     |      
     |      Args:
     |          model (torch.nn.Module): Model to draw.
     |          input_to_model (torch.Tensor or list of torch.Tensor): A variable or a tuple of
     |              variables to be fed.
     |          verbose (bool): Whether to print graph structure in console.
     |          use_strict_trace (bool): Whether to pass keyword argument `strict` to
     |              `torch.jit.trace`. Pass False when you want the tracer to
     |              record your mutable container types (list, dict)
     |  
     |  add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)
     |      Add histogram to summary.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          values (torch.Tensor, numpy.array, or string/blobname): Values to build histogram
     |          global_step (int): Global step value to record
     |          bins (string): One of {'tensorflow','auto', 'fd', ...}. This determines how the bins are made. You can find
     |            other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |      
     |      Examples::
     |      
     |          from torch.utils.tensorboard import SummaryWriter
     |          import numpy as np
     |          writer = SummaryWriter()
     |          for i in range(10):
     |              x = np.random.random(1000)
     |              writer.add_histogram('distribution centers', x + i, i)
     |          writer.close()
     |      
     |      Expected result:
     |      
     |      .. image:: _static/img/tensorboard/add_histogram.png
     |         :scale: 50 %
     |  
     |  add_histogram_raw(self, tag, min, max, num, sum, sum_squares, bucket_limits, bucket_counts, global_step=None, walltime=None)
     |      Adds histogram with raw data.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          min (float or int): Min value
     |          max (float or int): Max value
     |          num (int): Number of values
     |          sum (float or int): Sum of all values
     |          sum_squares (float or int): Sum of squares for all values
     |          bucket_limits (torch.Tensor, numpy.array): Upper value per bucket.
     |            The number of elements of it should be the same as `bucket_counts`.
     |          bucket_counts (torch.Tensor, numpy.array): Number of values per bucket
     |          global_step (int): Global step value to record
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |          see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/histogram/README.md
     |      
     |      Examples::
     |      
     |          from torch.utils.tensorboard import SummaryWriter
     |          import numpy as np
     |          writer = SummaryWriter()
     |          dummy_data = []
     |          for idx, value in enumerate(range(50)):
     |              dummy_data += [idx + 0.001] * value
     |      
     |          bins = list(range(50+2))
     |          bins = np.array(bins)
     |          values = np.array(dummy_data).astype(float).reshape(-1)
     |          counts, limits = np.histogram(values, bins=bins)
     |          sum_sq = values.dot(values)
     |          writer.add_histogram_raw(
     |              tag='histogram_with_raw_data',
     |              min=values.min(),
     |              max=values.max(),
     |              num=len(values),
     |              sum=values.sum(),
     |              sum_squares=sum_sq,
     |              bucket_limits=limits[1:].tolist(),
     |              bucket_counts=counts.tolist(),
     |              global_step=0)
     |          writer.close()
     |      
     |      Expected result:
     |      
     |      .. image:: _static/img/tensorboard/add_histogram_raw.png
     |         :scale: 50 %
     |  
     |  add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)
     |      Add a set of hyperparameters to be compared in TensorBoard.
     |      
     |      Args:
     |          hparam_dict (dict): Each key-value pair in the dictionary is the
     |            name of the hyper parameter and it's corresponding value.
     |            The type of the value can be one of `bool`, `string`, `float`,
     |            `int`, or `None`.
     |          metric_dict (dict): Each key-value pair in the dictionary is the
     |            name of the metric and it's corresponding value. Note that the key used
     |            here should be unique in the tensorboard record. Otherwise the value
     |            you added by ``add_scalar`` will be displayed in hparam plugin. In most
     |            cases, this is unwanted.
     |          hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
     |            contains names of the hyperparameters and all discrete values they can hold
     |          run_name (str): Name of the run, to be included as part of the logdir.
     |            If unspecified, will use current timestamp.
     |      
     |      Examples::
     |      
     |          from torch.utils.tensorboard import SummaryWriter
     |          with SummaryWriter() as w:
     |              for i in range(5):
     |                  w.add_hparams({'lr': 0.1*i, 'bsize': i},
     |                                {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
     |      
     |      Expected result:
     |      
     |      .. image:: _static/img/tensorboard/add_hparam.png
     |         :scale: 50 %
     |  
     |  add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
     |      Add image data to summary.
     |      
     |      Note that this requires the ``pillow`` package.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
     |          global_step (int): Global step value to record
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |          dataformats (string): Image data format specification of the form
     |            CHW, HWC, HW, WH, etc.
     |      Shape:
     |          img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
     |          convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
     |          Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
     |          corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
     |      
     |      Examples::
     |      
     |          from torch.utils.tensorboard import SummaryWriter
     |          import numpy as np
     |          img = np.zeros((3, 100, 100))
     |          img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
     |          img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
     |      
     |          img_HWC = np.zeros((100, 100, 3))
     |          img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
     |          img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
     |      
     |          writer = SummaryWriter()
     |          writer.add_image('my_image', img, 0)
     |      
     |          # If you have non-default dimension setting, set the dataformats argument.
     |          writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
     |          writer.close()
     |      
     |      Expected result:
     |      
     |      .. image:: _static/img/tensorboard/add_image.png
     |         :scale: 50 %
     |  
     |  add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None, walltime=None, rescale=1, dataformats='CHW', labels=None)
     |      Add image and draw bounding boxes on the image.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
     |          box_tensor (torch.Tensor, numpy.array, or string/blobname): Box data (for detected objects)
     |            box should be represented as [x1, y1, x2, y2].
     |          global_step (int): Global step value to record
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |          rescale (float): Optional scale override
     |          dataformats (string): Image data format specification of the form
     |            NCHW, NHWC, CHW, HWC, HW, WH, etc.
     |          labels (list of string): The label to be shown for each bounding box.
     |      Shape:
     |          img_tensor: Default is :math:`(3, H, W)`. It can be specified with ``dataformats`` argument.
     |          e.g. CHW or HWC
     |      
     |          box_tensor: (torch.Tensor, numpy.array, or string/blobname): NX4,  where N is the number of
     |          boxes and each 4 elements in a row represents (xmin, ymin, xmax, ymax).
     |  
     |  add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
     |      Add batched image data to summary.
     |      
     |      Note that this requires the ``pillow`` package.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
     |          global_step (int): Global step value to record
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |          dataformats (string): Image data format specification of the form
     |            NCHW, NHWC, CHW, HWC, HW, WH, etc.
     |      Shape:
     |          img_tensor: Default is :math:`(N, 3, H, W)`. If ``dataformats`` is specified, other shape will be
     |          accepted. e.g. NCHW or NHWC.
     |      
     |      Examples::
     |      
     |          from torch.utils.tensorboard import SummaryWriter
     |          import numpy as np
     |      
     |          img_batch = np.zeros((16, 3, 100, 100))
     |          for i in range(16):
     |              img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
     |              img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i
     |      
     |          writer = SummaryWriter()
     |          writer.add_images('my_image_batch', img_batch, 0)
     |          writer.close()
     |      
     |      Expected result:
     |      
     |      .. image:: _static/img/tensorboard/add_images.png
     |         :scale: 30 %
     |  
     |  add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None)
     |      Add meshes or 3D point clouds to TensorBoard. The visualization is based on Three.js,
     |      so it allows users to interact with the rendered object. Besides the basic definitions
     |      such as vertices, faces, users can further provide camera parameter, lighting condition, etc.
     |      Please check https://threejs.org/docs/index.html#manual/en/introduction/Creating-a-scene for
     |      advanced usage.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          vertices (torch.Tensor): List of the 3D coordinates of vertices.
     |          colors (torch.Tensor): Colors for each vertex
     |          faces (torch.Tensor): Indices of vertices within each triangle. (Optional)
     |          config_dict: Dictionary with ThreeJS classes names and configuration.
     |          global_step (int): Global step value to record
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |      
     |      Shape:
     |          vertices: :math:`(B, N, 3)`. (batch, number_of_vertices, channels)
     |      
     |          colors: :math:`(B, N, 3)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.
     |      
     |          faces: :math:`(B, N, 3)`. The values should lie in [0, number_of_vertices] for type `uint8`.
     |      
     |      Examples::
     |      
     |          from torch.utils.tensorboard import SummaryWriter
     |          vertices_tensor = torch.as_tensor([
     |              [1, 1, 1],
     |              [-1, -1, 1],
     |              [1, -1, -1],
     |              [-1, 1, -1],
     |          ], dtype=torch.float).unsqueeze(0)
     |          colors_tensor = torch.as_tensor([
     |              [255, 0, 0],
     |              [0, 255, 0],
     |              [0, 0, 255],
     |              [255, 0, 255],
     |          ], dtype=torch.int).unsqueeze(0)
     |          faces_tensor = torch.as_tensor([
     |              [0, 2, 3],
     |              [0, 3, 1],
     |              [0, 1, 2],
     |              [1, 3, 2],
     |          ], dtype=torch.int).unsqueeze(0)
     |      
     |          writer = SummaryWriter()
     |          writer.add_mesh('my_mesh', vertices=vertices_tensor, colors=colors_tensor, faces=faces_tensor)
     |      
     |          writer.close()
     |  
     |  add_onnx_graph(self, prototxt)
     |  
     |  add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None)
     |      Adds precision recall curve.
     |      Plotting a precision-recall curve lets you understand your model's
     |      performance under different threshold settings. With this function,
     |      you provide the ground truth labeling (T/F) and prediction confidence
     |      (usually the output of your model) for each target. The TensorBoard UI
     |      will let you choose the threshold interactively.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          labels (torch.Tensor, numpy.array, or string/blobname):
     |            Ground truth data. Binary label for each element.
     |          predictions (torch.Tensor, numpy.array, or string/blobname):
     |            The probability that an element be classified as true.
     |            Value should be in [0, 1]
     |          global_step (int): Global step value to record
     |          num_thresholds (int): Number of thresholds used to draw the curve.
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |      
     |      Examples::
     |      
     |          from torch.utils.tensorboard import SummaryWriter
     |          import numpy as np
     |          labels = np.random.randint(2, size=100)  # binary label
     |          predictions = np.random.rand(100)
     |          writer = SummaryWriter()
     |          writer.add_pr_curve('pr_curve', labels, predictions, 0)
     |          writer.close()
     |  
     |  add_pr_curve_raw(self, tag, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, global_step=None, num_thresholds=127, weights=None, walltime=None)
     |      Adds precision recall curve with raw data.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          true_positive_counts (torch.Tensor, numpy.array, or string/blobname): true positive counts
     |          false_positive_counts (torch.Tensor, numpy.array, or string/blobname): false positive counts
     |          true_negative_counts (torch.Tensor, numpy.array, or string/blobname): true negative counts
     |          false_negative_counts (torch.Tensor, numpy.array, or string/blobname): false negative counts
     |          precision (torch.Tensor, numpy.array, or string/blobname): precision
     |          recall (torch.Tensor, numpy.array, or string/blobname): recall
     |          global_step (int): Global step value to record
     |          num_thresholds (int): Number of thresholds used to draw the curve.
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |          see: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md
     |  
     |  add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False)
     |      Add scalar data to summary.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          scalar_value (float or string/blobname): Value to save
     |          global_step (int): Global step value to record
     |          walltime (float): Optional override default walltime (time.time())
     |            with seconds after epoch of event
     |          new_style (boolean): Whether to use new style (tensor field) or old
     |            style (simple_value field). New style could lead to faster data loading.
     |      Examples::
     |      
     |          from torch.utils.tensorboard import SummaryWriter
     |          writer = SummaryWriter()
     |          x = range(100)
     |          for i in x:
     |              writer.add_scalar('y=2x', i * 2, i)
     |          writer.close()
     |      
     |      Expected result:
     |      
     |      .. image:: _static/img/tensorboard/add_scalar.png
     |         :scale: 50 %
     |  
     |  add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None)
     |      Adds many scalar data to summary.
     |      
     |      Args:
     |          main_tag (string): The parent name for the tags
     |          tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
     |          global_step (int): Global step value to record
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |      
     |      Examples::
     |      
     |          from torch.utils.tensorboard import SummaryWriter
     |          writer = SummaryWriter()
     |          r = 5
     |          for i in range(100):
     |              writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
     |                                              'xcosx':i*np.cos(i/r),
     |                                              'tanx': np.tan(i/r)}, i)
     |          writer.close()
     |          # This call adds three values to the same scalar plot with the tag
     |          # 'run_14h' in TensorBoard's scalar section.
     |      
     |      Expected result:
     |      
     |      .. image:: _static/img/tensorboard/add_scalars.png
     |         :scale: 50 %
     |  
     |  add_text(self, tag, text_string, global_step=None, walltime=None)
     |      Add text data to summary.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          text_string (string): String to save
     |          global_step (int): Global step value to record
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |      Examples::
     |      
     |          writer.add_text('lstm', 'This is an lstm', 0)
     |          writer.add_text('rnn', 'This is an rnn', 10)
     |  
     |  add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None)
     |      Add video data to summary.
     |      
     |      Note that this requires the ``moviepy`` package.
     |      
     |      Args:
     |          tag (string): Data identifier
     |          vid_tensor (torch.Tensor): Video data
     |          global_step (int): Global step value to record
     |          fps (float or int): Frames per second
     |          walltime (float): Optional override default walltime (time.time())
     |            seconds after epoch of event
     |      Shape:
     |          vid_tensor: :math:`(N, T, C, H, W)`. The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.
     |  
     |  close(self)
     |  
     |  flush(self)
     |      Flushes the event file to disk.
     |      Call this method to make sure that all pending events have been written to
     |      disk.
     |  
     |  get_logdir(self)
     |      Returns the directory where event files will be written.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)


​    


```python
writer = SummaryWriter("logs") #将事件文件存储到logs文件夹下
```

- 在下面的实例中，主要会用到两个方法：
~~~python
writer.add_image() #添加图片
writer.add_scalar() #添加标量即数
~~~

### （1）writer.add_scalar()的使用


```python
help(writer.add_scalar)
```

    Help on method add_scalar in module torch.utils.tensorboard.writer:
    
    add_scalar(tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False) method of torch.utils.tensorboard.writer.SummaryWriter instance
        Add scalar data to summary.
        
        Args:
            tag (string): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              with seconds after epoch of event
            new_style (boolean): Whether to use new style (tensor field) or old
              style (simple_value field). New style could lead to faster data loading.
        Examples::
        
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = range(100)
            for i in x:
                writer.add_scalar('y=2x', i * 2, i)
            writer.close()
        
        Expected result:
        
        .. image:: _static/img/tensorboard/add_scalar.png
           :scale: 50 %


​    

- 从帮助文档中可以看出，global_step即为横坐标，scalar_value即为纵坐标

**一个简单的例子：使用SummaryWriter绘制y=x的图像**


```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs") #将事件文件存储到logs文件夹下

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()
```

- 运行完后，打开anaconda prompt,输入以下命令：
~~~
tensorboard --logdir=logs 
~~~
在得到的结果中找到网址，按住ctrl键进入网址，如果发现程序没有读到数据，那么将logs改为绝对路径即可

- 此外，为了防止和别人冲突，可以使用下面的命令
~~~
tensorboard --logdir=logs --port=XXXX
~~~
其中XXXX可以改为除6006外其他的端口

- 这样的话，就可以帮助我们显示训练几轮以后，损失的变化

### （2）writer.add_image()的使用


```python
help(writer.add_image)
```

    Help on method add_image in module torch.utils.tensorboard.writer:
    
    add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW') method of torch.utils.tensorboard.writer.SummaryWriter instance
        Add image data to summary.
        
        Note that this requires the ``pillow`` package.
        
        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event
            dataformats (string): Image data format specification of the form
              CHW, HWC, HW, WH, etc.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
        
        Examples::
        
            from torch.utils.tensorboard import SummaryWriter
            import numpy as np
            img = np.zeros((3, 100, 100))
            img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
            img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
        
            img_HWC = np.zeros((100, 100, 3))
            img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
            img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
        
            writer = SummaryWriter()
            writer.add_image('my_image', img, 0)
        
            # If you have non-default dimension setting, set the dataformats argument.
            writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
            writer.close()
        
        Expected result:
        
        .. image:: _static/img/tensorboard/add_image.png
           :scale: 50 %


​    


```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs") #将事件文件存储到logs文件夹下
img_path = "dataset/train/daisy/5547758_eea9edfd54_n.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(type(img_array))

writer.add_image("test", img_array, 1, dataformats="HWC")

writer.close()
```

    <class 'numpy.ndarray'>


- 按照上面的方法便在tensorboard上查看图片，其中，需要注意图片格式的转换，

- 在np.array那一步骤中，要将PIL格式的数据转化为torch.Tensor, numpy.array, or string/blobname中的一种，这里选择了numpy类型，

- 此外，通过查看img_array.shape可以发现，数据类型并不是默认的（3，H,W），而是（H,W,3），所以需要将图片转化,使用add_image中的dataformats即可转换

最后，通过在anaconda prompt输入上面刚刚提到的命令，就可以在下面的端口中查看图片了。

# 四、图片格式变换工具——Transform的使用

## 1.Compose
- 作用：将多个transforms转换打包

## 2.ToTensor()
- 把一个PIL或者numpy类型的图片转化为tensor数据类型

## 3.Resize()
- 图片尺寸变换

## 通过Transform.ToTensor()去看两个问题

- 1.transform如何使用
- 2.为什么需要Tensor数据类型


```python
from PIL import Image
from torchvision import transforms

# Python用法 -》tensor数据类型
# 通过Transform.ToTensor()去看两个问题
# 1.transform如何使用
# 2.为什么需要Tensor数据类型

# 绝对路径 D:\Micro_Climate_Summer_Task\Pytorch_For_Deep_Learning\dataset\train\daisy\5547758_eea9edfd54_n.jpg
# 相对路径 dataset/train/daisy/5547758_eea9edfd54_n.jpg
img_path = "dataset/train/daisy/5547758_eea9edfd54_n.jpg"
img = Image.open(img_path)
print(type(img))
```

    <class 'PIL.JpegImagePlugin.JpegImageFile'>


- 可以看出Image.open函数的返回结果是PIL类型数据，需要转化为tensor数据类型才便于操作


```python
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(type(tensor_img))
```

    <class 'torch.Tensor'>


- 数据类型转化成功！

### 第一个问题解决：
可以通过下面的两行代码来完成对图片类型的转化
~~~python
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
~~~

- 实际使用过程中，需要自己创将工具，即：

图片->使用transform.py创建自己的工具->得到结果

### 第二个问题：为什么要使用Tensor的数据类型

- 通过将上面的代码输入到控制台中，观察右边的tensor数据类型可知，这种数据类型包含了深度学习中一些重要的参数，
- 比如反向传播参数，神经元相关参数等，方便我们更好学习深度学习


```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Python用法 -》tensor数据类型
# 通过Transform.ToTensor()去看两个问题
# 1.transform如何使用
# 2.为什么需要Tensor数据类型

# 绝对路径 D:\Micro_Climate_Summer_Task\Pytorch_For_Deep_Learning\dataset\train\daisy\5547758_eea9edfd54_n.jpg
# 相对路径 dataset/train/daisy/5547758_eea9edfd54_n.jpg
img_path = "dataset/train/daisy/5547758_eea9edfd54_n.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1.transform如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img",tensor_img)

writer.close()
```

- 通过以上的代码，便能在tensorboard中查看tensor类型的图片了

## 常用的Transforms函数

### 1.ToTensor()的使用方法


```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

writer = SummaryWriter("logs")
img = Image.open("imgs/dog.jpg")
print(img)

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)
writer.close()
```

    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=488x331 at 0x17403C9C820>


运行完毕后就可以在tensorboard中查看图片了

### 2.ToPILImage() 讲其他格式转化为PIL格式，不常用，不提了

### 3.Normalize()的使用方法

- 官方文档解释:
    ```python
    ""Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    ```


```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

writer = SummaryWriter("logs")
img = Image.open("imgs/dog.jpg")

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

writer.close()
```

### 4.Resize()的使用方法

#### 第一种方式先resize,再转化为tensor类型


```python
# Resize
print(img)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
```

    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=488x331 at 0x17417F482B0>
    <PIL.Image.Image image mode=RGB size=512x512 at 0x174040F3B50>


运行以后可以看出原尺寸和resize之后尺寸的变化

#### 第二种方式  使用Compose函数


```python
# Compose Resize - 2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)
```

- 图片转化在Compose中是按顺序执行的,后面函数的输入和前面函数的输出是匹配的

### 5.RandomCrop()的使用方法


```python
# RandomCrop
trans_random = transforms.RandomCrop(64)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)
```

根据以上代码，便在tensorboard中看到十张随机裁剪的图片

除了只在RandomCrop传入一个参数代表要裁剪的是正方形以外，还可以传入两个不同的参数代表要裁剪的是一个长方形

## 技巧与方法：

- 学习一个函数的时候首先关注输入与输出
- 看官方文档的来看这个函数具体是用来做什么的
- 关注需要什么参数
- 不知道返回值的时候print()或者print(type())或者debug

# 五、torchvision中数据集的使用

[Pytorch官方网站](http://pytorch.org)

点击官方文档中的Docs，Pytorch为Pytorch的核心模块，主要使用的是torchvision,点开以后可以在官方文档中的torchvision.dataset中看到官方给出的数据集

注意要使用0.9.0版本


## Dataset和transform的联合使用

- 以CIFAR10为例

数据集相关参数
~~~python
Parameters:	
root (string) – Root directory of dataset where directory cifar-10-batches-py exists or will be saved to if download is set to True.
train (bool, optional) – If True, creates dataset from training set, otherwise creates from test set.
transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
target_transform (callable, optional) – A function/transform that takes in the target and transforms it.
download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
~~~


```python
import torchvision

train_set = torchvision.datasets.CIFAR10(root="dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="dataset", train=False, download=True)
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to dataset\cifar-10-python.tar.gz



      0%|          | 0/170498071 [00:00<?, ?it/s]


    Extracting dataset\cifar-10-python.tar.gz to dataset
    Files already downloaded and verified



```python
print(test_set[0])
```

    (<PIL.Image.Image image mode=RGB size=32x32 at 0x17418B06E80>, 3)


- test_set中有一个属性叫做classes,即当为0时是‘Airplane’


```python
print(test_set.classes)
```

    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



```python
img, target = test_set[0]
print(img)
print(target)
```

    <PIL.Image.Image image mode=RGB size=32x32 at 0x174040FD340>
    3



```python
print(test_set.classes[target])
img.show()
```

    cat


### 如何将图片转化为tensor类型？


```python
import torchvision
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=dataset_transform, download=True)
print(test_set[0])
```

    Files already downloaded and verified
    Files already downloaded and verified
    (tensor([[[0.6196, 0.6235, 0.6471,  ..., 0.5373, 0.4941, 0.4549],
             [0.5961, 0.5922, 0.6235,  ..., 0.5333, 0.4902, 0.4667],
             [0.5922, 0.5922, 0.6196,  ..., 0.5451, 0.5098, 0.4706],
             ...,
             [0.2667, 0.1647, 0.1216,  ..., 0.1490, 0.0510, 0.1569],
             [0.2392, 0.1922, 0.1373,  ..., 0.1020, 0.1137, 0.0784],
             [0.2118, 0.2196, 0.1765,  ..., 0.0941, 0.1333, 0.0824]],
    
            [[0.4392, 0.4353, 0.4549,  ..., 0.3725, 0.3569, 0.3333],
             [0.4392, 0.4314, 0.4471,  ..., 0.3725, 0.3569, 0.3451],
             [0.4314, 0.4275, 0.4353,  ..., 0.3843, 0.3725, 0.3490],
             ...,
             [0.4863, 0.3922, 0.3451,  ..., 0.3804, 0.2510, 0.3333],
             [0.4549, 0.4000, 0.3333,  ..., 0.3216, 0.3216, 0.2510],
             [0.4196, 0.4118, 0.3490,  ..., 0.3020, 0.3294, 0.2627]],
    
            [[0.1922, 0.1843, 0.2000,  ..., 0.1412, 0.1412, 0.1294],
             [0.2000, 0.1569, 0.1765,  ..., 0.1216, 0.1255, 0.1333],
             [0.1843, 0.1294, 0.1412,  ..., 0.1333, 0.1333, 0.1294],
             ...,
             [0.6941, 0.5804, 0.5373,  ..., 0.5725, 0.4235, 0.4980],
             [0.6588, 0.5804, 0.5176,  ..., 0.5098, 0.4941, 0.4196],
             [0.6275, 0.5843, 0.5176,  ..., 0.4863, 0.5059, 0.4314]]]), 3)


这样便可以将图片转化为tensor的数据类型了


```python
### 如何查看图片类型
```


```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=dataset_transform, download=True)

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
```

    Files already downloaded and verified
    Files already downloaded and verified


这样就可以在tensorboard中查看test数据集中的前十张图片了

# 六、Dataloader的使用

Dataloader相关参数

~~~python
Parameters
dataset (Dataset) – dataset from which to load the data.

batch_size (int, optional) – how many samples per batch to load (default: 1).

shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).

sampler (Sampler or Iterable, optional) – defines the strategy to draw samples from the dataset. Can be any Iterable with __len__ implemented. If specified, shuffle must not be specified.

batch_sampler (Sampler or Iterable, optional) – like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last.

num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)

collate_fn (callable, optional) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.

pin_memory (bool, optional) – If True, the data loader will copy Tensors into CUDA pinned memory before returning them. If your data elements are a custom type, or your collate_fn returns a batch that is a custom type, see the example below.

drop_last (bool, optional) – set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)

timeout (numeric, optional) – if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: 0)

worker_init_fn (callable, optional) – If not None, this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading. (default: None)

prefetch_factor (int, optional, keyword-only arg) – Number of samples loaded in advance by each worker. 2 means there will be a total of 2 * num_workers samples prefetched across all workers. (default: 2)

persistent_workers (bool, optional) – If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. (default: False)
~~~


```python
import torchvision

from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0,drop_last=False)

# 测试集中的第一张图片及其target
img, target = test_data[0]
print(img.shape)
print(target)

for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)
```

    torch.Size([3, 32, 32])
    3
    torch.Size([4, 3, 32, 32])
    tensor([0, 5, 7, 3])
    torch.Size([4, 3, 32, 32])
    tensor([3, 9, 7, 2])
    torch.Size([4, 3, 32, 32])
    tensor([9, 4, 7, 5])
    torch.Size([4, 3, 32, 32])
    tensor([6, 1, 1, 5])
    ...
    torch.Size([4, 3, 32, 32])
    tensor([7, 8, 4, 7])


### 再使用tensorboard就可以看到图片了|


```python
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0,drop_last=False)

# 测试集中的第一张图片及其target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()
```

    torch.Size([3, 32, 32])
    3


### 如果修改drop_last=True那么最后几张图片会被舍去

### shuffle负责打乱数据
