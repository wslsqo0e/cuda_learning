# cuda learning

## z\_code\_templates

存储了代码工程的模板，可以通过拷贝该文件夹创建新的代码工程

+ `zee_utils.h`: 提供一些常用的宏，工具函数，时间统计函数等
+ `zee_ndarray.h`: 提供 `NDArray` 数据结构，方便在CPU或者GPU上快速创建高维数组

里面的 `Makefile` 还提供了测试功能，执行如下命令，可触发对应测试用例的编译和执行
```bash
make zee_ndarray
```

## bank\_conflict\_test
提供了示例代码，验证从 shared memory 加载数据的 bank conflit现象，提供了命令行监控cuda kernel执行时，bank conflict出现次数的命令

验证了 ldmatrix 命令会触发 bank conflict，且bank conflict次数角度。需要使用 swizzle 方法规避ldmatrix的bank conflict

## coalesces\_memory\_access

验证了让warp访问不连续HBM区域对性能的影响
最好是让warp访问地址对其，且连续的HBM地址

## matrix\_transpose
探讨多种不同的矩阵转置方法对性能的影响

v3版本最优
1. 通过shared memory中转，避免对HBM的过多读写。
2. 避免shared memory的bank conflict
3. 让cuda线程执行时尽可能多处理数据

## ldmatrix\_instruction\_test
对 ldmatrix 指令的简单测试

## cuda\_attributes\_get
一些通过cuda api获取gpu属性的例子
