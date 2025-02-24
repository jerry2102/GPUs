# Tensor Core

![Tensor Core](images/pascal%20vs%20volta%20tensor%20core.gif)

Tensor Core可以实现混合精度的矩阵乘加运算，在深度学习的大规模矩阵计算中可以发挥重要作用。


CUDA core针对各种并行计算任务进行优化，更适合通用并行计算任务。


|      | Volta | Turning | Ampere | Hopper |
|------|-------|---------|--------|--------|
|CUDA Core| FP64、FP32、FP16、INT8|FP64、FP32、FP16、INT8|FP64、TF32、FP16、BF16、INT8|FP64、TF32、FP16、BF16、INT8|
|Tensor Core|FP16|FP16、INT8、INT4、INT1|FP64、TF32、BF16、FP16、INT8、INT4、INT1|FP64、TF32、BF16、FP16、FP8、INT8|


## demo
[A tensor core demo](https://zhuanlan.zhihu.com/p/620766588)



*****
参考文献

[1] https://dwpl6xgouw.feishu.cn/wiki/PnYmw4KcdiPBphkJBFdcqdDxnke

[2] https://chenzomi12.github.io/02Hardware04NVIDIA/03DeepTC.html

[3] https://blog.csdn.net/hit_shaoqi/article/details/134498937

[4] https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/


|  1/t0 |  2/t0 |  3/t1 | 4/t1  |  5/t2 |  6/t2 |  7/t3 |  8/t3 |  9/t0 | 10/t0 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17/t4 | 18/t4 | 19/t5 | 20/t5 | 21/t6 | 22/t6 | 23/t7 | 24/t7 | 25/t4 | 26/t4 | 27 | 28 | 29 | 30 | 31 | 32 |
| 33/t8 |    |    |    |    |    |    |  40/t11  |  41/t8  |    |    |    |    |    |    |    |
| 49/t12 |    |    |    |    |    |    |  56/t15  | 57/t12 |    |    |    |    |    |    |    |
| 65/t16 |    |    |    |    |    |    |  72/t19  | 73/t16 |    |    |    |    |    |    |    |
| 81/t20 |    |    |    |    |    |    |  88/t23  | 89/t20 |    |    |    |    |    |    |    |
| 97/t24 |    |    |    |    |    |    |  104/t27 | 105/t24|    |    |    |    |    |    |    |
|113/t28 |    |    |    |    |    |    |  120/t31 | 121/t28|    |    |    |    |    |    |    |
|129/t0 | 130/t0 |    |    |    |    |    | 136/t3 | 137/t0 |    |    |    |    |    |    |    |
|145 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|161 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|177 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|193 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|209 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|225 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|241 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |


|  1/t0 |  2 |  3 | 4  |  5 |  6 |  7 |  8/t3 |  9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17/t4 | 18 | 19 | 20 | 21 | 22 | 23 | 24/t7 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
| 33/t8 |    |    |    |    |    |    |  /t11  |    |    |    |    |    |    |    |    |
| 49/t12 |    |    |    |    |    |    |  /t15  |    |    |    |    |    |    |    |    |
| 65/t16 |    |    |    |    |    |    |  /t19  |    |    |    |    |    |    |    |    |
| 81/t20 |    |    |    |    |    |    |  /t23  |    |    |    |    |    |    |    |    |
| 97/t24 |    |    |    |    |    |    |  /t27  |    |    |    |    |    |    |    |    |
|113/t28 |    |    |    |    |    |    |  /t31  |    |    |    |    |    |    |    |    |
|129/t0 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|145 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|161 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|177 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|193 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|209 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|225 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
|241 |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
