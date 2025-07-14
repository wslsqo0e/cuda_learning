# 参考内容
https://blog.csdn.net/LostUnravel/article/details/137613493

通过shared memory过渡，访问global memory时尽量连续访存。
shared memory 访问时避免bank-conflict，可以达到很高的性能

# 执行结果
```
--- Running matrix_transpose ---
build/tests/matrix_transpose
check_transpose_correction_v0 success!
------------------------------------------
v0: Average duration over 5 runs: 0.163008 ms
------------------------------------------
Bandwidth: 157.69 GB/s

check_transpose_correction_v1 success!
------------------------------------------
v1: Average duration over 5 runs: 0.103974 ms
------------------------------------------
Bandwidth: 247.22 GB/s

check_transpose_correction_v2 success!
------------------------------------------
v2: Average duration over 5 runs: 0.10167 ms
------------------------------------------
Bandwidth: 252.82 GB/s

check_transpose_correction_v3 success!
------------------------------------------
v3: Average duration over 5 runs: 0.04848 ms
------------------------------------------
Bandwidth: 530.21 GB/s

check_transpose_correction_v4 success!
------------------------------------------
v4: Average duration over 5 runs: 0.0491712 ms
------------------------------------------
Bandwidth: 522.76 GB/s
```
