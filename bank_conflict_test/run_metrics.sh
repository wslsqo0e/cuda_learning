# l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum: shared memory load操作总计有几轮wavefronts
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum: shared memory load操作总计有几bank conflict
# l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum: shared memory store操作总计有几轮wavefronts
# l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum: shared memory store操作总计有几bank conflict

if [ -n "$1" ]; then
sudo /usr/local/cuda-12.8/bin/ncu --metrics \
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  ./build/tests/"$1"
exit $?
fi

sudo /usr/local/cuda-12.8/bin/ncu --metrics \
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
 ./build/main
