
if [[ "$1" == "ldmatrix_with_bank_conflict" ]]; then
    make $1
    bash run_metrics.sh $1
    exit $?
fi

make
bash run_metrics.sh
