#!/bin/bash
# pip install -U git+https://github.com/pSpitzner/python2jupyter.git

it_dir="$(cd "$(dirname "$0")"; pwd -P)"
py_dir=$(dirname "$it_dir")

cd $it_dir
# git pull
sh ./.generate_ipynb.sh

convert_and_run() {
    ipynb=$(basename $1 ".py")".ipynb"
    # jupyter nbconvert --ExecutePreprocessor.timeout=16000 --to notebook --inplace --allow-errors --execute $ipynb
}

echo "running all files in parallel, output will get messy"
for filename in $py_dir/*.py; do
    convert_and_run $filename &
done
wait

git add $it_dir/*.ipynb
git commit -m "Automatic ipynb update $(date '+%Y-%m-%d %T')"
