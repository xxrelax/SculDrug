#!/bin/bash

# 检查是否传入实验名称和 revision 参数
if [ $# -lt 3 ]; then
    echo "请提供实验名称 {exp_name} 和 revision 参数"
    exit 1
fi

# 获取传入的实验名称和 revision
EXP_NAME=$1
REVISION=$2
TEST_OUTPUT=$3
# 设置路径变量
SAVE_DIR="/root/project/bfn_mol/results/denovo/${EXP_NAME}/saved_data"
GENERATED_DIR="/root/project/bfn_mol/logs/root_bfn_sbdd/${EXP_NAME}/${REVISION}/${TEST_OUTPUT}"

# 查找 GENERATED_FILE
GENERATED_FILE=$(find "${GENERATED_DIR}" -type f -name "*.pt" | head -n 1)

# 检查是否找到 .pt 文件
if [ -z "$GENERATED_FILE" ]; then
    echo "在目录 ${GENERATED_DIR} 中未找到 .pt 文件"
    exit 1
fi

echo "使用的 .pt 文件为：${GENERATED_FILE}"

# 第一步：运行 exact_mol.py
echo "运行 exact_mol.py..."
python ./evaluate_scripts/exact_mol.py --save_dir "${SAVE_DIR}" --generated_file "${GENERATED_FILE}"
if [ $? -ne 0 ]; then
    echo "执行 exact_mol.py 时发生错误"
    exit 1
fi

# 第二步：运行 evaluate_chem_folder.py
echo "运行 evaluate_chem_folder.py..."
python ./evaluate_scripts/evaluate_chem_folder.py --base_result_path "${SAVE_DIR}"
if [ $? -ne 0 ]; then
    echo "执行 evaluate_chem_folder.py 时发生错误"
    exit 1
fi

# 第三步：运行 cal_chem_results.py
echo "运行 cal_chem_results.py..."
python ./evaluate_scripts/cal_chem_results.py --root_directory "${SAVE_DIR}"
if [ $? -ne 0 ]; then
    echo "执行 cal_chem_results.py 时发生错误"
    exit 1
fi

echo "所有步骤执行完毕！"