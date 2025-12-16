# Chilean数据集训练脚本
# Warsaw University of Technology

import os
import sys
import torch

# 添加项目根目录到path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import do_train
from misc.utils import TrainingParams
from eval.evaluate_chilean import evaluate_chilean, print_eval_stats, chilean_write_eval_stats
from models.model_factory import model_factory


def do_train_chilean(params: TrainingParams):
    """
    Chilean数据集训练主函数
    训练完成后自动在Chilean数据集上评估
    """
    # 执行训练
    model, model_pathname = do_train(params, skip_final_eval=True)

    # 训练完成后在Chilean数据集上评估
    print('\n' + '=' * 60)
    print('在Chilean数据集上评估最终模型')
    print('=' * 60)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 加载最终模型权重
    final_model_path = model_pathname + '_final.pth'
    if os.path.exists(final_model_path):
        print(f'加载最终模型权重: {final_model_path}')
        model.load_state_dict(torch.load(final_model_path, map_location=device))
    else:
        print('警告: 未找到最终模型权重，使用当前模型状态')

    model.to(device)

    # 运行Chilean评估
    stats = evaluate_chilean(model, device, params, log=False, show_progress=True)
    print_eval_stats(stats)

    # 保存评估结果
    if stats is not None:
        model_params_name = os.path.split(params.model_params.model_params_path)[1]
        config_name = os.path.split(params.params_path)[1]
        model_name = os.path.splitext(os.path.split(final_model_path)[1])[0]
        prefix = f"{model_params_name}, {config_name}, {model_name}"
        chilean_write_eval_stats("chilean_experiment_results.txt", prefix, stats)


if __name__ == '__main__':
    # 直接设置参数
    class Args:
        def __init__(self):
            self.config = '../config/config_chilean_baseline.txt'
            self.model_config = '../models/minkloc3dv2.txt'
            self.debug = False


    args = Args()

    print('=' * 60)
    print('Chilean数据集训练')
    print('=' * 60)
    print(f'训练配置: {args.config}')
    print(f'模型配置: {args.model_config}')
    print(f'Debug模式: {args.debug}')
    print('')

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    # 开始训练
    do_train_chilean(params)