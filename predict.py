import argparse
import torch
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from model import resnet34
import json
import csv


def is_image_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def load_image(path: str, transform):
    img = Image.open(path).convert('RGB')
    return transform(img)


def predict_one(net, tensor_img, device):
    net.eval()
    with torch.no_grad():
        input_tensor = tensor_img.unsqueeze(0).to(device)
        output = torch.squeeze(net(input_tensor)).cpu()
        probs = torch.softmax(output, dim=0)
        pred = torch.argmax(probs).item()
        prob = probs[pred].item()
    return pred, prob


def extract_label_from_filename(fname: str):
    # 常见格式: "<label>_<rest>.jpg"，尝试解析前缀为真实label
    base = os.path.basename(fname)
    name = os.path.splitext(base)[0]
    if '_' in name:
        prefix = name.split('_')[0]
        if prefix.isdigit():
            return int(prefix)
    # 否则返回 None，由调用方决定如何处理
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=False, default='test_cifar10',
                        help='Image file or folder to predict (default: test_cifar10)')
    parser.add_argument('-w', '--weights', default='resnet34_cifar10.pth', help='Model weights path')
    parser.add_argument('-c', '--classes', default='class_index.json', help='Class index json')
    parser.add_argument('-o', '--output', default='predictions_with_truth.csv', help='Output CSV file')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 预处理（与 train.py 验证保持一致）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    assert os.path.exists(args.classes), f'Class file {args.classes} not found'
    with open(args.classes, 'r') as f:
        cls_index = json.load(f)

    # 加载模型
    net = resnet34(num_classes=10)
    assert os.path.exists(args.weights), f'Weights file {args.weights} not found'
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.to(device)

    # 收集待预测图片列表
    inp = args.input
    files = []
    if os.path.isdir(inp):
        for root, _, fnames in os.walk(inp):
            for fn in fnames:
                if is_image_file(fn):
                    files.append(os.path.join(root, fn))
    elif os.path.isfile(inp):
        if is_image_file(inp):
            files = [inp]
    else:
        raise ValueError(f'Input path {inp} does not exist')

    if len(files) == 0:
        print('No image files found')
        return

    # 写 CSV: filename, true_label, true_label_name, pred_label, pred_name, prob, correct
    total_with_truth = 0
    correct_count = 0
    missing_truth = 0
    # per-class counters
    per_class_total = {str(k): 0 for k in range(10)}
    per_class_correct = {str(k): 0 for k in range(10)}

    with open(args.output, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['filename', 'true_label', 'true_label_name', 'pred_label', 'pred_name', 'probability', 'correct'])

        for i, fpath in enumerate(sorted(files)):
            try:
                img_t = load_image(fpath, transform)
                pred_label, prob = predict_one(net, img_t, device)

                true_idx = extract_label_from_filename(fpath)
                true_name = cls_index.get(str(true_idx), '') if true_idx is not None else ''
                pred_name = cls_index.get(str(pred_label), str(pred_label))

                correct = ''
                if true_idx is None:
                    missing_truth += 1
                else:
                    total_with_truth += 1
                    per_class_total[str(true_idx)] = per_class_total.get(str(true_idx), 0) + 1
                    if int(true_idx) == int(pred_label):
                        correct = True
                        correct_count += 1
                        per_class_correct[str(true_idx)] = per_class_correct.get(str(true_idx), 0) + 1
                    else:
                        correct = False

                writer.writerow([os.path.relpath(fpath), true_idx if true_idx is not None else '', true_name, pred_label, pred_name, f'{prob:.6f}', correct])

                # 前 5 个打印预览
                if i < 5:
                    print(f'{fpath} -> true: {true_name} ({true_idx}), pred: {pred_name} ({prob:.4f}), correct: {correct}')
            except Exception as e:
                print(f'Error processing {fpath}: {e}')

        # 写入汇总信息到 CSV
        writer.writerow([])
        overall_acc = (correct_count / total_with_truth) if total_with_truth > 0 else 0.0
        writer.writerow(['__SUMMARY__', 'total_with_truth', total_with_truth, 'correct', correct_count, 'missing_truth', missing_truth])
        writer.writerow(['__SUMMARY__', 'overall_accuracy', f'{overall_acc:.6f}'])
        # 每类准确率
        writer.writerow(['__PER_CLASS__', 'class_id', 'class_name', 'total', 'correct', 'accuracy'])
        for k in sorted(per_class_total.keys(), key=lambda x: int(x)):
            tot = per_class_total.get(k, 0)
            corr = per_class_correct.get(k, 0)
            acc = (corr / tot) if tot > 0 else 0.0
            writer.writerow(['', k, cls_index.get(k, ''), tot, corr, f'{acc:.6f}'])

    # 控制台打印汇总
    print('Done. Predictions saved to', args.output)
    print(f'Total images: {len(files)}, with ground-truth: {total_with_truth}, missing truth: {missing_truth}')
    print(f'Correct predictions: {correct_count}, Overall accuracy: {overall_acc:.6f}')
    print('Per-class accuracy:')
    for k in sorted(per_class_total.keys(), key=lambda x: int(x)):
        tot = per_class_total.get(k, 0)
        corr = per_class_correct.get(k, 0)
        acc = (corr / tot) if tot > 0 else 0.0
        print(f'  Class {k} ({cls_index.get(k, "")}): {corr}/{tot} = {acc:.4f}')


if __name__ == '__main__':
    main()