from sklearn.metrics import cohen_kappa_score
from .spearman import spearman
from sklearn.metrics import f1_score, precision_score, recall_score


def test(name):
    best_result = [0. for _ in range(4)]
    for epoch in range(100):

        data = []
        for fold in range(10):
            data.extend([line[:-1] for line in open(f'{name}_{fold}_{epoch}.txt', encoding='utf-8')])

        prediction = [int(line.split(',')[0]) for line in data]
        label = [int(line.split(',')[1]) for line in data]
        recall = [[0, 0] for _ in range(5)]
        for p, l in zip(prediction, label):
            recall[l][1] += 1
            recall[l][0] += int(p == l)
        recall_value = [item[0] / max(item[1], 1) for item in recall]
        # print('Recall value:', recall_value)
        # print('Recall:', recall)
        UAR = sum(recall_value) / len(recall_value)
        kappa = cohen_kappa_score(prediction, label)
        rho = spearman(prediction, label)
        # rho = 0

        bi_pred = [int(item < 2) for item in prediction]
        bi_label = [int(item < 2) for item in label]
        bi_recall = sum([int(p == l) for p, l in zip(bi_pred, bi_label) if l == 1]) / max(bi_label.count(1), 1)
        bi_precision = sum([int(p == l) for p, l in zip(bi_pred, bi_label) if p == 1]) / max(bi_pred.count(1), 1)
        bi_f1 = 2 * bi_recall * bi_precision / max((bi_recall + bi_precision), 1)

        test_result = [UAR, kappa, rho, bi_f1]
        best_result = [max(i1, i2) for i1, i2 in zip(test_result, best_result)]
        print(epoch, best_result, test_result)


def test_act(name):
    best_result = [0. for _ in range(4)]
    for epoch in range(100):

        data = []
        for fold in range(10):
            data.extend([line[:-1] for line in open(f'{name}_{fold}_{epoch}.txt', encoding='utf-8')])

        prediction = [int(line.split(',')[0]) for line in data]
        label = [int(line.split(',')[1]) for line in data]
        acc = sum([int(p == l) for p, l in zip(prediction, label)]) / len(label)
        precision = precision_score(label, prediction, average='macro', zero_division=0)
        recall = recall_score(label, prediction, average='macro', zero_division=0)
        f1 = f1_score(label, prediction, average='macro', zero_division=0)

        test_result = [acc, precision, recall, f1]
        best_result = [max(i1, i2) for i1, i2 in zip(test_result, best_result)]
        print(epoch, best_result, test_result)


if __name__ == '__main__':

    test('outputs/jddc_emo/BERT')
    test_act('outputs/jddc_act/BERT')
