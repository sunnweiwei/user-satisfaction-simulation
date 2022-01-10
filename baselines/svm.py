from .train_sat import load_dstc
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
import jieba
import copy


def get_main_score(scores):
    number = [0, 0, 0, 0, 0]
    for item in scores:
        number[item] += 1
    score = np.argmax(number)
    return score


def load_jddc(dirname, lite=1):
    raw = [line[:-1] for line in open(dirname, encoding='utf-8')]

    from jddc_config import domain2actions

    act2domain = {}
    for line in domain2actions.split('\n'):
        domain = line[:line.index('[') - 1].strip()
        actions = [x[1:-1] for x in line[line.index('[') + 1:-1].split(', ')]
        # print(domain, actions)
        for x in actions:
            act2domain[x] = domain
    data = []
    for line in raw:
        if len(line) == 0:
            data.append([])
        else:
            data[-1].append(line)
    x = []
    emo = []
    act = []
    action_list = {'other': 0}
    for session in data:
        his_input_ids = []
        for turn in session:
            role, text, action, score = turn.split('\t')
            score = score.split(',')
            if role == 'USER':
                x.append(copy.deepcopy(' '.join(his_input_ids)))
                emo.append(get_main_score([int(item) - 1 for item in score]))
                action = action.strip()
                if lite:
                    action = act2domain.get(action, 'other')
                if action not in action_list:
                    action_list[action] = len(action_list)
                act.append(action_list[action])
            his_input_ids.append(' '.join(jieba.cut(text.strip())))
            # his_input_ids.append(' '.join(text.strip()))

    action_num = len(action_list)
    data = [x, emo, act, action_num]
    return data


def load_data(dirname):
    raw = [line[:-1] for line in open(dirname, encoding='utf-8')]
    data = []
    for line in raw:
        if line == '':
            data.append([])
        else:
            data[-1].append(line)
    x = []
    emo = []
    act = []
    action_list = {}
    for session in data:
        his_input_ids = []
        for turn in session:
            role, text, action, score = turn.split('\t')
            score = score.split(',')
            action = action.split(',')
            action = action[0]
            if role.upper() == 'USER':
                x.append(copy.deepcopy(' '.join(his_input_ids)))
                emo.append(get_main_score([int(item) - 1 for item in score]))
                action = action.strip()
                if action not in action_list:
                    action_list[action] = len(action_list)
                act.append(action_list[action])
            his_input_ids.append(text.strip())

    action_num = len(action_list)
    data = [x, emo, act, action_num]
    return data


def train(fold=0):
    print('fold', fold)

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    from sklearn.metrics import cohen_kappa_score
    from .spearman import spearman

    dataset = 'MWOZ'

    # x, emo, act, action_num = load_jddc(f'dataset/{dataset}')
    x, emo, act, action_num = load_data(f'dataset/{dataset},txt')

    ll = int(len(x) / 10)
    train_x = x[:ll * fold] + x[ll * (fold + 1):]
    train_act = emo[:ll * fold] + emo[ll * (fold + 1):]

    test_x = x[ll * fold:ll * (fold + 1)]
    test_act = emo[ll * fold:ll * (fold + 1)]

    # ===================

    print('build tf-idf')
    vectorizer = CountVectorizer()
    train_feature = vectorizer.fit_transform(train_x)
    test_feature = vectorizer.transform(test_x)

    model = XGBClassifier()
    model.fit(train_feature, train_act)
    prediction = model.predict(test_feature)

    # svm = SVC(C=1.0, kernel="linear")
    # svm.fit(train_feature, train_act)
    # prediction = svm.predict(test_feature)

    # lr = LogisticRegression()
    # lr.fit(train_feature, train_act)
    # prediction = lr.predict(test_feature)

    label = test_act

    recall = [[0, 0] for _ in range(5)]
    for p, l in zip(prediction, label):
        recall[l][1] += 1
        recall[l][0] += int(p == l)
    recall_value = [item[0] / max(item[1], 1) for item in recall]
    print('Recall value:', recall_value)
    print('Recall:', recall)
    UAR = sum(recall_value) / len(recall_value)
    kappa = cohen_kappa_score(prediction, label)
    rho = spearman(prediction, label)

    bi_pred = [int(item < 2) for item in prediction]
    bi_label = [int(item < 2) for item in label]
    bi_recall = sum([int(p == l) for p, l in zip(bi_pred, bi_label) if l == 1]) / max(bi_label.count(1), 1)
    bi_precision = sum([int(p == l) for p, l in zip(bi_pred, bi_label) if p == 1]) / max(bi_pred.count(1), 1)
    bi_f1 = 2 * bi_recall * bi_precision / max((bi_recall + bi_precision), 1)

    print(UAR, kappa, rho, bi_f1)

    with open(f'outputs/{dataset}_emo/xgb_{fold}_0.txt', 'w', encoding='utf-8') as f:
        for p, l in zip(prediction, label):
            f.write(f'{p}, {l}\n')


def train_act(fold=0):
    print('fold', fold)

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    from sklearn.metrics import cohen_kappa_score
    from .spearman import spearman

    dataset = 'JDDC'

    x, emo, act, action_num = load_jddc(f'dataset/{dataset}.txt')
    # x, emo, act, action_num = load_data(f'dataset/{dataset}.txt')

    ll = int(len(x) / 10)
    train_x = x[:ll * fold] + x[ll * (fold + 1):]
    train_act = act[:ll * fold] + act[ll * (fold + 1):]

    test_x = x[ll * fold:ll * (fold + 1)]
    test_act = act[ll * fold:ll * (fold + 1)]

    print('build tf-idf')
    vectorizer = CountVectorizer()
    train_feature = vectorizer.fit_transform(train_x)
    test_feature = vectorizer.transform(test_x)

    # model = XGBClassifier()
    # model.fit(train_feature, train_act)
    # prediction = model.predict(test_feature)

    # svm = SVC(C=1.0, kernel="linear")
    # svm.fit(train_feature, train_act)
    # prediction = svm.predict(test_feature)

    lr = LogisticRegression()
    lr.fit(train_feature, train_act)
    prediction = lr.predict(test_feature)

    label = test_act

    acc = sum([int(p == l) for p, l in zip(prediction, label)]) / len(label)
    precision = precision_score(label, prediction, average='macro', zero_division=0)
    recall = recall_score(label, prediction, average='macro', zero_division=0)
    f1 = f1_score(label, prediction, average='macro', zero_division=0)

    print(acc, precision, recall, f1)

    with open(f'outputs/{dataset}_act/lr_{fold}_0.txt', 'w', encoding='utf-8') as f:
        for p, l in zip(prediction, label):
            f.write(f'{p}, {l}\n')
