import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
import foolbox as fb
from sklearn.metrics import confusion_matrix
import argparse

def load_all_model(dataset, model_struct, approach):
    ori_mdl_path = os.path.join('models', f'{dataset}_{model_struct}', f'{model_struct}_ori_model.h5')
    model_ori = tf.keras.models.load_model(ori_mdl_path)
    # repaired model
    rep_mdl_path1 = os.path.join('evaluation_models', f'{approach}', f'{model_struct}', f'{approach}_{dataset}_{model_struct}_fixed_model1.h5')
    rep_mdl_path2 = os.path.join('evaluation_models', f'{approach}', f'{model_struct}', f'{approach}_{dataset}_{model_struct}_fixed_model2.h5')
    rep_mdl_path3 = os.path.join('evaluation_models', f'{approach}', f'{model_struct}', f'{approach}_{dataset}_{model_struct}_fixed_model3.h5')
    model1 = tf.keras.models.load_model(rep_mdl_path1)
    model2 = tf.keras.models.load_model(rep_mdl_path2)
    model3 = tf.keras.models.load_model(rep_mdl_path3)
    return model_ori, model1, model2, model3

def load_fairness_data(dataset):
    if dataset != 'utkface' and dataset != 'cifar10s':
        print('fairness not support other dataset')
    else:
        if dataset == 'cifar10s':
            fair_data_path1 = os.path.join('datasets', f'{dataset}', 'gray.h5')
            fair_data_path2 = os.path.join('datasets', f'{dataset}', 'color.h5')
            with h5py.File(fair_data_path1, 'r') as f:
                x_g1 = f['images'][:]
                y_g1 = f['labels'][:]
            with h5py.File(fair_data_path2, 'r') as f:
                x_g2 = f['images'][:]
                y_g2 = f['labels'][:]
            mean = np.array([0.16384, 0.16384, 0.16384])
            std = np.array([0.33210605, 0.328938, 0.3250164])
            x_g1 = (x_g1 - mean)/std
            x_g2 = (x_g2 - mean)/std
            y_g1 = tf.keras.utils.to_categorical(y_g1, num_classes=10)
            y_g2 = tf.keras.utils.to_categorical(y_g2, num_classes=10)
            return x_g1, y_g1, x_g2, y_g2
        elif dataset == 'utkface':
            fair_data_path1 = os.path.join('datasets', f'{dataset}', 'white.h5')
            fair_data_path2 = os.path.join('datasets', f'{dataset}', 'nonwhite.h5')
            with h5py.File(fair_data_path1, 'r') as f:
                x_g1 = f['images'][:]
                y_g1 = f['labels_gender'][:]
            with h5py.File(fair_data_path2, 'r') as f:
                x_g2 = f['images'][:]
                y_g2 = f['labels_gender'][:]
            mean = np.array([0.03317774, 0.03317774, 0.03317774])
            std = np.array([0.18214759, 0.18214759, 0.18214759])
            x_g1 = (x_g1 - mean)/std
            x_g2 = (x_g2 - mean)/std
            y_g1 = tf.keras.utils.to_categorical(y_g1, num_classes=2)
            y_g2 = tf.keras.utils.to_categorical(y_g2, num_classes=2)

            return x_g1, y_g1, x_g2, y_g2

def load_data(dataset):
    train_data_path = os.path.join('datasets', f'{dataset}', 'train.h5')
    val_data_path = os.path.join('datasets', f'{dataset}', 'val.h5')
    repair_data_path = os.path.join('datasets', f'{dataset}', 'repair.h5')
    test_data_path = os.path.join('datasets', f'{dataset}', 'test.h5')

    if dataset == 'mnist':
        with h5py.File(test_data_path, 'r') as f:
            x_test_ori = f['images'][:]
            y_test_ori = f['labels'][:]
        x_test = x_test_ori.reshape(-1, 28, 28, 1) / 255.0
        y_test = tf.keras.utils.to_categorical(y_test_ori, num_classes=10)
        return x_test_ori, y_test_ori, x_test, y_test
    elif dataset == 'cifar10':
        with h5py.File(test_data_path, 'r') as f:
            x_test_ori = f['images'][:]
            y_test_ = f['labels'][:]
            y_test_ori = np.array([label[0] for label in y_test_])
        mean = np.array([0.4914009, 0.48215896, 0.4465308])
        std = np.array([0.24703279, 0.24348423, 0.26158753])
        x_test = x_test_ori.astype('float32')/255.0
        x_test = (x_test - mean)/std
        y_test = tf.keras.utils.to_categorical(y_test_ori, num_classes=10)
        return x_test_ori, y_test_ori, x_test, y_test
    elif dataset == 'cifar10s':
        with h5py.File(test_data_path, 'r') as f:
            x_test_ori = f['images'][:]
            y_test_ori = f['labels'][:]
        mean = np.array([0.16384, 0.16384, 0.16384])
        std = np.array([0.33210605, 0.328938, 0.3250164])
        x_test = (x_test_ori - mean)/std
        y_test = tf.keras.utils.to_categorical(y_test_ori, num_classes=10)
        return x_test_ori, y_test_ori, x_test, y_test
    elif dataset == 'utkface':
        with h5py.File(test_data_path, 'r') as f:
            x_test_ori = f['images'][:]
            y_test_ori = f['labels_gender'][:]
        mean = np.array([0.03317774, 0.03317774, 0.03317774])
        std = np.array([0.18214759, 0.18214759, 0.18214759])
        x_test = (x_test_ori - mean)/std
        y_test = tf.keras.utils.to_categorical(y_test_ori, num_classes=2)
        return x_test_ori, y_test_ori, x_test, y_test
    elif dataset == 'imagenet':
        imagenet_test_data_path = os.path.join('datasets', dataset, 'new_test')
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_generator = test_datagen.flow_from_directory(
            imagenet_test_data_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        x_test, y_test = [], []
        for _ in range(len(test_generator)):
            images, labels = test_generator.next()
            x_test.append(images)
            y_test.append(labels)
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)
        x_test_ori = x_test
        y_test_ori = np.argmax(y_test, axis=1)
        return x_test_ori, y_test_ori, x_test, y_test

def save_correct_predictions(dataset, approach, x_test_ori, y_test_ori, indices):
    folder_path = f'robustness_ori_data/{dataset}/{approach}'
    file_path = os.path.join(folder_path, 'datasetboth_correct.h5')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('images', data=x_test_ori[indices])
        f.create_dataset('labels', data=y_test_ori[indices])

def load_both_correct_data(dataset, approach):
    folder_path = f'robustness_ori_data/{dataset}/{approach}'
    file_path = os.path.join(folder_path, 'datasetboth_correct.h5')
    if dataset == 'cifar10':
        with h5py.File(file_path, 'r') as f:
            x_both_correct = f['images'][:]
            y_both_correct = f['labels'][:]
        x_both_correct = x_both_correct.astype('float32')/255.0
        mean = np.array([0.4914009, 0.48215896, 0.4465308])
        std = np.array([0.24703279, 0.24348423, 0.26158753])
        x_both_correct = (x_both_correct - mean)/std
        y_both_correct = tf.keras.utils.to_categorical(y_both_correct, num_classes=10)
        return x_both_correct, y_both_correct
    elif dataset == 'mnist':
        with h5py.File(file_path, 'r') as f:
            x_both_correct = f['images'][:]
            y_both_correct = f['labels'][:]
        x_both_correct = x_both_correct.reshape(-1, 28, 28, 1) / 255.0
        y_both_correct = tf.keras.utils.to_categorical(y_both_correct, num_classes=10)
        return x_both_correct, y_both_correct
    elif dataset == 'cifar10s':
        with h5py.File(file_path, 'r') as f:
            x_both_correct = f['images'][:]
            y_both_correct = f['labels'][:]
        mean = np.array([0.16384, 0.16384, 0.16384])
        std = np.array([0.33210605, 0.328938, 0.3250164])
        x_both_correct = (x_both_correct - mean)/std
        y_both_correct = tf.keras.utils.to_categorical(y_both_correct, num_classes=10)
        return x_both_correct, y_both_correct
    elif dataset == 'utkface':
        with h5py.File(file_path, 'r') as f:
            x_both_correct = f['images'][:]
            y_both_correct = f['labels'][:]
        mean = np.array([0.03317774, 0.03317774, 0.03317774])
        std = np.array([0.18214759, 0.18214759, 0.18214759])
        x_both_correct = (x_both_correct - mean)/std
        y_both_correct = tf.keras.utils.to_categorical(y_both_correct, num_classes=2)
        return x_both_correct, y_both_correct
    elif dataset == 'imagenet':
        with h5py.File(file_path, 'r') as f:
            x_both_correct = f['images'][:]
            y_both_correct = f['labels'][:]
        y_both_correct = tf.keras.utils.to_categorical(y_both_correct, num_classes=1000)
        return x_both_correct, y_both_correct

def test_fairness_AAOD_keras(model, x_g1, y_g1, x_g2, y_g2):
    num_classes = y_g1.shape[1]
    preds_g1 = np.argmax(model.predict(x_g1), axis=1)
    preds_g2 = np.argmax(model.predict(x_g2), axis=1)
    labels_g1 = np.argmax(y_g1, axis=1)
    labels_g2 = np.argmax(y_g2, axis=1)
    cm_g1 = confusion_matrix(labels_g1, preds_g1, labels=np.arange(num_classes))
    cm_g2 = confusion_matrix(labels_g2, preds_g2, labels=np.arange(num_classes))
    tpr_diff_sum = 0
    fpr_diff_sum = 0
    for c in range(num_classes):
        tp_g1 = cm_g1[c, c]
        fn_g1 = cm_g1[c, :].sum() - tp_g1
        fp_g1 = cm_g1[:, c].sum() - tp_g1
        tn_g1 = cm_g1.sum() - (tp_g1 + fn_g1 + fp_g1)
        tp_g2 = cm_g2[c, c]
        fn_g2 = cm_g2[c, :].sum() - tp_g2
        fp_g2 = cm_g2[:, c].sum() - tp_g2
        tn_g2 = cm_g2.sum() - (tp_g2 + fn_g2 + fp_g2)
        tpr_g1 = tp_g1 / (tp_g1 + fn_g1) if (tp_g1 + fn_g1) > 0 else 0
        fpr_g1 = fp_g1 / (fp_g1 + tn_g1) if (fp_g1 + tn_g1) > 0 else 0
        tpr_g2 = tp_g2 / (tp_g2 + fn_g2) if (tp_g2 + fn_g2) > 0 else 0
        fpr_g2 = fp_g2 / (fp_g2 + tn_g2) if (fp_g2 + tn_g2) > 0 else 0
        tpr_diff_sum += abs(tpr_g1 - tpr_g2)
        fpr_diff_sum += abs(fpr_g1 - fpr_g2)
    aaod = 0.5 * (tpr_diff_sum / num_classes + fpr_diff_sum / num_classes)

    return aaod

def get_pgd_params(dataset):
    if dataset == 'mnist':
        epsilons = 0.15
    elif dataset == 'cifar10':
        epsilons = 0.031
    elif dataset == 'cifar10s':
        epsilons = 0.035
    elif dataset == 'utkface':
        epsilons = 0.07
    elif dataset == 'imagenet':
        epsilons = 0.007
    return epsilons

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process parameters.')
    
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model_struct', type=str, default='lenet5')
    parser.add_argument('--approach', type=str, default='apricot') # apricot arachne
    parser.add_argument('--stat_repair_break', type=bool, default=True)
    parser.add_argument('--stat_both_correct', type=bool, default=True)
    parser.add_argument('--stat_eval_accuracy', type=bool, default=True)
    parser.add_argument('--stat_eval_fairness', type=bool, default=False)
    parser.add_argument('--stat_eval_robustness', type=bool, default=True)
    
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_arguments()
    dataset = params.dataset
    model_struct = params.model_struct
    approach = params.approach
    stat_repair_break = params.stat_repair_break
    stat_both_correct = params.stat_both_correct
    stat_eval_accuracy = params.stat_eval_accuracy
    stat_eval_fairness = params.stat_eval_fairness
    stat_eval_robustness = params.stat_eval_robustness

    if dataset == 'utkface' or dataset == 'cifar10s':
        stat_eval_fairness = True

    model_ori, model1, model2, model3 = load_all_model(dataset, model_struct, approach)

    if stat_repair_break:
        x_test_ori, y_test_ori, x_test, y_test = load_data(dataset)
        preds_ori = np.argmax(model_ori.predict(x_test), axis=1)
        preds1 = np.argmax(model1.predict(x_test), axis=1)
        preds2 = np.argmax(model2.predict(x_test), axis=1)
        preds3 = np.argmax(model3.predict(x_test), axis=1)
        correct_indices_ori = np.where(preds_ori == y_test_ori)[0]
        wrong_indices_model1 = np.where(preds1 != y_test_ori)[0]
        wrong_indices_model2 = np.where(preds2 != y_test_ori)[0]
        wrong_indices_model3 = np.where(preds3 != y_test_ori)[0]
        m1_count = len(np.intersect1d(correct_indices_ori, wrong_indices_model1))
        m2_count = len(np.intersect1d(correct_indices_ori, wrong_indices_model2))
        m3_count = len(np.intersect1d(correct_indices_ori, wrong_indices_model3))
        print(f'break numbers by model1: {m1_count}')
        print(f'break numbers by model2: {m2_count}')
        print(f'break numbers by model3: {m3_count}')
        wrong_indices_ori = np.where(preds_ori != y_test_ori)[0]
        correct_indices_model1 = np.where(preds1 == y_test_ori)[0]
        correct_indices_model2 = np.where(preds2 == y_test_ori)[0]
        correct_indices_model3 = np.where(preds3 == y_test_ori)[0]
        m1_correct_count = len(np.intersect1d(wrong_indices_ori, correct_indices_model1))
        m2_correct_count = len(np.intersect1d(wrong_indices_ori, correct_indices_model2))
        m3_correct_count = len(np.intersect1d(wrong_indices_ori, correct_indices_model3))
        print(f'repaired numbers by repaired model1: {m1_correct_count}')
        print(f'repaired numbers by repaired model2: {m2_correct_count}')
        print(f'repaired numbers by repaired model3: {m3_correct_count}')

    if stat_eval_fairness:
        x_g1, y_g1, x_g2, y_g2 = load_fairness_data(dataset)
        aaod_value = test_fairness_AAOD_keras(model_ori, x_g1, y_g1, x_g2, y_g2)
        print(f'AAOD on ori model: {aaod:.4f}')
        aaod_value = test_fairness_AAOD_keras(model1, x_g1, y_g1, x_g2, y_g2)
        print(f'AAOD on repaired model1: {aaod:.4f}')
        aaod_value = test_fairness_AAOD_keras(model2, x_g1, y_g1, x_g2, y_g2)
        print(f'AAOD on repaired model2: {aaod:.4f}')
        aaod_value = test_fairness_AAOD_keras(model3, x_g1, y_g1, x_g2, y_g2)
        print(f'AAOD on repaired model3: {aaod:.4f}')

    if stat_eval_accuracy:
        x_test_ori, y_test_ori, x_test, y_test = load_data(dataset)
        loss_ori, accuracy_ori = model_ori.evaluate(x_test, y_test, verbose=0)
        loss_model1, accuracy_model1 = model1.evaluate(x_test, y_test, verbose=0)
        loss_model2, accuracy_model2 = model2.evaluate(x_test, y_test, verbose=0)
        loss_model3, accuracy_model3 = model3.evaluate(x_test, y_test, verbose=0)

        print(f'Accuracy of original model: {accuracy_ori:.4f}')
        print(f'Accuracy of repaired model 1: {accuracy_model1:.4f}')
        print(f'Accuracy of repaired model 2: {accuracy_model2:.4f}')
        print(f'Accuracy of repaired model 3: {accuracy_model3:.4f}')
        
    if stat_both_correct:
        preds_ori = np.argmax(model_ori.predict(x_test), axis=1)
        preds1 = np.argmax(model1.predict(x_test), axis=1)
        preds2 = np.argmax(model2.predict(x_test), axis=1)
        preds3 = np.argmax(model3.predict(x_test), axis=1)

        correct_indices = np.where((preds_ori == y_test_ori) & (preds1 == y_test_ori) & (preds2 == y_test_ori) & (preds3 == y_test_ori))[0]

        save_correct_predictions(dataset, approach, x_test_ori, y_test_ori, correct_indices)
    
    if stat_eval_robustness:
        x_both_correct, y_both_correct = load_both_correct_data(dataset, approach)
        num_samples_to_select = 500
        if len(y_both_correct) >= num_samples_to_select:
            indices = np.random.choice(len(y_both_correct), num_samples_to_select, replace=False)
        else:
            raise ValueError("Not enough samples to select from.")
        x_selected = x_both_correct[indices]
        y_selected = y_both_correct[indices]
        min_value, max_value = np.min(x_selected), np.max(x_selected)
        y_selected = np.argmax(y_selected, axis=1)
        x_selected = tf.convert_to_tensor(x_selected)
        y_selected = tf.convert_to_tensor(y_selected)
        fmodel_ori = fb.TensorFlowModel(model_ori, bounds=(min_value, max_value))
        fmodel1 = fb.TensorFlowModel(model1, bounds=(min_value, max_value))
        fmodel2 = fb.TensorFlowModel(model2, bounds=(min_value, max_value))
        fmodel3 = fb.TensorFlowModel(model3, bounds=(min_value, max_value))
        epsilons = get_pgd_params(dataset)
        attack = fb.attacks.PGD()
        _, advs, success = attack(fmodel_ori, x_selected, y_selected, epsilons=epsilons)
        print(f'ASR on ori model is {np.mean(success)}')
        _, advs, success = attack(fmodel1, x_selected, y_selected, epsilons=epsilons)
        print(f'ASR on repaired model1 is {np.mean(success)}')
        _, advs, success = attack(fmodel2, x_selected, y_selected, epsilons=epsilons)
        print(f'ASR on repaired model2 is {np.mean(success)}')
        _, advs, success = attack(fmodel3, x_selected, y_selected, epsilons=epsilons)
        print(f'ASR on repaired model3 is {np.mean(success)}')
