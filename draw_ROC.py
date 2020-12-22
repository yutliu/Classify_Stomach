import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from auc_calculation import count_auc
from scipy.special import softmax
import random
from collections import defaultdict

random.seed(10)
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Arial']


def draw_ROCcurve_3class(y_label, y_score):
    filename = "result_simple.csv"
    csv_data = pd.read_csv(filename, header=None)
    label = csv_data.iloc[:, 0].values
    first_result = csv_data.iloc[:, 1::2].values
    second_result = csv_data.iloc[:, 2::2].values
    precision = []
    recall = []
    for each_pre in first_result.T:
        if recall_score(label, each_pre, average='weighted') < 0.6:
            continue
        precision.append(1-precision_score(label, each_pre))
        recall.append(recall_score(label, each_pre))

    n_classes = 3
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    total_color = ["#42B540FF", "#ED0000FF", "#00468BFF", "#FDAF91FF"]
    lw=2
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    class_names = ["Normal esophagus", "Reflux esophagitis", "Early esophageal cancer"]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=total_color[i], lw=lw,
                 label='ROC curve of {0}'
                       ''.format(class_names[i]))

    plt.plot(fpr["micro"], tpr["micro"],
             label='Average ROC curve'
                   ''.format(roc_auc["micro"]),
             color=total_color[-1], linewidth=2)

    plt.scatter(precision, recall, c='#925E9FFF', s=20)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(f"ROC curve 3class.png"))
    plt.show()

    assert 1


def draw_ROCcurve_cancer(label_3class, score_3class, all_img_path):
    hospital_names = ["NJDTH", "WXPH", "TZPH"]

    label_cancer = [0 if label != 2 else 1 for label in label_3class]
    pre_cancer = list(score_3class[:, 2])
    pre_cancer_bool = [0 if pre != 2 else 1 for pre in np.argmax(score_3class, axis=1)]

    #model precision and recall
    precision_model = precision_score(label_cancer, pre_cancer_bool)
    recall_model = recall_score(label_cancer, pre_cancer_bool)

    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(label_cancer, pre_cancer_bool).ravel()
    specificity = tn / (tn+fp)

    all_img_name = [img_path.split('/')[-1] for img_path in all_img_path]
    # all_patient = list(set([img_name.split(".")[1] for img_name in all_img_name]))
    all_patient = [img_name.split(".")[1] for img_name in all_img_name]
    all_patient = sorted(set(all_patient))
    random.shuffle(all_patient)

    number_patient = len(all_patient)
    one_hospital_patient = all_patient[:int(number_patient/3)]
    two_hospital_patient = all_patient[int(number_patient/3) : int(2*number_patient/3)]
    three_hospital_patient = all_patient[int(2*number_patient/3):]

    different_hospitals_labels = [[], [], []]
    different_hospitals_pres = [[], [], []]
    normal_patient_number = [[], [], []]
    influ_patient_number = [[], [], []]
    cancer_patient_number = [[], [], []]
    different_hospitals_imgnames = [[], [], []]
    different_hospitals_imgpaths = [[], [], []]
    #generate every hospital result and labels
    different_hospitals_labels_pres = defaultdict(list)
    different_hospitals_info = defaultdict(list)


    normal_imgnumber = [0, 0, 0]
    influ_imgnumber = [0, 0, 0]
    cancer_imgnumber = [0, 0, 0]
    for index, img_name in enumerate(all_img_name):
        patient_name = img_name.split(".")[1]
        if patient_name in one_hospital_patient:
            different_hospitals_labels[0].append(label_cancer[index])
            different_hospitals_pres[0].append(pre_cancer[index])
            different_hospitals_imgnames[0].append(img_name)
            # different_hospitals_imgpaths[0].append()

            #save a hospital result
            different_hospitals_labels_pres["NJDTH"].append([label_cancer[index], pre_cancer[index], pre_cancer_bool[index]])
            different_hospitals_info["NJDTH"].append([img_name, label_cancer[index], pre_cancer[index], pre_cancer_bool[index]])

            if "0_IMG" in img_name:
                normal_imgnumber[0] += 1
                normal_patient_number[0].append(patient_name)
            elif "1_IMG" in img_name:
                influ_imgnumber[0] += 1
                influ_patient_number[0].append(patient_name)
            else:
                cancer_imgnumber[0] += 1
                cancer_patient_number[0].append(patient_name)

        elif patient_name in two_hospital_patient:
            different_hospitals_labels[1].append(label_cancer[index])
            different_hospitals_pres[1].append(pre_cancer[index])
            different_hospitals_imgnames[1].append(img_name)

            different_hospitals_labels_pres["WXPH"].append([label_cancer[index], pre_cancer[index], pre_cancer_bool[index]])
            different_hospitals_info["WXPH"].append([img_name, label_cancer[index], pre_cancer[index], pre_cancer_bool[index]])

            if "0_IMG" in img_name:
                normal_imgnumber[1] += 1
                normal_patient_number[1].append(patient_name)
            elif "1_IMG" in img_name:
                influ_imgnumber[1] += 1
                influ_patient_number[1].append(patient_name)
            else:
                cancer_imgnumber[1] += 1
                cancer_patient_number[1].append(patient_name)
        elif patient_name in three_hospital_patient:
            different_hospitals_labels[2].append(label_cancer[index])
            different_hospitals_pres[2].append(pre_cancer[index])
            different_hospitals_imgnames[2].append(img_name)

            different_hospitals_labels_pres["TZPH"].append([label_cancer[index], pre_cancer[index], pre_cancer_bool[index]])
            different_hospitals_info["TZPH"].append([img_name, label_cancer[index], pre_cancer[index], pre_cancer_bool[index]])

            if "0_IMG" in img_name:
                normal_imgnumber[2] += 1
                normal_patient_number[2].append(patient_name)
            elif "1_IMG" in img_name:
                influ_imgnumber[2] += 1
                influ_patient_number[2].append(patient_name)
            else:
                cancer_imgnumber[2] += 1
                cancer_patient_number[2].append(patient_name)
        else:
            assert 0, "patient name is error!"

    save_img_doctor(different_hospitals_info)

    for hp_name, infos in different_hospitals_labels_pres.items():
        with open(f'csv_files/cancer_{hp_name}.csv', 'w') as f:
            for info in infos:
                f.write("{},{},{}\n".format(*info))

    # normal_patient_number = [len(set(num)) for num in normal_patient_number]
    # influ_patient_number = [len(set(num)) for num in influ_patient_number]
    # cancer_patient_number = [len(set(num)) for num in cancer_patient_number]

    total_color = ["#42B540FF", "#ED0000FF", "#00468BFF"]
    for index, (label, pre) in enumerate(zip(different_hospitals_labels, different_hospitals_pres)):
        #count ROC curve
        fpr, tpr, thersholds = roc_curve(label, pre)
        #count ROC AUC
        auc95 = count_auc(label, pre)
        #draw ROC curve
        plt.plot(fpr, tpr,
                 label='{}: {:.3f}(95% CI {:.3f}-{:.3f})'.format(hospital_names[index], *auc95), lw=2, color=total_color[index])

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("ROC curve of cancer.png", dpi=330, bbox_inches='tight')
    plt.show()

    draw_ROCcurve_cancer_doctor_result(different_hospitals_labels, different_hospitals_pres)
    assert 1



def draw_ROCcurve_cancer_doctor_result(different_hospitals_labels, different_hospitals_pres):
    #Results of man machine competition
    filename = "csv_files/result_doctor.csv"
    csv_data = pd.read_csv(filename, header=None)
    doctor_label = csv_data.iloc[:, 0].values
    first_result = csv_data.iloc[:, 1::2].values
    second_result = csv_data.iloc[:, 2::2].values

    #change label and pre to [0, 1] (no cancer or cancer)
    doctor_label[doctor_label == 1] = 0
    doctor_label[doctor_label == 2] = 1
    first_result[first_result == 1] = 0
    first_result[first_result == 2] = 1

    precision = []
    recall = []
    for each_pre in first_result.T:
        if recall_score(doctor_label, each_pre, average='weighted') < 0.7:
            continue
        precision.append(1-precision_score(doctor_label, each_pre, average='weighted'))
        recall.append(recall_score(doctor_label, each_pre, average='weighted'))


    labels = []
    pres = []
    for index, (label, pre) in enumerate(zip(different_hospitals_labels, different_hospitals_pres)):
        labels.extend(label)
        pres.extend(pre)

    #model precision and recall
    precision_model = precision_score(labels, [int(pre > 0.5) for pre in pres], average='weighted')
    recall_model = recall_score(labels, [int(pre > 0.5) for pre in pres], average='weighted')

    #count ROC curve
    fpr, tpr, thersholds = roc_curve(labels, pres)
    #count ROC AUC
    auc95 = count_auc(labels, pres)
    #draw ROC curve
    plt.plot(fpr, tpr,
             label='All hospitals: {:.3f}(95% CI {:.3f}-{:.3f})'.format(*auc95), lw=2, color="#ED0000FF")
    plt.scatter(precision, recall, c='#925E9FFF', s=20)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(f"ROC curve of cancer with doctor result.png"), dpi=330)
    plt.show()
    assert 1



def draw_ROCcurve_inflam(label_3class, score_3class, all_img_path):
    """
    :param label_3class: list [N]
    :param score_3class: np.array [N, 3]
    :param all_img_path: list [N]
    :return:
    """
    label_inflam = label_3class[:]
    pre_inflam= softmax(score_3class[:, :2], axis=1)

    delete_index = []
    for index, label in enumerate(label_3class):
        if label == 2:
            delete_index.append(index)

    label_inflam = np.delete(label_inflam, delete_index, axis=0)
    pre_inflam = np.delete(pre_inflam, delete_index, axis=0)
    img_path_inflam = np.delete(all_img_path, delete_index, axis=0)
    pre_inflam = list(pre_inflam[:, 1])

    # all_patient = sorted(set(all_patient))
    # random.shuffle(all_patient)

    img_name_inflam = [img_path.split('/')[-1] for img_path in img_path_inflam]
    patient_inflam = list(set([img_name.split(".")[1] for img_name in img_name_inflam]))

    number_patient = len(patient_inflam)
    one_hospital_patient = patient_inflam[:int(number_patient/3)]
    two_hospital_patient = patient_inflam[int(number_patient/3) : int(2*number_patient/3)]
    three_hospital_patient = patient_inflam[int(2*number_patient/3):]

    different_hospitals_labels = [[], [], []]
    different_hospitals_pres = [[], [], []]

    different_hospitals_labels_pre = defaultdict(list)

    for index, img_name in enumerate(img_name_inflam):
        patient_name = img_name.split(".")[1]
        if patient_name in one_hospital_patient:
            different_hospitals_labels[0].append(label_inflam[index])
            different_hospitals_pres[0].append(pre_inflam[index])
            different_hospitals_labels_pre['NJDTH'].append([label_inflam[index], pre_inflam[index], int(pre_inflam[index]>0.5)])
        elif patient_name in two_hospital_patient:
            different_hospitals_labels[1].append(label_inflam[index])
            different_hospitals_pres[1].append(pre_inflam[index])
            different_hospitals_labels_pre['WXPH'].append([label_inflam[index], pre_inflam[index], int(pre_inflam[index] > 0.5)])
        elif patient_name in three_hospital_patient:
            different_hospitals_labels[2].append(label_inflam[index])
            different_hospitals_pres[2].append(pre_inflam[index])
            different_hospitals_labels_pre['TZPH'].append([label_inflam[index], pre_inflam[index], int(pre_inflam[index] > 0.5)])
        else:
            assert 0, "patient name is error!"

    for hp_name, infos in different_hospitals_labels_pre.items():
        with open(f'csv_files/inflam_{hp_name}.csv', 'w') as f:
            for info in infos:
                f.write("{},{},{}\n".format(*info))

    total_color = ["#42B540FF", "#ED0000FF", "#00468BFF"]
    hospital_names = ["NJDTH", "WXPH", "TZPH"]
    for index, (label, pre) in enumerate(zip(different_hospitals_labels, different_hospitals_pres)):
        #count ROC curve
        fpr, tpr, thersholds = roc_curve(label, pre)
        #count ROC AUC
        auc95 = count_auc(label, pre)
        #draw ROC curve
        plt.plot(fpr, tpr,
                 label='{}: {:.3f}(95% CI {:.3f}-{:.3f})'.format(hospital_names[index], *auc95), lw=2, color=total_color[index])

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(f"ROC curve of esophagitis.png"), dpi=330)
    plt.show()

    assert 1


def save_img_doctor(different_hospitals_info):
    #Results of man machine competition
    filename = "csv_files/result_doctor_imgname.csv"
    csv_data = pd.read_csv(filename, header=None)
    doctor_img_result_dict = defaultdict(list)
    for lines in csv_data.iterrows():
        lines = lines[1].values
        doctor_img_result_dict[lines[0]] = lines[2:]

    with open("csv_files/result_imgname_model_doctor.csv", "w") as f:
        hospital_names = ["NJDTH", "WXPH", "TZPH"]
        for hsp_name in hospital_names:
            pres = different_hospitals_info[hsp_name]
            f.write(f"{hsp_name}\n")
            for pre in pres:
                f.write("{}, {}, {}, {}, "
                        "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "
                        "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "
                        "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "
                        "{}, {}, {}, {}, {}, {}, {}, {}\n".format(*pre, *doctor_img_result_dict[pre[0]]))
            f.write("\n")