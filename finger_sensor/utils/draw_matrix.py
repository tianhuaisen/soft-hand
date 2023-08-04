# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from model_train.params import Params
import dill

def draw_confusion_matrix(label_true, label_pred, label_name, normlize, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    """
    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["A", "B", "C", "D", "E", "F", "G"],
                          normlize=True,
                          title="Confusion Matrix",
                          pdf_save_path="Confusion_Matrix.png",
                          dpi=300)
    """

    cm = confusion_matrix(label_true, label_pred) # Directly return the confusion matrix
    if normlize:
        row_sums = np.sum(cm, axis=1)
        cm = cm / row_sums[:, np.newaxis]  # Calculate the percentage of each element
    cm = cm.T
    plt.figure(figsize=(16, 12))
    plt.get_current_fig_manager().window.showMaximized()
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    # plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)	# Diagonal font white, others black
            value = float(format('%.2f' % cm[i, j]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color, fontsize=8)

    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, dpi=dpi)
    plt.show()



def draw_run():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_model_path = '../model_train/saved_model/motive 90.28 batch=100.pkl'
    model = torch.load(save_model_path)
    with open('../data_set/test_data/test_data.pkl', 'rb') as f:
        test_dataset = dill.load(f)
    params = Params()
    BATCH_SIZE = params.batch_size
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True)
    labels_name = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                   'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z', 'down', 'left', 'right', 'up']
    y_gt = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for _, s in enumerate(test_dataloader):
            x, y = s['sensor'].float().to(DEVICE), s['label'].float().to(DEVICE)
            labels_pd, _, _, _, _, _ = model(x, DEVICE)
            predict_np = list(np.argmax(labels_pd.cpu().detach().numpy(),
                                   axis=-1))
            labels_np = list(y.cpu().numpy().astype(int))
            y_pred.extend(predict_np)
            y_gt.extend(labels_np)
    # print(y_pred)
    # print(y_gt)
    draw_confusion_matrix(label_true=y_gt,  # y_gt=[0,5,1,6,3,...]
                          label_pred=y_pred,  # y_pred=[0,5,1,6,3,...]
                          label_name=labels_name,
                          normlize=True,
                          title="Confusion Matrix on Finger_Motion",
                          pdf_save_path="./Confusion_Matrix_on_Finger_Motion.png",
                          dpi=300)

if __name__ == '__main__':
    draw_run()