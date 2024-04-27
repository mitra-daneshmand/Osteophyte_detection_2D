import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from args import parse_args
from components import checkpoint
from train_utils import init_model
from Bootstraping_curves import curves
import datasets
import transforms
from tqdm import tqdm
from kvs import GlobalKVS
from seed import seed
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import sem, t


def balanced_accuracy_with_ci(y_true, y_pred, confidence_level=0.95):
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    n = len(y_true)
    std_err = sem([1 if p == t else 0 for p, t in zip(y_pred, y_true)])
    h = std_err * t.ppf((1 + confidence_level) / 2, n - 1)
    conf_int = (balanced_acc - h, balanced_acc + h)

    return balanced_acc, conf_int



if torch.cuda.is_available():
    maybe_gpu = 'cuda'
else:
    maybe_gpu = 'cpu'

seed.set_ultimate_seed()
kvs = GlobalKVS()
args = parse_args()
save_dir = args.output_dir + '/{}/'.format(args.target_comp)
kvs.update('save_dir', save_dir)

metadata = datasets.init_metadata(args.lm, args.target_comp, args.csv_dir)

test_index = pd.read_csv(os.path.join(save_dir, 'test.csv'))
test_set = metadata.iloc[test_index.values.flatten()]
test_set['KL'] = test_set['KL'].astype(int)

mean, std = 0.0509, 0.2125  # med
# mean, std = 0.0507, 0.2122  # lat

dataset_test = datasets.KneeDataset(
        dataset=test_set,
        transforms=[
            transforms.Normalize(mean=mean, std=std),
        ]
)
loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8,
                         pin_memory=torch.cuda.is_available())

test_res = 0
folds = 5
for fold_idx in range(folds):
    paths_weights_fold = os.path.join(args.output_dir, args.target_comp, f'fold_{fold_idx}')

    handlers_ckpt = checkpoint.CheckpointHandler(paths_weights_fold)

    paths_ckpt_sel = handlers_ckpt.get_last_ckpt()

    # Initialize and configure the model
    model = init_model().to(maybe_gpu)
    model.load_state_dict(torch.load(paths_ckpt_sel))
    model = nn.DataParallel(model)

    model.eval()

    preds_prog_fold = []
    y = []

    with tqdm(total=len(loader_test), desc=f'Eval, fold {fold_idx}') as prog_bar:
        for i, data_batch in enumerate(loader_test):
            xs, ys_true = data_batch['mask'], data_batch['Target']
            y.append(ys_true)
            xs, ys_true = xs.to(maybe_gpu), ys_true.to(maybe_gpu)

            ys_pred = model(xs)

            preds_prog_fold.append(F.sigmoid(ys_pred).cpu().detach().numpy())

            torch.cuda.empty_cache()
            xs.detach()
            ys_pred.detach()
            del xs
            del ys_pred

            prog_bar.update(1)

    y = [x.numpy().reshape(x.shape[0], 1) for x in y]
    y_true = np.vstack(y)
    preds_prog_fold = np.vstack(preds_prog_fold)
    test_res += preds_prog_fold

test_res /= folds

kvs.update('y_pred', test_res)
kvs.update('y_true', y_true)
kvs.update('eval_type', 'test')
np.savez_compressed(os.path.join(args.output_dir, args.target_comp, 'results.npz'), y_true=y_true, y_pred=test_res)

curves()

result = np.load(os.path.join(save_dir, 'results.npz'))

y_true = result['y_true']
test_res = result['y_pred']

df = pd.DataFrame(columns=['ID', 'SIDE', 'OSTs', 'preds', 'probs'])
df['ID'] = test_set['ID']
df['SIDE'] = test_set['SIDE']
df['OSTs'] = test_set[args.target_comp]
df['probs'] = test_res

fpr, tpr, thresholds = roc_curve(y_true, test_res)
scores = tpr - fpr

ix = np.argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

test_res = (test_res > thresholds[ix]).astype('int')
df['preds'] = test_res

####################BA#########################
confidence_level = 0.95
balanced_acc, conf_int = balanced_accuracy_with_ci(y_true, test_res, confidence_level=0.95)

# Print the results
print("Balanced accuracy: {:.2f}".format(balanced_acc))
print("Confidence interval ({:.0%}): [{:.2f}, {:.2f}, {}]".format(confidence_level, conf_int[0], conf_int[1], (conf_int[1]-conf_int[0])/2))



bal_acc = balanced_accuracy_score(y_true, test_res.round())
print('Balanced Accuracy is: ', bal_acc)

cm = pd.DataFrame(columns=[0, 1])

indx = df[(df['OSTs'] == 0) & (df['preds'] == 0)].index
cm.loc[0, 0] = len(indx)
indx = df[(df['OSTs'] == 0) & (df['preds'] != 0)].index
cm.loc[0, 1] = len(indx)

indx = df[(df['OSTs'] == 1) & (df['preds'] == 1)].index
cm.loc[1, 1] = len(indx)
indx = df[(df['OSTs'] == 1) & (df['preds'] != 1)].index
cm.loc[1, 0] = len(indx)

indx = df[(df['OSTs'] == 2) & (df['preds'] == 1)].index
cm.loc[2, 1] = len(indx)
indx = df[(df['OSTs'] == 2) & (df['preds'] != 1)].index
cm.loc[2, 0] = len(indx)

indx = df[(df['OSTs'] == 3) & (df['preds'] == 1)].index
cm.loc[3, 1] = len(indx)
indx = df[(df['OSTs'] == 3) & (df['preds'] != 1)].index
cm.loc[3, 0] = len(indx)

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

table = ax.table(cellText=cm.values, colWidths=[0.25] * len(cm.columns),
                 rowLabels=cm.index,
                 colLabels=cm.columns,
                 cellLoc='center', rowLoc='center',
                 loc='center')
fig.tight_layout()

cm_total = confusion_matrix(y_true, test_res)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_total)
disp.plot(cmap="Blues", values_format='d')

plt.show()
fig.savefig(os.path.join(save_dir, 'conf.pdf'), bbox_inches='tight')  # plt.savefig

sen_total = cm_total[1, 1] / (cm_total[1, 1] + cm_total[1, 0])
spec_total = cm_total[0, 0] / (cm_total[0, 0] + cm_total[0, 1])

sen_1 = cm.loc[1, 1] / (cm.loc[1, 1] + cm.loc[1, 0])
sen_2 = cm.loc[2, 1] / (cm.loc[2, 1] + cm.loc[2, 0])
sen_3 = cm.loc[3, 1] / (cm.loc[3, 1] + cm.loc[3, 0])

print(args.target_comp)
print('Total sensitivity = ', sen_total)
print('Total specificity = ', spec_total)
print('######################################')
print('Class1 sensitivity = ', sen_1)
print('Class2 sensitivity = ', sen_2)
print('Class3 sensitivity = ', sen_3)



