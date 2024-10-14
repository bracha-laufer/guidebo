from sklearn.metrics import confusion_matrix
import numpy as np

class DPDiff():

    def __init__(self, name='DP_diff', weight_vector=None, \
                 threshold=0.5, attribute_index=1, good_value=1):

        self.weight_vector = weight_vector
        self.threshold = threshold
        self.isFairnessLoss = True
        self.idx = attribute_index
        self.good_value = good_value

    def _DP_np(self, y_true, y_pred):
        epsilon = 1e-7
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if(self.good_value):
            pr = (tp.astype(np.float32)+fp) / (tp+fp+tn+fn)
        else:
        	pr = (tn.astype(np.float32)+fn) / (tp+fp+tn+fn)  
        return (pr+epsilon)


    def evaluate(self, y_true, y_pred):
        a = y_true[:,[self.idx]]
        y_true = y_true[:,[0]]

        y_pred = np.rint(y_pred)

        y_pred_0 = y_pred[a==0]
        y_true_0 = y_true[a==0]

        y_pred_1 = y_pred[a==1]
        y_true_1 = y_true[a==1]
        
        DP_0 = self._DP_np(y_true_0, y_pred_0)
        DP_1 = self._DP_np(y_true_1, y_pred_1)
        
        diff = abs(DP_0 - DP_1)
        return (diff)

