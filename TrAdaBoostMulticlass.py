import numpy as np
import pandas as pd
from sklearn.base import clone
import sys
import operator

class TradaboostClassifier(object):
    def __init__(self, learner, epoches):
        self.learner = learner
        self.epoches = epoches
        self.models = None
        self.alfas = None
        self.K = 0
        self.K_extended = None
        
    def indicator(self, y_predict, y):
        return np.array([1 if a != b else 0 for a,b in zip(y_predict, y)])

    def fit(self, same_X, diff_X, same_y, diff_y):
        assert len(same_X) == len(same_y), "same dist data size mismatch"
        assert len(diff_X) == len(diff_y), "diff dist data size mismatch"
        assert isinstance(same_X, pd.DataFrame), "same_X is not DataFrame"
        assert isinstance(diff_X, pd.DataFrame), "diff_X is not DataFrame"
        assert set(same_y) == set(diff_y), "Label are not equal"
        try:
            self.models = []
            l_all = len(same_y) + len(diff_y)
            w0 = np.ones(l_all)
            X = pd.concat([same_X, diff_X]).reset_index(drop=True)
            y = np.array(pd.concat([pd.Series(same_y),pd.Series(diff_y)]))
            n = len(diff_y) #len diff data
            m = len(same_y) #len source train
            self.K_extended = set(same_y)
            self.K = len(self.K_extended) #num classes
            
            wt = w0
            alfa = []
            for i in range(self.epoches):
                wt = wt / np.sum(wt)
                model = clone(self.learner).fit(X, y, sample_weight=wt)
                self.models.append(model)
                y_predict = model.predict(X)
                loss = self.indicator(y_predict, y)
                loss_same = loss[:m]
                loss_diff = loss[m+1:]
                eta = ((loss_diff * wt[m+1:]) / wt[m+1:].sum()).sum()

                if eta > 1-1/self.K: #old 0.5
                    #print(f"Attenzione soglia eta:{eta} ricalibrata step: {i}")
                    eta = 1-1/self.K #old 0.5
                if eta == 0:
                    print(f"Raggiunto il massimo interrompo a {i}")
                    self.epoches = i
                    break  
                alfa_t = np.log10((1-eta)/eta) + np.log10(self.K-1)
                beta_t = np.exp(alfa_t * loss_diff)
                beta = np.exp(np.log10(1/(1+np.sqrt(2*np.log(m)/self.epoches))) * loss_same)
                wt[:m] = wt[:m] * self.K * (1-eta) * beta
                wt[m+1:] = wt[m+1:] * beta_t

                alfa.append(alfa_t)
            self.alfas = alfa
            
        except Exception as e:
            print('fit: Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            raise NameError("fit error")


    def predict(self, X):
        
        y_pred = [list(model.predict(X)) for model in self.models]
        df = pd.DataFrame(y_pred).T
        w_vect = np.array(self.alfas)
        main_array = np.array(df)
        df_ones = np.ones(main_array.shape)
        multiplied = df_ones*w_vect
        df_multiplied = pd.DataFrame(multiplied)
        dict_y_pred_final = {}
        for label in self.K_extended:
            df_label = df[df==label].replace(label,1)
            dict_y_pred_final[label] = df_multiplied[(df_label == 1)].sum(axis=1)
        df_final = pd.DataFrame(dict_y_pred_final)
        return df_final.idxmax(axis=1)

    def score(self, X, y):
        return 1-(self.indicator(self.predict(X),y).sum()/len(y))
    
    def get_params(self):
        return self.models
