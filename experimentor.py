import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from sklearn.cluster import KMeans

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# importing custom library
import DNN_models
import config

class Experimentor(object):
    def __init__(self, name=None, Xs=None, ys=None, studies=None, feature_selection=False, num_max_features=256):
        self.name = name
        self.Xs = Xs
        self.ys = ys
        self.studies = studies
        self.num_studies = len(self.Xs)
        self.feature_selection = feature_selection
        self.num_max_features = num_max_features
        
        self.X_trains = []
        self.y_trains = []
        self.X_tests = []
        self.y_tests = []

        self.history = []

        # DeepBioGen
        self.num_clusters = None
        self.num_GANs = None
        self.aug_rates = None
        self.aug_name = None
        self.X_train_augs = None
        self.y_train_augs = None

        # Set saving path
        self.result_path = os.path.join(os.getcwd(), 'results', self.name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.model_path = os.path.join(self.result_path, 'models')
        self.augmentation_path = os.path.join(self.result_path, 'augmentations')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.augmentation_path):
            os.makedirs(self.augmentation_path)

        # Leave one-study out cross-validation settings
        for i in range(self.num_studies):
            X_test = self.Xs[i]
            y_test = self.ys[i]
            Xs_train = self.Xs[0:i] + self.Xs[i+1:len(self.Xs)]
            ys_train = self.ys[0:i] + self.ys[i+1:len(self.ys)]
            X_train = np.concatenate(Xs_train)
            y_train = np.concatenate(ys_train)

            # Standiardization
            scaler = StandardScaler(with_mean=True, with_std=True) 
            scaler.fit(X_train)
            X_train = scaler.transform(X_train) 
            X_test = scaler.transform(X_test)

            # Feature selection
            if self.feature_selection:
                clf = ExtraTreesClassifier(n_estimators=500, criterion="entropy", random_state=0)
                clf = clf.fit(X_train, y_train)
                model = SelectFromModel(clf, prefit=True, max_features=self.num_max_features)
                X_train = model.transform(X_train)
                X_test = model.transform(X_test)
                print(f'Shape of train: {X_train.shape}')

            # Append to the main object
            self.X_trains.append(X_train)
            self.y_trains.append(y_train)
            self.X_tests.append(X_test)
            self.y_tests.append(y_test)

        # Classifiers
        self.classifier_names = ["SVM", "RF", "NN"]

        scoring='roc_auc'
        n_jobs=-1
        cv=5
        
        self.classifiers = [
            GridSearchCV(SVC(probability=True, random_state=0, cache_size=2048), param_grid=config.svm_hyper_parameters, cv=StratifiedKFold(cv, shuffle=True, random_state=0), scoring=scoring, n_jobs=n_jobs, verbose=1, ),
            GridSearchCV(RandomForestClassifier(n_jobs=n_jobs, random_state=0), param_grid=config.rf_hyper_parameters, cv=StratifiedKFold(cv, shuffle=True, random_state=0), scoring=scoring, n_jobs=n_jobs, verbose=1),
            GridSearchCV(MLPClassifier(random_state=0, max_iter=1000), param_grid=config.mlp_hyper_parameters, cv=StratifiedKFold(cv, shuffle=True, random_state=0), scoring=scoring, n_jobs=n_jobs, verbose=1)
        ]
    # End of def __init__()

    def ae(self, dims = [256], epochs=3000, batch_size=128, verbose=2, loss='mean_squared_error', patience=30, val_rate=0.2):
        # Time stamp
        start_time = time.time()

        # insert input shape into dimension list
        dims.insert(0, self.X_trains[0].shape[1])

        for i in range(self.num_studies):
            print(f"\n===== Learning AE for run {i+1}/{self.num_studies} =====\n")
            # filename for temporary model checkpoint
            modelName = 'AE_' + '_'.join([str(x) for x in dims]) + '_for_' + f'X_trains[{i}]'+ '.h5'
            modelName = os.path.join(self.result_path, modelName)

            if os.path.isfile(modelName):
                os.remove(modelName)

            callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(modelName, monitor='val_loss', mode='min', verbose=1, save_best_only=True)]

            # spliting the training set into the inner-train and the inner-test set (validation set)
            X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_trains[i], self.y_trains[i], test_size=val_rate, random_state=0, stratify=self.y_trains[i])
       
            # create autoencoder model
            self.autoencoder, self.encoder = DNN_models.autoencoder(dims)
            self.autoencoder.summary()

            # compile model
            optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)
            self.autoencoder.compile(optimizer=optimizer, loss=loss)

            # fit model
            self.history.append(self.autoencoder.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                                verbose=verbose, validation_data=(X_inner_test, X_inner_test)))
            # save loss progress
            self.saveLossProgress(run=i)

            # load best model
            self.autoencoder = load_model(modelName)
            layer_idx = int((len(self.autoencoder.layers) - 1) / 2)
            self.encoder = Model(self.autoencoder.layers[0].input, self.autoencoder.layers[layer_idx].output)

            # applying the learned encoder into the whole training and the test set.
            self.X_trains[i] = self.encoder.predict(self.X_trains[i])
            self.X_tests[i] = self.encoder.predict(self.X_tests[i])
        
        print(f"--- AE training finished in {round(time.time() - start_time, 2)} seconds ---")
    # End of def ae()

    # ploting loss progress over epochs
    def saveLossProgress(self, run):
        #print(self.history[run].history.keys())
        #print(type(self.history[run].history['loss']))
        #print(min(self.history[run].history['loss']))

        loss_collector, loss_max_atTheEnd = self.saveLossProgress_ylim(run=run)

        # save loss progress - train and val loss only
        figureName = os.path.join(self.result_path, f'run{run}.png')
        plt.ylim(min(loss_collector)*0.9, loss_max_atTheEnd * 2.0)
        plt.plot(self.history[run].history['loss'])
        plt.plot(self.history[run].history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'val loss'],
                   loc='upper right')
        plt.savefig(figureName)
        plt.close()

        if 'recon_loss' in self.history[run].history:
            figureName = self.prefix + self.data + '_' + str(self.seed) + '_detailed'
            plt.ylim(min(loss_collector) * 0.9, loss_max_atTheEnd * 2.0)
            plt.plot(self.history[run].history['loss'])
            plt.plot(self.history[run].history['val_loss'])
            plt.plot(self.history[run].history['recon_loss'])
            plt.plot(self.history[run].history['val_recon_loss'])
            plt.plot(self.history[run].history['kl_loss'])
            plt.plot(self.history[run].history['val_kl_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train loss', 'val loss', 'recon_loss', 'val recon_loss', 'kl_loss', 'val kl_loss'], loc='upper right')
            plt.savefig(figureName)
            plt.close()
    # End of def saveLossProgress()

    # supporting loss plot
    def saveLossProgress_ylim(self, run):
        loss_collector = []
        loss_max_atTheEnd = 0.0
        for hist in self.history[run].history:
            current = self.history[run].history[hist]
            loss_collector += current
            if current[-1] >= loss_max_atTheEnd:
                loss_max_atTheEnd = current[-1]
        return loss_collector, loss_max_atTheEnd
    # End of def saveLossProgress_ylim()

    def classify(self):
        # Time stamp
        start_time = time.time()
        
        res_columns = ["ValidationOn", "Clf", "AUROC", "AUPRC", "ACC", "REC", "PRE", "F1"]
        res = []

        for i in range(self.num_studies):
            print(f'\nRun {i}')
            for clf, clf_name in zip(self.classifiers, self.classifier_names):
                clf.fit(self.X_trains[i], self.y_trains[i])
                print(f'Best parameter on training set: {clf.best_params_}')
                y_pred = clf.predict(self.X_tests[i])
                y_prob = clf.predict_proba(self.X_tests[i])

                precisions, recalls, _ = precision_recall_curve(self.y_tests[i], y_prob[:, 1])

                # Performance Metrics : AUROC, AUPRC, ACC, Recall, Precision, F1
                auroc = round(roc_auc_score(self.y_tests[i], y_prob[:, 1]), 3)
                auprc = round(auc(recalls, precisions), 3)
                acc = round(accuracy_score(self.y_tests[i], y_pred), 3)
                rec = round(recall_score(self.y_tests[i], y_pred), 3)
                pre = round(precision_score(self.y_tests[i], y_pred), 3)
                f1 = round(f1_score(self.y_tests[i], y_pred), 3)

                res.append([self.studies[i], clf_name, auroc, auprc, acc, rec, pre, f1])

        res_df = pd.DataFrame(res, columns=res_columns)

        # Save results
        res_df.to_csv(os.path.join(self.result_path, 'res.csv'), index=False)
        res_df.groupby(by="Clf").mean().round(3).to_csv(os.path.join(self.result_path, 'res_avg.csv'))
        res_df.groupby(by="Clf").std().round(3).to_csv(os.path.join(self.result_path, 'res_std.csv'))

        print(f"--- Classified in {round(time.time() - start_time, 2)} seconds ---")
    # End of def classify()

    def classify_with_DBG(self):
        # Time stamp
        start_time = time.time()
        
        res_columns = ["ValidationOn", "AugRate", "Clf", "AUROC", "AUPRC", "ACC", "REC", "PRE", "F1"]
        res = []

        for i in range(self.num_studies):
            for aug_rate in self.aug_rates:
                print(f'\nRun {i}, Aug_rate: {aug_rate}')
                for clf, clf_name in zip(self.classifiers, self.classifier_names):
                    
                    # Get best params from training data
                    clf.fit(self.X_trains[i], self.y_trains[i])
                    print(f'Best parameter on training set: {clf.best_params_}')
                    best_est = copy.deepcopy(clf.best_estimator_)

                    # Fit on augmented training data
                    best_est.fit(self.X_train_augs[i][aug_rate], self.y_train_augs[i][aug_rate])

                    # Test on test data
                    y_pred = best_est.predict(self.X_tests[i])
                    y_prob = best_est.predict_proba(self.X_tests[i])

                    precisions, recalls, _ = precision_recall_curve(self.y_tests[i], y_prob[:, 1])

                    # Performance Metrics : AUROC, AUPRC, ACC, Recall, Precision, F1
                    auroc = round(roc_auc_score(self.y_tests[i], y_prob[:, 1]), 3)
                    auprc = round(auc(recalls, precisions), 3)
                    acc = round(accuracy_score(self.y_tests[i], y_pred), 3)
                    rec = round(recall_score(self.y_tests[i], y_pred), 3)
                    pre = round(precision_score(self.y_tests[i], y_pred), 3)
                    f1 = round(f1_score(self.y_tests[i], y_pred), 3)

                    res.append([self.studies[i], aug_rate, clf_name, auroc, auprc, acc, rec, pre, f1])

        res_df = pd.DataFrame(res, columns=res_columns)

        # Save results
        res_df.to_csv(os.path.join(self.result_path, 'res.csv'), index=False)
        res_df.groupby(by="Clf").mean().round(3).to_csv(os.path.join(self.result_path, 'res_avg.csv'))
        res_df.groupby(by="Clf").std().round(3).to_csv(os.path.join(self.result_path, 'res_std.csv'))

        print(f"--- Classified in {round(time.time() - start_time, 2)} seconds ---")
    # End of def classify()

    def visualize_featurewise_wss(self, run, X_train):
        # Helper func
        def calculate_WSS(points, kmax):
            sse = []
            for k in range(1, kmax+1):
                kmeans = KMeans(n_clusters = k, random_state=1).fit(points)
                centroids = kmeans.cluster_centers_
                pred_clusters = kmeans.predict(points)
                curr_sse = 0
                # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
                for i in range(len(points)):
                    curr_center = centroids[pred_clusters[i]]
                    curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
                sse.append(curr_sse)
            return sse
        fig = plt.figure()
        plt.plot([k for k in range(1, 10+1)], calculate_WSS(X_train.T, 10))
        plt.savefig(os.path.join(self.result_path, f'featurewise_WSS_run{str(run)}.png'))
    # End of def visualize_featurewise_wss()

    def visualize_samplewise_wss(self, run, X_train):
        # Helper func
        def calculate_WSS(points, kmax):
            sse = []
            for k in range(1, kmax+1):
                kmeans = KMeans(n_clusters = k, random_state=1).fit(points)
                centroids = kmeans.cluster_centers_
                pred_clusters = kmeans.predict(points)
                curr_sse = 0
                # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
                for i in range(len(points)):
                    curr_center = centroids[pred_clusters[i]]
                    curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
                sse.append(curr_sse)
            return sse
        fig = plt.figure()
        plt.plot([k for k in range(1, 10+1)], calculate_WSS(X_train, 10))
        plt.savefig(os.path.join(self.result_path, f'samplewise_WSS_run{str(run)}.png'))
    # End of def visualize_samplewise_wss()

    # Compute feature-wise and sample-wise wss and save figs under result path
    def viz_wss(self):
        for i in range(self.num_studies):
            # Calculate WSSs
            self.visualize_featurewise_wss(run=i, X_train=self.X_trains[i])
            self.visualize_samplewise_wss(run=i, X_train=self.X_trains[i])