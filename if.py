import sys
import os

#Use if working on Colab
#from google.colab import drive
#drive.mount('/content/drive')
#PATH = '/content/drive/My Drive/PPM_Stability/'

#If working locally
PATH = os.getcwd()
sys.path.append(PATH)

import EncoderFactory
#from DatasetManager_for_colab import DatasetManager
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np
from scipy import stats
import math

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

import time
import os
import sys
from sys import argv
import pickle
from collections import defaultdict, Counter
import random
import joblib

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import lime
import lime.lime_tabular
from lime import submodular_pick;

import shap

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

def imp_df(column_names, importances):
        df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
        return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title, num_feat):
        imp_df.columns = ['feature', 'feature_importance']
        b= sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df.head(num_feat), orient = 'h', palette="Blues_r")


def generate_global_explanations(train_X, train_Y, cls, feature_combiner):
    print("The number of testing instances is ", len(train_Y))
    print("The total number of columns is", train_X.shape[1]);
    print("The total accuracy is ", cls.score(train_X, train_Y));

    sns.set(rc={'figure.figsize': (10, 10), "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18})
    sns.set
    feat_names = feature_combiner.get_feature_names()
    base_imp = imp_df(feat_names, cls.feature_importances_)
    base_imp.head(15)
    var_imp_plot(base_imp, 'Feature importance using XGBoost', 15)
    return base_imp


from lime import submodular_pick


def generate_lime_explanations(explainer, test_xi, cls, test_y, submod=False, test_all_data=None, max_feat=10):
    # print("Actual value ", test_y)
    exp = explainer.explain_instance(test_xi,
                                     cls.predict_proba, num_features=max_feat, labels=[0, 1])

    return exp

    if submod == True:
        sp_obj = submodular_pick.SubmodularPick(explainer, test_all_data, cls.predict_proba,
                                                sample_size=20, num_features=num_features, num_exps_desired=4)
        [exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_obj.sp_explanations];


def create_samples(shap_explainer, iterations, row, features, top=None):
    length = len(features)

    exp = []
    rel_exp = []

    for j in range(iterations):
        # Generate shap values for row
        shap_values = shap_explainer.shap_values(row)

        # Map SHAP values to feature names
        importances = []
        if type(shap_explainer) == shap.explainers.kernel.KernelExplainer:
            for i in range(length):
                feat = features[i]
                shap_val = shap_values[0][i]
                abs_val = abs(shap_values[0][i])
                entry = (feat, shap_val, abs_val)
                importances.append(entry)

        elif type(shap_explainer) == shap.explainers.tree.TreeExplainer:
            for i in range(length):
                feat = features[i]
                shap_val = shap_values[0][i]
                abs_val = abs(shap_values[0][i])
                entry = (feat, shap_val, abs_val)
                importances.append(entry)

        elif type(shap_explainer) == shap.explainers.deep.DeepExplainer:
            for i in range(length):
                feat = features[i]
                shap_val = shap_values[0][0][i]
                abs_val = abs(shap_values[0][0][i])
                entry = (feat, shap_val, abs_val)
                importances.append(entry)

        # Sort features by influence on result
        importances.sort(key=lambda tup: tup[2], reverse=True)

        # Create list of all feature
        exp.append(importances)

        # Create list of most important features
        rel_feat = []
        if top != None:
            for i in range(top):
                feat = importances[i]
                if feat[2] > 0:
                    rel_feat.append(feat)

            rel_exp.append(rel_feat)
        else:
            rel_exp = exp

    return exp, rel_exp


def generate_distributions(explainer, features, test_x, bin_min=-1, bin_max=1, bin_width=0.05):
    # generate shap values for entire test set
    shap_values = explainer.shap_values(test_x, check_additivity=False)
    shap_val_feat = np.transpose(shap_values)
    feats = np.transpose(test_x)

    shap_distribs = []

    # For each feature
    for i in range(len(features)):
        print(i + 1, "of", len(features), "features")
        shap_vals = shap_val_feat[i]

        # create bins based on shap value ranges
        bins = np.arange(bin_min, bin_max, bin_width)

        feat_vals = []
        for sbin in range(len(bins)):
            nl = []
            feat_vals.append(nl)

        # place relevant feature values into each bin
        for j in range(len(shap_vals)):
            val = shap_vals[j]
            b = 0
            cur_bin = bins[b]
            idx = b

            while val > cur_bin and b < len(bins) - 1:
                idx = b
                b += 1
                cur_bin = bins[b]

            feat_vals[idx].append(feats[i][j])

        # Find min and max values for each shap value bin
        mins = []
        maxes = []

        for each in feat_vals:
            if each != []:
                mins.append(min(each))
                maxes.append(max(each))

        # Create dictionary with list of bins and max and min feature values for each bin
        feat_name = features[i]

        feat_dict = {'Feature Name': feat_name}
        for each in feat_vals:
            if each != []:
                mins.append(min(each))
                maxes.append(max(each))
            else:
                mins.append(None)
                maxes.append(None)

        feat_dict['bins'] = bins
        feat_dict['mins'] = mins
        feat_dict['maxes'] = maxes

        shap_distribs.append(feat_dict)

    return shap_distribs

dataset_ref = "production"
params_dir = PATH + "params"
results_dir = "results"
bucket_method = "single"
cls_encoding = "agg"
cls_method = "xgboost"

gap = 1
n_iter = 1

method_name = "%s_%s"%(bucket_method, cls_encoding)

generate_samples = False
generate_lime = False
generate_kernel_shap = False
generate_model_shap = True

sample_size = 2
exp_iter = 10
#max_feat = 10

dataset_ref_to_datasets = {
    #"bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(5,6)],
    "bpic2017" : ["bpic2017_accepted"],
    "bpic2012" : ["bpic2012_accepted"],
    #"insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases_1"],#, "sepsis_cases_2", "sepsis_cases_4"]
    "production": ["production"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]

datasets

if generate_model_shap:
    for dataset_name in datasets:

        dataset_manager = DatasetManager(dataset_name)

        for ii in range(n_iter):
            num_buckets = range(len([name for name in os.listdir(
                os.path.join(PATH, '%s/%s_%s/models' % (dataset_ref, cls_method, method_name)))]))

            all_shap_changes = []
            all_lens = []
            all_probas = []
            all_case_ids = []

            for bucket in list(num_buckets):
                bucketID = bucket + 1
                print('Bucket', bucketID)

                # import everything needed to sort and predict
                feat_comb_path = os.path.join(PATH,
                                              "%s/%s_%s/bucketers_and_encoders/feature_combiner_bucket_%s.joblib" % (
                                              dataset_ref, cls_method, method_name, bucketID))
                cls_path = os.path.join(PATH, "%s/%s_%s/models/cls_bucket_%s.joblib" % (
                dataset_ref, cls_method, method_name, bucketID))
                cls = joblib.load(cls_path)
                feature_combiner = joblib.load(feat_comb_path)

                # import data for bucket
                X_test_path = os.path.join(PATH, "%s/%s_%s/test_data/bucket_%s_prefixes.pickle" % (
                dataset_ref, cls_method, method_name, bucketID))
                Y_test_path = os.path.join(PATH, "%s/%s_%s/test_data/bucket_%s_labels.pickle" % (
                dataset_ref, cls_method, method_name, bucketID))

                with open(X_test_path, 'rb') as f:
                    dt_test_bucket = pickle.load(f)
                with open(Y_test_path, 'rb') as f:
                    test_y = pickle.load(f)

                # import previously identified samples
                tn_path = os.path.join(PATH, "%s/%s_%s/samples/true_neg_bucket_%s_.pickle" % (
                dataset_ref, cls_method, method_name, bucketID))
                tp_path = os.path.join(PATH, "%s/%s_%s/samples/true_pos_bucket_%s_.pickle" % (
                dataset_ref, cls_method, method_name, bucketID))
                fn_path = os.path.join(PATH, "%s/%s_%s/samples/false_neg_bucket_%s_.pickle" % (
                dataset_ref, cls_method, method_name, bucketID))
                fp_path = os.path.join(PATH, "%s/%s_%s/samples/false_pos_bucket_%s_.pickle" % (
                dataset_ref, cls_method, method_name, bucketID))

                sample_instances = []

                with open(tn_path, 'rb') as f:
                    tn_list = pickle.load(f)
                with open(tp_path, 'rb') as f:
                    tp_list = pickle.load(f)
                with open(fn_path, 'rb') as f:
                    fn_list = pickle.load(f)
                with open(fp_path, 'rb') as f:
                    fp_list = pickle.load(f)

                # save results to a list
                sample_instances.append(tn_list)
                sample_instances.append(tp_list)
                sample_instances.append(fn_list)
                sample_instances.append(fp_list)

                # Create explainer and compile relevant information
                tree_explainer = shap.TreeExplainer(cls)
                test_x = feature_combiner.fit_transform(dt_test_bucket)
                feat_list = feature_combiner.get_feature_names()
                type_list = ['True Negatives', 'True Positives', 'False Negatives', 'False Positives']

                # We want to get the top 10% of features in each bucket
                max_feat = round(len(feat_list) * 0.1)

                print("Generating distributions for bucket")
                distribs = generate_distributions(tree_explainer, feat_list, test_x)

                for i_type in range(len(sample_instances)):
                    changes = []
                    probas = []
                    nr_events = []
                    case_ids = []

                    for n in range(len(sample_instances[i_type])):
                        print("Category %s of %s. Instance %s of %s" % (
                        i_type + 1, len(sample_instances), n + 1, len(sample_instances[i_type])))
                        instance = sample_instances[i_type][n]

                        ind = instance['predicted']
                        case_ids.append(instance['caseID'])
                        p1 = instance['proba']
                        probas.append(p1)
                        nr_events.append(instance['nr_events'])
                        input_ = instance['input']

                        test_x_group = feature_combiner.fit_transform(input_)

                        print("Creating explanations")
                        exp, rel_exp = create_samples(tree_explainer, exp_iter, test_x_group, feat_list, top=max_feat)

                        features = []
                        shap_vals = []

                        print("Identifying relevant features")
                        for explanation in rel_exp:
                            features.extend([feat[0] for feat in explanation])
                            shap_vals.extend([feat for feat in explanation])

                        counter = Counter(features).most_common(max_feat)

                        feats = [feat[0] for feat in counter]

                        # Choose relevant feature value distribution for each feature based on the current SHAP value
                        rel_feats = []
                        for feat in feats:
                            vals = [i[1] for i in shap_vals if i[0] == feat]
                            val = np.mean(vals)
                            rel_feats.append((feat, val))

                        intervals = []
                        for item in rel_feats:
                            feat = item[0]
                            val = item[1]

                            print("Creating distribution for feature", rel_feats.index(item) + 1, "of", len(rel_feats))

                            n = feat_list.index(feat)
                            feat_dict = distribs[n]

                            if feat_dict['Feature Name'] != feat:
                                for each in distribs:
                                    if feat_dict['Feature Name'] == feat:
                                        feat_dict = each

                            bins = feat_dict['bins']
                            mins = feat_dict['mins']
                            maxes = feat_dict['maxes']

                            i = 0
                            while val > bins[i] and i < len(bins) - 1:
                                idx = i
                                i += 1
                            # print (i)
                            if mins[i] != None:
                                min_val = mins[i]
                                max_val = maxes[i]
                            else:
                                j = i
                                while mins[j] == None and j > 0:
                                    min_val = mins[j - 1]
                                    max_val = maxes[j - 1]
                                    j = j - 1

                            interval = max_val - min_val
                            if interval == 0:
                                interval = 1

                            index = feat_list.index(feat)
                            int_min = max_val
                            int_max = max_val + interval
                            intervals.append((feat, index, int_min, int_max))

                        diffs = []
                        # Create and test perturbed instances, and record the differences between the original
                        # and perturbed prediction proabilities
                        for iteration in range(exp_iter):
                            print("Pertubing - Run", iteration + 1)
                            alt_x = np.copy(test_x_group)
                            # print("original:", alt_x)
                            for each in intervals:
                                new_val = random.uniform(each[2], each[3])
                                alt_x[0][each[1]] = new_val
                            p2 = cls.predict_proba(alt_x)[0][ind]
                            diff = p1 - p2
                            diffs.append(diff)

                        changes.append(np.mean(diffs))

                        instance['shap_fid_change'] = diffs

                    all_shap_changes.extend(changes)
                    all_lens.extend(nr_events)
                    all_probas.extend(probas)
                    all_case_ids.extend(case_ids)

                # Save dictionaries updated with scores
                with open(tn_path, 'wb') as f:
                    pickle.dump(sample_instances[0], f)
                with open(tp_path, 'wb') as f:
                    pickle.dump(sample_instances[1], f)
                with open(fn_path, 'wb') as f:
                    pickle.dump(sample_instances[2], f)
                with open(fp_path, 'wb') as f:
                    pickle.dump(sample_instances[3], f)