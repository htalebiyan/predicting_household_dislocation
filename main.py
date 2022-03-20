import numpy as np
import pandas as pd
import random
# from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.dummy import DummyClassifier
# from sklearn.metrics import accuracy_score, recall_score, precision_score, \
#     confusion_matrix, precision_recall_curve, roc_curve, auc
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings

warnings.filterwarnings("ignore")


def clean_data(df_original):
    df = df_original.copy(deep=True)
    df[['deck', 'num', 'side']] = df['Cabin'].fillna('Z/-1/Z').str.split('/', 0, expand=True)
    df.drop(['Cabin', 'Name'], axis=1, inplace=True)
    money_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[money_cols] = df[money_cols].fillna(0)
    cat_na_cols = ['HomePlanet', 'Destination']
    df[cat_na_cols] = df[cat_na_cols].fillna('Z')
    df['CryoSleep'] = df['CryoSleep'].fillna(bool(random.getrandbits(1)))
    df['VIP'] = df['VIP'].fillna(bool(random.getrandbits(1)))
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    # df = df[df['country'] == 'USA']
    # excluded_columns = []
    # df = df[[x for x in df.columns if x not in excluded_columns]]
    # if 'compliance' in df.columns.tolist():
    #     df = df[df['compliance'].notna()]
    # else:
    #     print('no compliance column')
    # df['hearing_date'].fillna(df['ticket_issued_date'], inplace=True)
    # for col in ['ticket_issued_date', 'hearing_date']:
    #     days_since_epoch = pd.to_datetime(df[col]) - pd.datetime(1970, 1, 1)
    #     df[col] = days_since_epoch.dt.days
    # int_cols = ['admin_fee', 'state_fee', 'discount_amount', 'clean_up_cost', 'fine_amount',
    #             'ticket_issued_date', 'hearing_date']
    # df[int_cols] = df[int_cols].astype('float')
    # df = df.loc[df['zip_code'].str.contains('[0-9]{5}').fillna(False)]
    # df = df.loc[~df['zip_code'].str.contains('`')]
    # df['zip_code'] = np.where(df['zip_code'].str.len() == 5, df['zip_code'], df['zip_code'].str[:5]).astype('float')
    # # df['grafitti_status'] = df['grafitti_status'].replace(np.nan, 'ng', regex=True)
    # df = df.dropna()
    return df


def scale_and_encode(df, scale_cols, encode_cals, fitted_obj=None):
    if fitted_obj:
        scaled_columns = fitted_obj[0].transform(df[scale_cols])
        encoded_columns = fitted_obj[1].transform(df[encode_cals])
        return np.concatenate([scaled_columns, encoded_columns], axis=1)
    else:
        scaler = MinMaxScaler()
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        scaled_columns = scaler.fit_transform(df[scale_cols])
        encoded_columns = encoder.fit_transform(df[encode_cals])
        return np.concatenate([scaled_columns, encoded_columns], axis=1), scaler, encoder


train_data = pd.read_csv('population_raw.csv', delimiter=',', index_col=None).dropna(axis=1, how='all')
aaa, bbb = train_test_split(train_data, test_size=0.2, random_state=1)
# train_data_clean = clean_data(train_data)
# # train_data_clean.isna().any()
# X = train_data_clean.loc[:, [x for x in train_data_clean.columns.tolist() if x != 'Transported']]
# y = train_data_clean.loc[:, 'Transported']
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# scale_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'num']
# encode_cals = [x for x in X_train.columns if x not in scale_cols]
# X_train, scaler, encoder = scale_and_encode(X_train, scale_cols, encode_cals, fitted_obj=None)
# X_test = scale_and_encode(X_test, scale_cols, encode_cals, fitted_obj=[scaler, encoder])
#
# # clf = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
# # clf = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
# # clf = LogisticRegression(solver='liblinear', max_iter=1000).fit(X_train, y_train)
# # clf = SVC(C=0.01, gamma=.01).fit(X_train, y_train)
# # clf = GaussianNB().fit(X_train, y_train)
# # clf = DecisionTreeClassifier(max_depth=10).fit(X_train, y_train)
# # clf = RandomForestClassifier(n_estimators=5, max_features=10).fit(X_train, y_train)
# clf = GradientBoostingClassifier(learning_rate=1, max_depth=3, min_samples_leaf=5, min_samples_split=50,
#                                  n_estimators=15).fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)
# print('Accuracy:', accuracy_score(y_pred, y_test))
# print('Recall:', recall_score(y_test, y_pred))
# print('precision:', precision_score(y_test, y_pred))
# try:
#     y_scores = clf.decision_function(X_test)
#     fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores)
#     print('AUC:', auc(fpr_lr, tpr_lr))
#
#     precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
#     plt.figure()
#     plt.plot(precision, recall, label='Precision-Recall Curve')
#     plt.xlabel('Precision', fontsize=16)
#     plt.ylabel('Recall', fontsize=16)
#     plt.show()
#     fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores)
#     roc_auc_lr = auc(fpr_lr, tpr_lr)
#
#     plt.figure()
#     plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
#     plt.xlabel('False Positive Rate', fontsize=16)
#     plt.ylabel('True Positive Rate', fontsize=16)
#     plt.title('ROC curve', fontsize=16)
#     plt.legend(loc='lower right', fontsize=13)
#     plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
#     plt.show()
# except:
#     print('no AUC')
#
# # grid_values = {'n_neighbors': [1, 2, 3, 4, 5, 8, 10, 12, 16, 20]}
# # grid_values = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
# # grid_values = {'gamma': [0.01, 0.1, 1, 10], 'C': [0.01, 0.1, 1, 10]}
# # grid_values = {'max_depth': [2, 3, 4, 5], 'min_samples_split': [2, 5, 10, 50], 'min_samples_leaf': [1, 5, 10, 50]}
# # grid_values = {'max_depth': [2, 3, 4, 5], 'min_samples_split': [2, 5, 10, 50], 'min_samples_leaf': [1, 5, 10, 50],
# #                'n_estimators': [5, 10, 15], 'max_features': [5, 8, 12, 16]}
# grid_values = {'max_depth': [2, 3, 4, 5], 'min_samples_split': [2, 5, 10, 50], 'min_samples_leaf': [1, 5, 10, 50],
#                'n_estimators': [10, 12, 15], 'learning_rate': [0.2, 0.5, 0.7, 0.9, 1, 1.2]}
#
# scoring = ['accuracy']  # ['recall', 'precision', 'accuracy', 'f1_macro', 'f1_micro', "roc_auc"]
# grid_clf = GridSearchCV(clf, param_grid=grid_values, cv=10, refit='accuracy', scoring=scoring,
#                         return_train_score=True).fit(X_train, y_train)
# print('Grid best parameter : ', grid_clf.best_params_)
# print('Grid best score: ', grid_clf.best_score_)
# y_pred = grid_clf.predict(X_test)
# print('Accuracy:', accuracy_score(y_pred, y_test))
# print('Recall:', recall_score(y_pred, y_test))
# print('Precision:', precision_score(y_test, y_pred))
# results = grid_clf.cv_results_
#
# # parameter_oi = list(grid_values.keys())[0]
# # plt.figure(figsize=(8, 8))
# # plt.title("GridSearchCV", fontsize=16)
# # plt.xlabel(parameter_oi)
# # plt.ylabel("Score")
# # ax = plt.gca()
# # ax.set_ylim(0.3, 1)
# # X_axis = np.array(results["param_" + parameter_oi].data, dtype=float)
# # for scorer, color in zip(sorted(scoring), ["b", "k", "g", "r", "m", "c"]):
# #     for sample, style in (("train", "--"), ("test", "-")):
# #         sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
# #         sample_score_std = results["std_%s_%s" % (sample, scorer)]
# #         ax.fill_between(X_axis, sample_score_mean - sample_score_std, sample_score_mean + sample_score_std,
# #                         color=color, alpha=0.1 if sample == "test" else 0)
# #         ax.plot(X_axis, sample_score_mean, style, alpha=1 if sample == "test" else 0.7,
# #                 color=color, label="%s (%s)" % (scorer, sample))
# #     best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
# #     best_score = results["mean_test_%s" % scorer][best_index]
# #     # # Plot a dotted vertical line at the best score for that scorer marked by x
# #     # ax.plot([X_axis[best_index], ] * 2, [0, best_score], linestyle="-.", marker="x",
# #     #         markeredgewidth=3, ms=8, color=color)
# #     # Annotate the best score for that scorer
# #     ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))
# #     # ax.set_xscale('log')
# # plt.legend(loc="best")
# # plt.grid(False)
# # plt.show()
#
# test = pd.read_csv('test.csv', index_col=0)
# test_cleaned = clean_data(test)
# test_conv = scale_and_encode(test_cleaned, scale_cols, encode_cals, fitted_obj=[scaler, encoder])
# y_pred = grid_clf.predict(test_conv)
# output = pd.Series(y_pred, index=test_cleaned.index)
# index_removed = [x for x in test.index if x not in test_cleaned.index]
# temp_series = pd.Series([bool(random.getrandbits(1)) for x in index_removed], index=index_removed)
# output = pd.concat([output, temp_series])
# output.to_csv('output.csv', index_label='PassengerId', header=['Transported'])
