from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, matthews_corrcoef
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold 
machine_learners = ["Random Forest S", "KNN", "GB"]
ml_list={
    "Random Forest S":OneVsRestClassifier(RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1), n_jobs=-1),
    "KNN":KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', n_jobs=-1, n_neighbors=31, p=2, weights='distance'),
    "GB":OneVsRestClassifier(HistGradientBoostingClassifier(random_state=1), n_jobs=-1)
    }
main_dataset=pd.read_csv("../FS.csv")
Non_IoT = main_dataset[main_dataset['Label'].isin(['MacBook/Iphone', 'IPhone', 'Android Phone', 'MacBook', 'Laptop', 'unknown maybe cam', 'Samsung Galaxy Tab', 'TPLink Router Bridge LAN (Gateway)'])].index
main_dataset.drop(Non_IoT, inplace=True)
indexNames = main_dataset[ ((main_dataset['payload_l'].isin([0])) | (main_dataset['DNS'].isin([1]))) ].index
main_dataset.drop(indexNames, inplace=True)
datasets = ["../IoTSentinel_FS.csv", "../IoTDevID_FS.csv", "../IoTSense_FS.csv"]
packet_labels = {}
Other = []
No_major = []
Same_correct = []
Same_incorrect = []
High_conf_correct = []
Major_correct = []
guess_packets = []
One_confidence = []
Zero_confidence = []
correct_packets = []
for index in range(len(datasets)):
    df = pd.read_csv(datasets[index])
    df.drop(Non_IoT , inplace=True)
    df.drop(indexNames , inplace=True)
    if "MAC" in df.columns:
            del df["MAC"]
    X = df.loc[:, df.columns != "Label"]
    df["Label"] = df["Label"].astype('category')
    Y=df["Label"].cat.codes
    names = dict(enumerate(df["Label"].cat.categories))
    X_train, X_test = X[:int(len(X)*0.7)], X[int(len(X)*0.7):]
    y_train, y_test = Y[:int(len(X)*0.7)], Y[int(len(X)*0.7):]
    clf = ml_list[machine_learners[index]]
    clf.fit(X_train, y_train)
    predict =clf.predict(X_test)
    f_1=f1_score(y_test, predict,average= "weighted")
    accuracy = (accuracy_score(y_test, predict))  
    matt = matthews_corrcoef(y_test, predict)
    proba = clf.predict_proba(X_test)
    print(str(machine_learners[index])+": Accuracy is {} and F1 is {} and Matt Coeff is {}".format(accuracy, f_1, matt))
    print(classification_report(y_test, predict, labels=Y.unique(), target_names=df["Label"].unique()))
    point = 0
    for label in proba:
        #if np.isnan(np.sum(label)) or max(label) < 0.7:
        #    pass
        key = point
        point += 1
        guess = clf.classes_[np.argmax(label, axis=0)]
        if key not in packet_labels:
            packet_labels[key] = [(guess, max(label))]
        else:
            packet_labels[key].append((guess, max(label)))
for k,v in packet_labels.items():
    lab_set = set([z[0] for z in v])
    packet_index = k+int(len(X)*0.7)
    ground_turth = main_dataset.iloc[packet_index]['Label']
    if len(lab_set) == 3:
        most_c = max(v[0][1], v[1][1], v[2][1])
        if most_c == v[0][1]:
            ind = 0
        elif most_c == v[1][1]:
            ind = 1
        else:
            ind = 2
        classifier_lab = names[v[ind][0]]
        if ground_turth == classifier_lab:
            High_conf_correct.append((ground_turth, classifier_lab, v[ind][1]))
        elif ground_turth == names[v[0][0]]:
            One_confidence.append((ground_turth,names[v[0][0]], v[0][1], classifier_lab, v[ind][1])) 
        elif ground_turth == names[v[1][0]]:
            One_confidence.append((ground_turth,names[v[1][0]], v[1][1], classifier_lab, v[ind][1])) 
        elif ground_turth == names[v[2][0]]:
            One_confidence.append((ground_turth,names[v[2][0]], v[2][1], classifier_lab, v[ind][1])) 
        else:
            Zero_confidence.append((ground_turth, names[v[ind][0]], v[ind][1])) 
    elif len(lab_set) == 2:
        if v[0][0] == v[1][0]:
            majority = names[v[0][0]]
            majority_c = v[0][1]
            other = v[2]
        elif v[0][0] == v[2][0]:
            majority = names[v[0][0]]
            majority_c = v[0][1]
            other = v[1]
        else:
            majority = names[v[1][0]]
            majority_c = v[1][1]
            other = v[0]
        if ground_turth == majority:
            Major_correct.append((ground_turth, majority, majority_c, names[other[0]], other[1]))
        elif ground_turth == names[other[0]]:
            Other.append((ground_turth,names[other[0]], other[1], majority, majority_c)) 
        else:
            No_major.append((ground_turth, majority, majority_c)) 
    elif len(lab_set) == 1:
        most_c = max(v[0][1], v[1][1], v[2][1])
        min_c = min(v[0][1], v[1][1], v[2][1])
        if ground_turth == names[v[0][0]]:
            Same_correct.append((ground_turth, most_c, min_c))
        else:
            Same_incorrect.append((ground_turth, names[v[0][0]], most_c, min_c)) 
with open('High_conf_correct_No_payload.csv','w', newline='') as output:
    csv_out=csv.writer(output)
    csv_out.writerow(['Ground_Truth','Confidence_Classifier_Label', 'Confidence'])
    for row in High_conf_correct:
        csv_out.writerow(row)
with open('One_confidence_No_payload.csv','w', newline='') as output:
    csv_out=csv.writer(output)
    csv_out.writerow(['Ground_Truth','Correct_Classifier_Label', 'Confidence', 'Most_Confident_Label', 'Confidence'])
    for row in One_confidence:
        csv_out.writerow(row)
with open('Zero_confidence_No_payload.csv','w', newline='') as output:
    csv_out=csv.writer(output)
    csv_out.writerow(['Ground_Truth','Confidence_Classifier_Label', 'Confidence'])
    for row in Zero_confidence:
        csv_out.writerow(row)
with open('Majority_correct_No_payload.csv','w', newline='') as output:
    csv_out=csv.writer(output)
    csv_out.writerow(['Ground_Truth','Majority_Classifier_Label', 'Confidence', 'Other_label', 'Confidence'])
    for row in Major_correct:
        csv_out.writerow(row)
with open('Other_majority_No_payload.csv','w', newline='') as output:
    csv_out=csv.writer(output)
    csv_out.writerow(['Ground_Truth','Correct_Classifier_Label', 'Confidence', 'Majority_Classifier_Label', 'Confidence'])
    for row in Other:
        csv_out.writerow(row)
with open('None_majority_No_payload.csv','w', newline='') as output:
    csv_out=csv.writer(output)
    csv_out.writerow(['Ground_Truth','Majority_Classifier_Label', 'Confidence'])
    for row in No_major:
        csv_out.writerow(row)
with open('All_right_No_payload.csv','w', newline='') as output:
    csv_out=csv.writer(output)
    csv_out.writerow(['Ground_Truth', 'Highest_confidence', 'Lowest_confidence'])
    for row in Same_correct:
        csv_out.writerow(row)
with open('All_wrong_No_payload.csv','w', newline='') as output:
    csv_out=csv.writer(output)
    csv_out.writerow(['Ground_Truth', 'Classifiers_label', 'Highest_confidence', 'Lowest_confidence'])
    for row in Same_incorrect:
        csv_out.writerow(row)