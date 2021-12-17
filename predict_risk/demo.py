from .forms import Predict_Form
from accounts.models import UserProfileInfo
from predict_risk import add_info
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib.auth.decorators import login_required,permission_required
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('D:/Projects/Heart-disease-prediction-master/predict_risk/classified_dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x[:, 11:13])
x[:, 11:13] = imputer.transform(x[:, 11:13])


def LogisticRegression(Newdataset):
    from sklearn.linear_model import LogisticRegression
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    ynew = classifier.predict(Newdataset)
    return ynew[0]


def KNN(Newdataset):
    from sklearn import neighbors
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    ynew = clf.predict(Newdataset)
    return ynew[0]


def SupportVectorClassifier(Newdataset):
    from sklearn.svm import SVC
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    sc = SVC(kernel='rbf')
    sc_classifier = sc.fit(x_train, y_train)  # model building
    ynew = sc_classifier.predict(Newdataset)
    return ynew[0]


def DecisionTree(Newdataset):
    from sklearn.tree import DecisionTreeClassifier
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    dtc_clf = DecisionTreeClassifier()
    dtc_clf.fit(x_train, y_train)
    ynew = dtc_clf.predict(Newdataset)
    return ynew[0]


def RandomForest(Newdataset):
    from sklearn.ensemble import RandomForestClassifier
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    rmf = RandomForestClassifier(max_depth=3, random_state=0)
    rf_classi = rmf.fit(x_train, y_train)
    ynew = rf_classi.predict(Newdataset)
    return ynew[0]


def NaiveBayes(Newdataset):
    from sklearn.naive_bayes import GaussianNB
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    ynew = classifier.predict(Newdataset)
    return ynew[0]


@login_required(login_url='/')
def PredictRisk(request,pk):
    predicted = False
    predictions={}
    #if request.session.has_key('user_id'):
    u_id = request.session['user_id']

    if request.method == 'POST':
        form = Predict_Form(data=request.POST)
        profile = get_object_or_404(UserProfileInfo, pk=pk)

        if form.is_valid():
            features = [[ form.cleaned_data['age'], form.cleaned_data['sex'], form.cleaned_data['cp'], form.cleaned_data['resting_bp'], form.cleaned_data['serum_cholesterol'],
            form.cleaned_data['fasting_blood_sugar'], form.cleaned_data['resting_ecg'], form.cleaned_data['max_heart_rate'], form.cleaned_data['exercise_induced_angina'],
            form.cleaned_data['st_depression'], form.cleaned_data['st_slope'], form.cleaned_data['number_of_vessels'], form.cleaned_data['thallium_scan_results']]]
            print(features)
            Newdataset=features

            predictions = {}
            predictions['LogisticRegression'] = LogisticRegression(Newdataset)
            predictions['KNN'] = KNN(Newdataset)
            predictions['SupportVectorClassifier'] = SupportVectorClassifier(Newdataset)
            predictions['NaiveBayes'] = NaiveBayes(Newdataset)
            predictions['DecisionTree'] = DecisionTree(Newdataset)
            predictions['RandomForest'] = RandomForest(Newdataset)

            print(predictions)

            from collections import Counter

            l=list(map(int,predictions.values()))
            l=dict(Counter(l))
            print(l)
            if len(set(l.values()))==1:
                result='equal'
                num=-1
            else:
                for w in sorted(l, key=l.get, reverse=True):
                    result=str(w)
                    num=w
                    break
            print(predictions.values())
            predicted = True
            print(result)

            pred = form.save(commit=False)
            pred.profile = profile
            pred.num=num
            pred.save()

            add_data=features
            add_data[0].append(num)
            print(add_data[0])
            add_info.insert_into_dataset(add_data[0])


            return render(request, 'predict.html',
                          {'form': form, 'predicted': predicted, 'user_id': u_id, 'result':result, 'predictions': predictions})

    else:
        form = Predict_Form()

        return render(request, 'predict.html',
                      {'form': form, 'predicted': predicted, 'user_id': u_id, 'predictions': predictions})
