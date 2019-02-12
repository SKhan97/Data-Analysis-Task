import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.patches as mpatches


#Importing the attributes csv file and organising data (simplify headings, drop unnecessary columns and change to numerical values)
df = pd.read_csv("attributes_report.csv")
df = df.rename(index=str, columns={'company_type':'Type', 'subscribed_after_free_trial' : 'Subscribed', 'company':'Company'})
subscription_mapping = {False:0, True: 1}
company_mapping = {}

for x in df['Type']:
    if x in company_mapping:
        pass
    else:
        company_mapping[x] = len(company_mapping)

df = df.applymap(lambda s: subscription_mapping.get(s) if s in subscription_mapping else s)
df = df.applymap(lambda s: company_mapping.get(s) if s in company_mapping else s)


#Combining this dataframe with dataframe for number of logins per day for the week-long subscription
df_login = pd.read_csv("7_features_data.csv")
df = pd.merge(df_login, df, left_on=('Company'), right_on=('Company'))
df.drop('Company', inplace=True, axis=1)


#Creating correlation matrix and plotting a diagonal representation of the correlations
corr_matrix = df.corr()
sns.set(style="white")

mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, l=40, s = 90 ,as_cmap=True)

sns_plot = sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-0.1, vmax=.6, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
f.savefig("Correlation Matrix.png", bbox_inches = "tight")

#Plotting a bar graph to compare average number of logins of subscribed users to users who didn't subcribe
dataframe_dict = dict(tuple(df.groupby('Subscribed')))
df_unsubscribed = dataframe_dict[0]
df_subscribed = dataframe_dict[1]

login_unsubscribed = df_unsubscribed.mean(axis=0).values[:7]
login_subscribed = df_subscribed.mean(axis=0).values[:7]
login_days = np.arange(1,8)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim([0,1.5])
width=0.25
ax.bar(login_days+width, login_subscribed, width=width)
ax.bar(login_days, login_unsubscribed, width=width, color='orange')
plt.xlabel('Day')
plt.ylabel('Average number of logins')
orange_bar = mpatches.Patch(color='orange', label='Not Subscribed')
blue_bar = mpatches.Patch(color='blue', label='Subscribed')
plt.legend(handles=[orange_bar, blue_bar])
plt.show()
fig.savefig('logins_per_day.png', bbox_inches = 'tight')


#Splitting features and output into arrays whose elements are floats. Scaling of features for better results
features = ['Type', ' Day 1', ' Day 2', ' Day 3', ' Day 4', ' Day 5', ' Day 6', ' Day 7']
X = df[features].values.astype(float)
y = df['Subscribed'].values.astype(float)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

#Training the model and checking accuracy both by fraction of correct predictions and also cross validation
from sklearn.metrics import confusion_matrix

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0) #gini criterion since it is computationally easier
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
correct_predictions = cm[0][0] +cm[1][1]
false_predictions = cm[1][0] +cm[0][1]
accuracy = correct_predictions/(correct_predictions + false_predictions)


fig_cm = plt.figure()
df_cm = pd.DataFrame(cm, index=['Subscribed - Data', 'Unscubscribed'], columns=['Subscribed - Prediction', 'Unsubscribed'])
sns.set(font_scale=1)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})
plt.show()
fig_cm.savefig("Confusion_Matrix.png", bbox_inches = 'tight')

print(cm, "\n This has given us a total of %i correct prediction and %i false positives/negatives. This indicates a total score of %0.2f" % (correct_predictions, false_predictions,accuracy))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)

print("5 Fold Cross validation shows the model predicted with an accuracy of %0.2f (+/- %0.2f)" % (accuracies.mean(), accuracies.std() * 2))

"""
Visualising the Random Forest Classifier by writing to dot and converting to png. Need Graphviz and have to change 
path variable to location of Graphviz in order to use dot command
"""

from sklearn.tree import export_graphviz
import os
estimator = classifier.estimators_[5]

export_graphviz(estimator, out_file='tree.dot',
                feature_names = features,
                class_names = 'Subscribed',
                rounded = True, proportion = False,
                precision = 2, filled = True)

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.system('dot -Tpng tree.dot -o tree.png')
