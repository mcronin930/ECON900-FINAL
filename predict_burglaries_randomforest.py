from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np

n = 90
#
data = pd.read_csv("training_data_burglary.csv")
Training_Target = np.where((data['day_delta'] < 14), 1, 0)
print(Training_Target)
Training_Data = data.iloc[:,np.r_[32:115]]
print(Training_Data)

data_training, data_test, target_training, target_test = train_test_split(Training_Data, Training_Target, test_size = 0.2, random_state=1)
random_forest_machine = RandomForestClassifier(n_estimators=n)
random_forest_machine.fit(data_training, target_training)
predictions = random_forest_machine.predict(data_test)
print(accuracy_score(target_test, predictions))

cm = confusion_matrix(target_test,predictions)
confusion_matrix = pd.DataFrame(
	cm,
	columns = ['Predict No', 'Predict Yes'],
	index = ['True No', 'True Yes']
)
print(confusion_matrix)
print(dict(zip(Training_Data.columns, random_forest_machine.feature_importances_)))

### Train Model to Select Most Important Features and Refine The Model
# Fine Best Features
select_features = SelectFromModel(random_forest_machine, threshold = 0.1)
select_features.fit(data_training, target_training)
x_refined_train = select_features.transform(data_training)
x_refined_test = select_features.transform(data_test)

# Print Top Features
important_features = select_features.get_support()
feature_name = Training_Data.columns[important_features]
print(feature_name.value_counts())

#Train and Test New Model
refined_forest_machine = RandomForestClassifier(n_estimators=n)
refined_forest_machine.fit(x_refined_train, target_training)
refined_predictions = refined_forest_machine.predict(x_refined_test)
print(accuracy_score(target_test, refined_predictions))

# for some reason i could not run this part of the code until i imported the confusion matrix library again
from sklearn.metrics import confusion_matrix
cmr = confusion_matrix(target_test,refined_predictions)
confusion_matrix = pd.DataFrame(
	cmr,
	columns = ['Predict No', 'Predict Yes'],
	index = ['True No', 'True Yes'])

print(confusion_matrix)
