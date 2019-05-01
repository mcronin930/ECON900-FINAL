from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
import pandas as pd
import numpy as np

# #
data = pd.read_csv("training_data_burglary.csv")

Training_Target = np.where((data['day_delta'] <14), 1, 0)
Training_Data = data.iloc[:,np.r_[111:114]].values
# print(Training_Target)
print(Training_Data)



data_training, data_test, target_training, target_test = train_test_split(Training_Data, Training_Target, test_size = 0.2, random_state=1)


random_forest_machine = RandomForestClassifier(n_estimators=50)

random_forest_machine.fit(data_training, target_training)

predictions = random_forest_machine.predict(data_test)

print(accuracy_score(target_test, predictions))

confusion_matrix = pd.DataFrame(
	confusion_matrix(target_test,predictions),
	columns = ['Predict No', 'Predict Yes'],
	index = ['True No', 'True Yes']
)

print(confusion_matrix)
