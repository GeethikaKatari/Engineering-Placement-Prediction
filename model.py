import os
import matplotlib.pyplot as plt
import pickle

#reading the file
data=pd.read_csv('C:\\Users\\Prudhvi\\OneDrive\\Desktop\\NLP\\placement_prediction.csv')
data

#droping the serial no and salary col
data=data.drop('StudentID',axis=1)
data.head()

# for i in range(0,215):
#     if(data['gender'][i]=='M'):
#         data['gender'][i]=1
#     else:
#         data['gender'][i]=0


#catgorising col for further labelling
data["CGPA"]=data["CGPA"].astype('category')
data["Internships"]=data["Internships"].astype('category')
data["Projects"]=data["Projects"].astype('category')
data["Workshops/Certifications"]=data["Workshops/Certifications"].astype('category')
data["AptitudeTestScore"]=data["AptitudeTestScore"].astype('category')
data["SoftSkillsRating"]=data["SoftSkillsRating"].astype('category')
data["ExtracurricularActivities"]=data["ExtracurricularActivities"].astype('category')
data["PlacementTraining"]=data["PlacementTraining"].astype('category')
data["SSC_Marks"]=data["SSC_Marks"].astype('category')
data["HSC_Marks"]=data["HSC_Marks"].astype('category')
data.dtypes

#labelling the col
data["CGPA"]=data["CGPA"].cat.codes
data["Internships"]=data["Internships"].cat.codes
data["Projects"]=data["Projects"].cat.codes
data["Workshops/Certifications"]=data["Workshops/Certifications"].cat.codes
data["AptitudeTestScore"]=data["AptitudeTestScore"].cat.codes
data["SoftSkillsRating"]=data["SoftSkillsRating"].cat.codes
data["ExtracurricularActivities"]=data["ExtracurricularActivities"].cat.codes
data["PlacementTraining"]=data["PlacementTraining"].cat.codes
data["SSC_Marks"]=data["SSC_Marks"].cat.codes
data["HSC_Marks"]=data["HSC_Marks"].cat.codes
data

#selecting the features and labels
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values
Y

#dividing the data into train and split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X_train
data.head()

#creating a classifier using sklearn
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train,Y_train)
#printing the acc
clf.score(X_test,Y_test)


#predicting for random value
clf.predict([[7.5,1,1,1,65,4.4,0,0,51,79]])


#creating a Y_pred for test data
Y_pred=clf.predict(X_test)
Y_pred

#model generation
pickle.dump(clf, open('placement_prediction.pkl','wb'))
#evalution of the classifier
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
accuracy_score(Y_test, Y_pred)