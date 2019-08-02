# Import pandas for our data frame
import pandas as pd

# Import simplefilter for warning control
from warnings import simplefilter

# Import train test split to split our data
from sklearn.model_selection import train_test_split

# Import four different classifiers
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Import accuracy score to see how accurate our classifiers are
from sklearn.metrics import accuracy_score

# Using simplefilter from warnings for warning control
simplefilter(action='ignore', category=FutureWarning)
print()

# Using a xlsx format file as our training data that only contains integers
# On the other version this is web scripted from ufc.com and sherdog.com
# And it contains categorical values that get converted to numerical values using labelencoder
# But due to debugging issues I ran into, I decided to go this route for now
training = pd.read_excel("ufcfightstats30int.xlsx")

# Head gives us the first five rows from the data frame
training.head()

# UI/UX
print("************WELCOME TO UFC FIGHT PREDICTOR************")
print('-'*100)

print("Program Description: \nThis program uses four different classifiers to predict who was the winner from multiple fights. "
      "\nThe classifiers use a data frame that has all the stat outcomes from specific recent UFC fights. ")

print("\nClassifiers used:"
      "\n- Decision Tree Classifier"
      "\n- Logistic Regression Classifier"
      "\n- KNeighbors Classifier"
      "\n- Multilayer Perceptron Classifier")

print("\nStats used: "
      "\n- Strikes"
      "\n- Significant strike"
      "\n- Takedowns"
      "\n- Knockdowns"
      "\n- Submissions attempts"
      "\n- Guard passes"
      "\n- Reversals"
      "\n- Win methods"
      "\n- Last 3 fight outcomes")

print("\nData frame legend:"
      "\n- Win method: 1 = KO/TKO, 2 = Submissions, 3 = Decision"
      "\n- Winner: 1 = Red fighter, 2 = Blue fighter"
      "\n- Last fight: 1 = Win, 2 = Loss, 3 = Draw")

print('-'*100)

wait = input("\nPress enter to view dataframe example.")

print('-'*100)

# Selecting what our feature columns will be
training_features = ['r_strikes', 'b_strikes', 'r_sigstrikes', 'b_sigstrikes', 'r_takedowns', 'b_takedowns',
                'r_knockdowns', 'b_knockdowns', 'r_subtemps', 'b_subtemps', 'r_pass', 'b_pass', 'r_rev',
                'b_rev', 'win_method', 'r_last1', 'b_last1', 'r_last2', 'b_last2', 'r_last3', "b_last3"]


# Using the location function to get particular labels from the index and setting it as X
X = training.loc[:, training_features]

# Printing out our data frame
# On the other version this is displayed in a graph but due to resolution issues I ran into, this version prints it on the console
print(X)

# We set what we want to predict, which is the winner
y = training.winner

# Defining our testing split and testing size, testing size is set to 50%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)


# Call all four classifiers and print out their accuracy score
print('-'*100)

wait = input("\nPress enter to view Decision Tree Classifier accuracy results.")

print()
print('-'*100)
# We call our first classifier, the decision tree classifier from sklearn and display it's accuracy percentage
DecisionTree_Classifier = tree.DecisionTreeClassifier()
DecisionTree_Classifier.fit(X_train, y_train)
DecisiionTree_Prediction = DecisionTree_Classifier.predict(X_test)
print(accuracy_score(y_test, DecisiionTree_Prediction))
print()
print('-'*100)

wait = input("\nPress enter to view Logistic Regression Classifier accuracy results.")

print('-'*100)
# We call our second classifier, the logistic regression classifier from sklearn and display it's accuracy percentage
LogisticRegression_Classifier = LogisticRegression()
LogisticRegression_Classifier.fit(X_train, y_train)
LogisticRegression_Prediction = LogisticRegression_Classifier.predict(X_test)
print(accuracy_score(y_test, LogisticRegression_Prediction))
print()
print('-'*100)

wait = input("\nPress enter to view KNeighbors Classifier accuracy results.")

print('-'*100)
# We call our third classifier, the KNeighbors classifier from sklearn and display it's accuracy percentage
KNeighbors_Classifier = KNeighborsClassifier()
KNeighbors_Classifier.fit(X_train, y_train)
KNeighbors_Prediction = KNeighbors_Classifier.predict(X_test)
print(accuracy_score(y_test, KNeighbors_Prediction))
print()
print('-'*100)

wait = input("\nPress enter to view Multilayer Perceptron Classifier accuracy results.")

print('-'*100)

# We then call our fourth and final classifier, the Multilayer Perceptron classifier from
MLP_CLassifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
MLP_CLassifier.fit(X_train, y_train)
MLP_Prediction = MLP_CLassifier.predict(X_test)
print(accuracy_score(y_test, MLP_Prediction))
print()
print('-'*100)