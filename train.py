def Cross_validation(data, targets, clf_cv, model_name): #### Performs cross-validation on model
    
    kf = KFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
    scores=[]
    data_train_list = []
    targets_train_list = []
    data_test_list = []
    targets_test_list = []
    iteration = 0
    print("Performing cross-validation for {}...".format(model_name))
    
    # Split data into training and test sets
    for train_index, test_index in kf.split(data):
        iteration += 1
        print("Iteration ", iteration)
        data_train_cv, targets_train_cv = data.iloc[train_index], targets.iloc[train_index]
        data_test_cv, targets_test_cv = data.iloc[test_index], targets.iloc[test_index]
        data_train_list.append(data_train_cv) # appending training data for each iteration
        data_test_list.append(data_test_cv) # appending test data for each iteration
        targets_train_list.append(targets_train_cv) # appending training targets for each iteration
        targets_test_list.append(targets_test_cv) # appending test targets for each iteration
        
        clf_cv.fit(data_train_cv, targets_train_cv) # Fitting model
        score = clf_cv.score(data_test_cv, targets_test_cv) # Calculating accuracy
        scores.append(score) # appending cross-validation accuracy for each iteration
    
    print("List of cross-validation accuracies for {}: ".format(model_name), scores)
    mean_accuracy = np.mean(scores)
    print("Mean cross-validation accuracy for {}: ".format(model_name), mean_accuracy)
    print("Best cross-validation accuracy for {}: ".format(model_name), max(scores))
    
    max_acc_index = scores.index(max(scores)) # best cross-validation accuracy
    max_acc_data_train = data_train_list[max_acc_index] # training data corresponding to best cross-validation accuracy
    max_acc_data_test = data_test_list[max_acc_index] # test data corresponding to best cross-validation accuracy
    max_acc_targets_train = targets_train_list[max_acc_index] # training targets corresponding to best cross-validation accuracy
    max_acc_targets_test = targets_test_list[max_acc_index] # test targets corresponding to best cross-validation accuracy
    
    return mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, scores

def visualize_results(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, targets, clf, model_name): #### Creates Confusion matrix for model
    
    clf.fit(max_acc_data_train, max_acc_targets_train) # Fitting model
    targets_pred = clf.predict(max_acc_data_test) # Prediction on test data
    
    conf_mat = confusion_matrix(max_acc_targets_test, targets_pred)
    d={-1:'Lose', 0: 'Draw', 1: 'Win'}
    sentiment_df = targets.drop_duplicates().sort_values()
    
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=sentiment_df.values, yticklabels=sentiment_df.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix (Best Accuracy) - {}".format(model_name))
    plt.show()
    
    return


if __name__ == "__main__":


    # Importing necessary libraries
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import KFold
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Importing the datasets
    data = pd.read_csv('bestfeatures.csv')
    targets = data['target']
    data = data.drop(['target'], axis=1)

    # Creating the TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=1000)

    # Defining the classifiers
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    clf1 = LogisticRegression()
    clf2 = MultinomialNB()
    clf3 = DecisionTreeClassifier()
    clf4 = RandomForestClassifier()
    clf5 = SVC()

    # Defining the list of classifiers and their names
    clf_list = [(clf1, "Logistic Regression"),
                (clf2, "Naive Bayes"),
                (clf3, "Decision Tree"),
                (clf4, "Random Forest"),
                (clf5, "SVM")]

    # Looping through the classifiers and performing cross-validation
    for clf, clf_name in clf_list:
        print("---------------------------------------------------------------")
        mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test,scores = Cross_validation(data, targets, tfidf, clf, clf_name)
        print("Results for {}: ".format(clf_name))
        print("Mean cross-validation accuracy: {:.3f}".format(mean_accuracy))
        print("Best cross-validation accuracy: {:.3f}".format(max(scores)))
        visualize_results(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, tfidf, targets, clf, clf_name)

