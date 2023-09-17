#Implementación de un algoritmo de ML con framework
#José María Ibarra a01706970

#Librerías y métodos
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

#Máquina de soporte vectorial (SVM)

iris = load_iris()
cancer = load_breast_cancer()

#Datos de ejemplo (Iris)
def dataset_(dataset_num):
    """Prepara el dataeset para entrenamiento 

	Args:
		dataset (int) numero correspondiendo a uno de los datasets disponibles: iris (1), cancer (2)

	Returns:
		Dataset preparado
	"""
    datasets_dict = {1:load_iris(), 2:load_breast_cancer()}
    dataset = datasets_dict[dataset_num]
    X = dataset.data
    y = dataset.target

    if dataset_num == 1: #Seleccionar dos clases y dos atributos
        X = X[y != 2, :]  
        y = y[y != 2]
    
    y[y == 0] = -1  # Ajuste de clases (-1, 1)
    
    # Split de datos en test y train
    split_ratio = 0.7
    split_index = int(split_ratio * len(X))

    # Shuffle
    random_indices = np.random.permutation(len(X))
    X_shuffled = X[random_indices]
    y_shuffled = y[random_indices]

    X_train = X_shuffled[:split_index]
    y_train = y_shuffled[:split_index]
    X_test = X_shuffled[split_index:]
    y_test = y_shuffled[split_index:]

    #añadir bias a train
    samples_train = [] 
    for i in range(X_train.shape[0]):
        sample = np.concatenate(([1], X_train[i]))
        samples_train.append(sample)

    X_train = np.array(samples_train)

    #añadir bias a test
    samples_test = []
    for i in range(X_test.shape[0]):
        sample = np.concatenate(([1], X_test[i]))
        samples_test.append(sample)
    X_test = np.array(samples_test)

    return X_train, y_train, X_test, y_test

def run(data):
    """Genera una corrida de entrenamiento y evaluación de modelo con el dataset indicado

	Args:
		data (int) numero correspondiendo a uno de los datasets disponibles: iris (1), cancer (2)

	Yields:
		Evaluación del modelo
	"""

    #pipeline de modelo
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(penalty='l2' #regularización L2, 
                  ,loss='hinge', dual=True,
                  C = 0.001, fit_intercept=True,
                  random_state=10, max_iter=2000)
    )

    #limpiar dataset
    X_train, y_train, X_test, y_test = dataset_(data)

    #fit (entrenamiento) del modelo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #cross validation
    k=10 #folds
    cross_val = cross_val_score(model, X_train, y_train, cv=k) 
    print(f'{k}-fold cross-validation score: {cross_val}')
    print(f'Mean cv score: {cross_val.mean()}')

    #evaluación del modelo
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()

#corridas
run(1)
run(2)

'''
def plot_learning_curve(data):
    X_train, y_train, _, _ = dataset_(data)
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(penalty='l2',
                  loss='hinge',
                  dual=True,
                  C=0.001,
                  fit_intercept=True,
                  random_state=10,
                  max_iter=2000)
    )

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=10, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker="o")
    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )
    plt.plot(train_sizes, test_mean, label="Validation Score", color="green", marker="s")
    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve (Dataset {data})")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Call this function after your run(1) and run(2) calls to plot learning curves
plot_learning_curve(1)
plot_learning_curve(2)'''

# Define a list of C values to test
C_values = [0.001, 0.01, 0.1, 1, 10]

# Function to plot learning curves for a given model
def plot_learning_curves(model, X_train, y_train, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=10, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker="o")
    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )
    plt.plot(train_sizes, test_mean, label="Validation Score", color="green", marker="s")
    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# Function to run and evaluate the model with different C values
def run_with_multiple_C(data, C_values):
    X_train, y_train, _, _ = dataset_(data)

    for C in C_values:
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(  # L2 regularization or no regularization
                      loss='hinge',
                      dual=True,
                      C=C,
                      fit_intercept=True,
                      random_state=10,
                      max_iter=2000)
        )
        
        model.fit(X_train, y_train)

        # Plot learning curves with the selected penalty
        plot_learning_curves(model, X_train, y_train, f"Learning Curve (Dataset {data}, C={C}, Penalty=None)")

        print(f"Model evaluation with C={C} and Penalty=None")
        y_pred = model.predict(X_test)
        
        print(f"Classification Report (C={C}, Penalty=None):")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()

# Run the models with different C values and penalties
data = 2  # You can change this to 2 for the Breast Cancer dataset
X_train, y_train, X_test, y_test = dataset_(data)
run_with_multiple_C(data, C_values)  # L2 regularization
#run_with_multiple_C(data, C_values, None)  # No regularization (L1 penalty)