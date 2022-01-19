# Import modules
import codecademylib3_seaborn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Import breast cancer data
from sklearn.datasets import load_breast_cancer

# Load dataset into breast_cancer_data variable
breast_cancer_data = load_breast_cancer()

# Examine the dataset
#print(breast_cancer_data.data)

# Print out the first datapoints in the dataset
print(breast_cancer_data.data[0])

# Examine the feature_names datapoints from the dataset
print(breast_cancer_data.feature_names)

# Examine the target datapoints from the dataset
print(breast_cancer_data.target)

# Examine the target_names datapoints from the dataset
print(breast_cancer_data.target_names)

# Split data
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

# Output the length of training_data
print(len(training_data))

# Output the length of training_labels
print(len(training_labels))

# Create an empty list
accuracies = []

# Create a loop for k which increases from 1 to 100
for k in range(1, 101):
  # Create KNeighborsClassifier object
  classifier = KNeighborsClassifier(n_neighbors = k)
  # Train classifier
  classifier.fit(training_data, training_labels)
  # Calculate the score on classifier
  accuracies.append(classifier.score(validation_data, validation_labels))

# Create list of numbers between 1 and 100
k_list = range(1, 101)

# Create a line plot
plt.plot(k_list, accuracies)

# Set x-axis label
plt.xlabel("K")

# Set y-axis label
plt.ylabel("Validation Accuracy")

# Set plot title
plt.title("Breast Cancer Classifier Accuracy")

# Show plot
plt.show()