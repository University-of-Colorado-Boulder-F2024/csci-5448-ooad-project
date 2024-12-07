

---

# **OOAD Comparison of TensorFlow and DL4J**

## **Project Overview**
This repository contains the code, analysis, and performance benchmarks for comparing the object-oriented design (OOD) implementations in **TensorFlow** (Python) and **DL4J** (Java). The project focuses on core object-oriented principles such as:
- **Encapsulation**
- **Polymorphism**
- **Inheritance**

It evaluates their impact on key performance metrics:
- **Memory usage**
- **Training time**
- **Prediction time**

Through hands-on examples and benchmarking, this project highlights the trade-offs between **flexibility** (TensorFlow) and **structure** (DL4J) in object-oriented programming for machine learning frameworks.

---

## **Folder Structure**

```plaintext
project-root/
├── src/
│   ├── main/
│   │   ├── ooad/
│   │   │   ├── dl4j/
│   │   │   │   ├── RandomForestModel.java          # Demonstrates encapsulation in DL4J
│   │   │   │   ├── DenseLayerModel.java            # Demonstrates polymorphism in DL4J
│   │   │   │   ├── RandomForestInheritanceModel.java # Demonstrates inheritance in DL4J
│   │   │   │   ├── Model.java                      # Interface for polymorphism in DL4J
│   │   │   │   ├── DL4JBenchmark.java              # DL4J performance benchmarking
├── target/
│   │   ├── test-classes/
│   │   │   ├── DL4JPerformanceBenchmarkTest.java   # Unit tests for DL4J code
├── pom.xml
├── tensorflow/
│   │   ├── Encapsulation.py        # Demonstrates encapsulation in TensorFlow
│   │   ├── Inheritance.py          # Demonstrates inheritance in TensorFlow
│   │   ├── Polymorphism.py         # Demonstrates polymorphism in TensorFlow
│   │   ├── tensorflow_performance_analysis.py # TensorFlow performance benchmarking
│   │   ├── test_performance_analysis.py  # Unit tests for Python code



```

---

## **Prerequisites**

### **For TensorFlow (Python)**
- **Python Version:** Python 3.8 or later  
- **Required Libraries:**
  - TensorFlow Decision Forests
  - Pandas
  - Matplotlib
  - Psutil

**Install the dependencies:**
```bash
pip install tensorflow_decision_forests pandas matplotlib psutil
```

---

### **For DL4J (Java)**
- **Java Version:** JDK 11 or later  
- **Build Tool:** Maven or Gradle  
- **Dependency Management:**

Add the following DL4J dependency to your `pom.xml` (Maven) or `build.gradle` (Gradle):

**Maven:**
 ```xml
  <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-core</artifactId>
      <version>1.0.0-M2</version>
  </dependency>
  <dependency>
      <groupId>org.jfree</groupId>
      <artifactId>jfreechart</artifactId>
      <version>1.5.3</version>
  </dependency>
  ```

**Gradle:**
```gradle
implementation 'org.deeplearning4j:deeplearning4j-core:1.0.0-M2'
```

---

## **How to Run**

### **TensorFlow Scripts**
1. **Navigate to the TensorFlow directory:**
   ```bash
   cd src/tensorflow
   ```
2. **Run the desired script:**
   - **Encapsulation:**
     ```bash
     python Encapsulation.py
     ```
   - **Polymorphism:**
     ```bash
     python Polymorphism.py
     ```
   - **Inheritance:**
     ```bash
     python Inheritance.py
     ```
   - **Performance Analysis:**
     ```bash
     python tensorflow_performance_analysis.py
     ```
     This script outputs:
     - **Performance metrics:** Training time, memory usage, and prediction time.
     - **Graph:** `tensorflow_performance_analysis.png`.

---

### **DL4J Scripts**
1. **Compile the Java files:**
   ```bash
   javac -cp deeplearning4j-core.jar;. ooad/dl4j/*.java
   ```
2. **Run the desired file:**
   - **Encapsulation:** Run `RandomForestModel.java`.
     ```bash
     java -cp deeplearning4j-core.jar;. ooad.dl4j.RandomForestModel
     ```
   - **Polymorphism:** Run `DenseLayerModel.java`.
     ```bash
     java -cp deeplearning4j-core.jar;. ooad.dl4j.DenseLayerModel
     ```
   - **Inheritance:** Run `RandomForestInheritanceModel.java`.
     ```bash
     java -cp deeplearning4j-core.jar;. ooad.dl4j.RandomForestInheritanceModel
     ```
   - **Performance Benchmarking:** Run `DL4JBenchmark.java`.

**Outputs:**
- **Performance metrics:** Training time, memory usage, and prediction time.
- **Graph:** `DL4J_performance_analysis.png`.

---

## **Example Outputs**

### **TensorFlow Outputs**

#### **Encapsulation.py**
```plaintext
PS C:\Users\softy\Downloads\csci-5448-compilecrew-ooad-project-main> python Encapsulation.py
Encapsulation Example:
Model trained successfully.
Predictions:
[[0]
 [1]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [0]
 [1]
 ...
]
```

#### **Polymorphism.py**
```plaintext
PS C:\Users\softy\Downloads\csci-5448-compilecrew-ooad-project-main> python Polymorphism.py
Polymorphism Example:
Model A Output:
[[0.551245  0.742198 ...]]
Model B Output:
[[0.712198  0.482137 ...]]
```

#### **Inheritance.py**
```plaintext
PS C:\Users\softy\Downloads\csci-5448-compilecrew-ooad-project-main> python Inheritance.py
Inheritance Example:
tf.Tensor(
[[0.451785  0.242157 ...]])
```

---

### **DL4J Outputs**

#### **RandomForestModel.java**
```plaintext
PS C:\Users\softy\Downloads\csci-5448-compilecrew-ooad-project-main> java -cp deeplearning4j-core.jar;. ooad.dl4j.RandomForestModel

Building Random Forest Model with 10 trees and max depth 5
Training the Random Forest Model...
Training complete.
Making predictions...
Predictions:
[[0.453, 0.547],
 [0.329, 0.671],
 [0.789, 0.211],
 ...
 [0.234, 0.766]]
```

#### **DenseLayerModel.java**
```plaintext
PS C:\Users\softy\Downloads\csci-5448-compilecrew-ooad-project-main> java -cp deeplearning4j-core.jar;. ooad.dl4j.DenseLayerModel

Initializing Dense Layer Model with input size 10 and output size 2
Predicting with Dense Layer Model...
Predictions:
[[0.712, 0.288],
 [0.849, 0.151],
 [0.501, 0.499],
 ...
 [0.612, 0.388]]
```

#### **RandomForestInheritanceModel.java**
```plaintext
PS C:\Users\softy\Downloads\csci-5448-compilecrew-ooad-project-main> java -cp deeplearning4j-core.jar;. ooad.dl4j.RandomForestInheritanceModel

Building Random Forest Inheritance Model...
Training the model...
Epoch 1 complete.
Epoch 2 complete.
Epoch 3 complete.
Epoch 4 complete.
Epoch 5 complete.
Training complete.
```
## Running Unit Tests

### Python Unit Tests:
1. Ensure you are in the TensorFlow directory:
   ```bash
   cd tensorflow
   ```
2. Run the unit tests:
   ```bash
   python -m unittest test_performance_analysis.py
   ```
3. Example Output:
   ```plaintext
   .....
   ----------------------------------------------------------------------
   Ran 5 tests in 0.452s

   OK
   ```

### Java Unit Tests:
1. Run the unit tests using Maven:
   ```bash
   mvn test
   ```
2. Example Output:
   ```plaintext
   [INFO] Scanning for projects...
   -------------------------------------------------------
    T E S T S
   -------------------------------------------------------
   Running ooad.dl4j.DL4JPerformanceBenchmarkTest
   Tests run: 4, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.832 s - in ooad.dl4j.DL4JPerformanceBenchmarkTest

   Results:

   Tests run: 4, Failures: 0, Errors: 0, Skipped: 0

   [INFO] BUILD SUCCESS
   ```

---







---

## **Results and Analysis**

### **Memory Usage**
- **TensorFlow:** Consumes more memory due to Python's dynamic memory allocation.
- **DL4J:** More memory-efficient, benefiting from Java's strict memory management.

### **Training Time**
- **TensorFlow:** Faster for smaller datasets, making it ideal for prototyping.
- **DL4J:** Scales better for larger datasets and complex models.

### **Prediction Time**
- **DL4J:** Demonstrates faster and more consistent predictions due to its compiled nature.
- **TensorFlow:** Slightly slower predictions due to runtime overhead.

### **Graphical Comparison**
- Performance graphs:
  - **TensorFlow:** `Tensorflow_performance_analysis.png`
  - **DL4J:** `DL4J_performance_analysis.png`

---

## **Contributors**
- **Anchal Basia** (anchal.basia@colorado.edu)  
- **Dharini Baskaran** (dharini.baskaran@colorado.edu)

---

## **Acknowledgments**
This project was developed as part of the **Object-Oriented Analysis and Design (OOAD)** course at the **University of Colorado, Boulder**. Special thanks to the faculty for their guidance and support.




