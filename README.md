# Enhancing Protein Structure Analysis

## Introduction

The Residue Interaction Network Generator (RING) and Ribbon Diagrams have revolutionized the study of protein structures, enabling researchers to delve into the intricate details of residue interactions.
The aim of this powerfull tool in the field of structural biology is to provide a simple and intuitive way to visualize the interactions between residues in a protein structure.
In our case we are going to develop a software that will be able to predict the interactions between residues in a protein structure, given a dataset of protein structures and its corresponding interactions.
In our code, we have prioritized flexibility, allowing us to experiment with different configurations and observe the prediction results in the output. This adaptability empowers us to fine-tune our analysis and explore various scenarios effortlessly.

## Example of usage

In the README of your GitHub repository, you can include the following explanation to guide users on how to use the `main.py` script with various parameters:

## Running the Script

**Create conda enviroment**:

[Install conda enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) if you don't have it.

```bash
conda create --name your_new_env_name --file requirements.txt
conda activate your_new_env_name
```

**Usage of the script:**

The `main.py` script allows you to train and evaluate different models with various normalization techniques and data options. To run the script, you can use the following command-line arguments:

```bash
python3 main.py -m [model] -n [normalization] -d [data_option]
```

**Parameters:**

- `[model]`: Choose one of the available models: "model_1", "model_2", or "model_3". Each model represents a specific machine learning model or architecture.
- `[normalization]`: Choose one of the available normalization techniques: "MinMaxScaler" or "StandardScaler". This parameter allows you to preprocess the data before training the models.
- `[data_option]`: Choose one of the available data options: "eliminate_unclassified" or "unclassified". This option determines how the unclassified data is handled during training and evaluation.

**Example:**

```bash
python3 main.py -m model_1 -n MinMaxScaler -d eliminate_unclassified
python3 main.py -m model_1 -n MinMaxScaler -d unclassified
# ... (similar commands for other models and combinations)
```

Feel free to experiment with different combinations to find the best configuration for your specific use case.

By following these instructions, users can easily utilize the `main.py` script and explore various models, normalization techniques, and data options to identify the optimal configuration for their machine learning tasks.

**REMEMBER**: you have to unpack the data with this command:

```bash
cd data
cat data_part_a* > data_all.tar.gz
tar -xvzf data_all.tar.gz
unzip feature_ring.zip
```

## Dataset

The first look of the project were given to datasets provided to understand the problem and the data. The datasets provide different type of geometrical information about the protein structures. We are going to summarize, the data are devided in two parts the source amino acid and the target amino acid. The source amino acid is the amino acid that is going to interact with the target amino acid. The data for the source and target aminoacids are provided in a csv file with the following columns:

- **pdb_id**: The id of the protein structure.
- **ch**: The chain of the protein structure.
- **resi**: The residue number of the amino acid.
- **resn**: The name of the amino acid.
- **ss8**: The secondary structure of the amino acid, calculated with DSSP.
- **rsa**: The relative solvent accessibility of the amino acid, calculated with DSSP.
- **phi**: The phi angle of the amino acid, ramachandran plot used to determine the angles.
- **psi**: The psi angle of the amino acid, ramachandran plot used to determine the angles.
- **up**: Half sphere exposure of the amino acid.
- **down**: Half sphere exposure of the amino acid.
- **ss3**: secondary structure
- **a1 to a5**: Atchley scale of the amino acid.

## Adding New data

To enhance the accuracy of our predictions regarding interactions between amino acids, we recognized that the existing dataset was insufficient. As a result, we made the decision to augment the dataset with additional data. Prior to this decision, we conducted preliminary experiments using simple prediction models. These models and their outcomes will be presented in detail later in the report, providing a clearer understanding of our rationale.

The first step involved incorporating a new set of data obtained from the DSSP module. To achieve this, we made modifications to the code provided by our professor, specifically in the calcfeatures.py file. During this process, we also identified and rectified errors present in the code, which were caused by the use of deprecated libraries. By addressing these issues, we ensured the proper functionality of the code and obtained the necessary data from the DSSP module.

These additions to the dataset, obtained through modifications and improvements to the code, significantly bolstered the quality and comprehensiveness of the data used for predicting amino acid interactions. The subsequent sections of the paper will delve into the details of the modifications made to the code and the specific improvements achieved:

```python
   if file_extension == ".pdb":
        structure = PDBParser(QUIET=True).get_structure(pdb_id, 
        args.pdb_file)
    elif file_extension == ".cif":
        structure = MMCIFParser(QUIET=True).get_structure(pdb_id, 
        args.pdb_file)
    else:
        print("Error: Unsupported file extension.")
```

By implementing these changes, our code is now capable of processing and extracting relevant data from both CIF and PDB files. This flexibility allows us to incorporate a broader range of structural data sources, enhancing the scope and versatility of our analysis.

Overall, this modification provides us with greater flexibility in data acquisition and enables us to explore a wider range of structural information, contributing to a more comprehensive analysis of amino acid interactions.

We encountered challenges with the deprecated version of the DSSP module, which caused conflicts with our Python version. To address this issue, we took a two-step approach.

First, we used an external software tool to generate the DSSP file separately. This allowed us to obtain the required structural information for further analysis. In the second step, we carefully examined the code and identified the specific problem causing the conflict. By updating the module and replacing the deprecated dssp function with mkdssp, we resolved the compatibility issues.

The following code snippet demonstrates the implementation of this update:

```python
   dssp = {}
    try:
        dssp = dict(DSSP(structure[0], args.pdb_file, dssp="mkdssp"))
    except Exception:
        logging.warning("{} DSSP error".format(pdb_id))
``` 

By resolving the compatibility issues, we were able to obtain the required structural information from the DSSP module. This enabled us to incorporate a broader range of structural data sources, enhancing the scope and versatility of our analysis.

To address the absence of PDB files associated with the feature_ring dataset provided, we developed a Python script named **collect_data.py**. This script efficiently downloads all the necessary PDB files related to the dataset. The process is straightforward: the script takes the filename as input from the feature_ring folder and proceeds to download the corresponding PDB files.

Furthermore, after the successful download of the PDB files, the script automatically executes the calcfeatures.py script for each downloaded PDB file. This allows us to calculate the new features specific to each PDB file by making appropriate calls from the terminal.

By utilizing the **collect_data.py** script, we ensure the availability of all the required PDB files for the feature_ring dataset. This enables us to generate accurate and up-to-date features for further analysis.

### New Features

The new features added to the dataset consist of 8 features for the source amino acid and 8 features for the target amino acid. These features represent hydrogen bond interactions between the amino acids and are obtained from the official DSSP website.

Example of the new features **"N-H-->O"**:

The hydrogen bond interactions are denoted by the notation "N-H-->O", where "N-H" represents the hydrogen donor (N-H group of the source amino acid), and "O" represents the hydrogen acceptor (C=O group of the target amino acid). The numbers associated with each hydrogen bond indicate the distance between the two amino acids involved in the interaction. For example, "-3, -1.4" signifies that if the source residue is residue "i", then the N-H group of "i" forms a hydrogen bond with the C=O group of "i-3" (three residues before "i") with an electrostatic hydrogen bond energy of "-1.4 kcal/mol".

By incorporating these new hydrogen bond features into the dataset, we gain valuable information about the interactions between amino acids.
Also we have added all the bonds features that are present in the DSSP module.

[DSSP official Documentation](https://swift.cmbi.umcn.nl/gv/dssp/)

### Looking at the data

Upon careful examination of the data and file, we have observed instances where certain rows are identical, except for the interactions recorded. This occurs because a single amino acid can interact with multiple other amino acids. As a result, we encounter a challenge of having duplicated data entries within the dataset.

To address this issue, we have implemented a solution involving the creation of a new column called "interaction." This column serves as a list to store the various interactions involving the amino acid.

To achieve this, we utilize a dictionary structure. The key of the dictionary comprises the **pdb_id**, **s_resi**, and **t_resi** values. By using this key, we can identify and select the duplicated rows within the same file of the dataset. Subsequently, we append the corresponding interactions to the "interaction" column.

By employing this approach, we effectively consolidate the duplicated data entries and store the diverse interactions of the amino acid within a single row. This enhances the clarity and efficiency of our dataset, avoiding unnecessary duplication of information.

*Example of duplicated rows*:

| pdb_id | s_resi |...| t_resi |...| Interaction |
|--------|--------|---|--------|---|-------------|
| 1b0y   | 28     |...|76      |...| HBOND       |
| 1b0y   | 28     |...|76      |...| VDW         |

*Solution provided*:

| pdb_id | s_resi |...|t_resi |...|Interaction|
|--------|--------|---|-------|---|------------|
| 1b0y   | 28     |...|76     |...|[HBOND, VDW]|

This will increase the number of possible interactions and will make the dataset more realistic.

## Data preprocessing

In our dataset, it is important to note that not all possible interactions between residues are present. This occurs because some interactions may not be captured in the protein's structure. This poses a challenge as it results in a significant amount of data that cannot be utilized for analysis. 

To address this issue, we have made a deliberate decision. Rather than excluding those rows entirely, we have introduced a new category called **unclassified** to represent these interactions. By including this category, we aim to create a more comprehensive and robust model. We understand that this approach may result in a potential loss of accuracy in the final model. However, after careful consideration, we believe it is the best solution to handle the absence of certain interactions in the dataset.

By adding the **unclassified** category, we acknowledge the presence of unobserved or unclassified interactions, ensuring that they are accounted for in our analysis. This approach allows us to capture a broader range of potential interactions, even though they may not be directly observable in the protein structure.

But also we have added a version of the code in which the unclassified interactions are not taken into account. This version of the code is called **"eliminate_unclassified"**.

## Analysis of the interaction

![](data/images/interaction_simple.png)
**Figure 1**: In this figure, we observe the distribution of interactions in the dataset provided for the project. It is evident that the majority of interactions fall under the categories of **HBOND** and **Unclassified**. Conversely, **SSBOND** and **PICATION** are found to be almost non-existent in the dataset.

![](data/images/interaction_complex.png)
**Figure 2**: In this figure, we observe the distribution of interactions in the newly created dataset. As anticipated, the interactions that are most prevalent remain consistent with our previous findings, being primarily **HBOND** and **Unclassified** interactions. Additionally, we have discovered the emergence of new interaction pairs, which represent instances where certain interactions are highly likely to co-occur. These findings provide valuable insights into the relationship between different interactions within the dataset.

![](data/images/confusion_matrix.png)

**Figure 3**: In this figure, we present the confusion matrix that illustrates the correlations among the data. However, a significant issue becomes apparent: certain interactions are overly represented compared to others. This imbalance in representation poses a considerable challenge and warrants further investigation.

## Normalization of the data

The normalization proposed is done by normalize the data depending on the type of data. The normalization is done by using the MinMaxScaler and StandardScaler from sklearn.

- **MinMaxScaler**: scales the data to a specific range, typically between 0 and 1. It ensures that all features are on the same scale, making them suitable for certain machine learning algorithms that are sensitive to feature magnitudes.
- **StandardScaler**: standardizes the data by removing the mean and scaling each feature to have unit variance. It ensures that all features have a mean of 0 and a standard deviation of 1, making them comparable regardless of their original units or distributions.

Deepley in the normalization: 

1. **Normalization of Angles:**
   The first normalization function focuses on columns representing angles in the dataset. Angles are mathematical values that describe the shape or orientation of certain features.

2. **Normalization of Acetylcholine Features:**
   The second normalization function deals with specific columns related to acetylcholine features. Acetylcholine is a neurotransmitter that plays a role in various biological processes.

3. **Normalization of Relative Solvent Accessibility (RSA):**
   The third normalization function focuses on columns representing the relative solvent accessibility (RSA). RSA refers to how exposed or buried a certain part of a molecule is within its environment.

4. **Normalization of Half-Sphere Coordinates:**
   The fourth normalization function addresses columns containing half-sphere coordinates. These coordinates describe the position or orientation of certain features.

5. **Normalization of Categorical Features:**
   The final normalization function deals with categorical features in the dataset. Categorical features represent different categories or labels, such as types of structures or residues. To transform these categorical labels into numerical representations, a technique called **LabelEncoder** is used. This transformation enables the categorical features to be used effectively in machine learning algorithms and analysis.

6. **Normalization of bond interaction:**
   The fifth normalization function addresses columns containing bond interactions. These interactions describe the interactions between the amino acids.

By applying these normalization techniques, the data is prepared in a standardized and comparable format. This ensures that the various features are on a consistent scale, allowing for accurate analysis, modeling, and interpretation of the data.

## Splitting

At this point, we have decided to utilize two primary libraries for machine learning and deep learning: TensorFlow with the assistance of Keras. These two libraries are widely recognized and extensively used in the field of machine learning and deep learning.

To begin the modeling process, we need to partition our data into training, validation, and testing sets. We have chosen to achieve this using the train_test_split function from the sklearn library, which allows us to divide the data into training and testing subsets. Subsequently, we further split the training data into two subsets: one for training and the other for validation. This partitioning is facilitated by the implementation of the **split.py** script.

## Multiclass Classification Problem

In machine learning and statistical classification, multiclass classification or multinomial classification is the problem of classifying instances into one of three or more classes (classifying instances into one of two classes is called binary classification). Multiclass classification should not be confused with multi-label classification, where multiple labels are to be predicted for each instance.
We take also in account to use this last possibility by codify our interaction as one hot encoding vector for our prediction.

### Models

We used different models to predict our labels, and among them, one stood out as the best performer that is **model 3**. The other models that were utilized but did not provide satisfactory results were used but does not provide good result.

#### Model1

```python
_____________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 dense (Dense)               (None, 128)               5504      
                                                                 
 batch_normalization (BatchN (None, 128)               512       
 ormalization)                                                   
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 batch_normalization_1 (Batc (None, 64)                256       
 hNormalization)                                                 
                                                                 
 dropout_1 (Dropout)         (None, 64)                0         
                                                                 
 dense_2 (Dense)             (None, 32)                2080      
                                                                 
 batch_normalization_2 (Batc (None, 32)                128       
 hNormalization)                                                 
                                                                 
 dropout_2 (Dropout)         (None, 32)                0         
                                                                 
 dense_3 (Dense)             (None, 7)                 231       
                                                                 
=================================================================
Total params: 16,967
Trainable params: 16,519
Non-trainable params: 448
_____________________________________________________________
``` 

#### Model2

```python
_________________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 dense (Dense)               (None, 128)               5504      
                                                                 
 batch_normalization (Batch  (None, 128)               512       
 Normalization)                                                   
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 256)               33024     
                                                                 
 batch_normalization_1(Batc  (None, 256)               1024      
 hNormalization)                                                 
                                                                 
 dropout_1 (Dropout)         (None, 256)               0         
                                                                 
 dense_2 (Dense)             (None, 512)               131584    
                                                                 
 batch_normalization_2(Batc  (None, 512)               2048      
 hNormalization)                                                 
                                                                 
 dropout_2 (Dropout)         (None, 512)               0         
                                                                 
 dense_3 (Dense)             (None, 7)                 3591      
                                                                 
=================================================================
Total params: 177,287
Trainable params: 175,495
Non-trainable params: 1,792
_________________________________________________________________
```

#### Model3

```python
_________________________________________________________________
 Layer (type)                Output Shape              Param   
=================================================================
 dense_4 (Dense)             (None, 896)               38528     
                                                                 
 dense_5 (Dense)             (None, 256)               229632    
                                                                 
 dense_6 (Dense)             (None, 7)                 1799      
                                                                 
=================================================================
Total params: 269,959
Trainable params: 269,959
Non-trainable params: 0
_________________________________________________________________
```
## Results

In this phase, we decided to run various models with different normalization techniques and data options for a limited number of epochs (20, in this case). After this preliminary investigation, we can determine which model performs best for our problem and further investigate it.

From the tables, it is evident that using the StandardScaler for normalization consistently yields better results in each case compared to the MinMaxScaler. This difference in performance is likely attributed to the fact that the StandardScaler is more resilient to outliers than the MinMaxScaler.

Regarding the models, there isn't a definitive winner that stands out. However, when we exclude the unclassified data, model 2 demonstrates superior performance compared to the other models. Conversely, when including the unclassified data, model 3 shows better performance than the other models. These observations suggest that the effectiveness of the models might be influenced by the presence of unclassified data and the specific data context in each scenario.

**Not using unclassified data:**

| manipulation    | scaler        | model_name | average_accuracy |
|-----------------|---------------|------------|------------------|
| no unclassified | StandardScaler| model_2    | 0.768782435129741|
| no unclassified | StandardScaler| model_3    | 0.768086683775307|
| no unclassified | StandardScaler| model_1    | 0.76311377245509 |
| no unclassified | MinMaxScaler  | model_2    | 0.759190191046478|
| no unclassified | MinMaxScaler  | model_3    | 0.749426860564585|
| no unclassified | MinMaxScaler  | model_1    | 0.74783005417736 |

**Using unclassifed data:**

| manipulation  | scaler        | model_name | average_accuracy |
|---------------|---------------|------------|------------------|
| unclassified  | StandardScaler| model_3    | 0.642158056561844|
| unclassified  | StandardScaler| model_2    | 0.640316295777649|
| unclassified  | StandardScaler| model_1    | 0.630262717282109|
| unclassified  | MinMaxScaler  | model_2    | 0.633185180675211|
| unclassified  | MinMaxScaler  | model_3    | 0.623382751377515|
| unclassified  | MinMaxScaler  | model_1    | 0.613884745349935|

Also at the end of the pdf in the images section we have provided a confusion matrix to better explain the result.

## K-fold

K-fold cross-validation is a widely used technique in machine learning to assess the performance and generalization ability of a model. It involves partitioning the dataset into k equally sized folds, where k is a pre-defined value. The model is then trained and evaluated k times, each time using a different fold as the validation set and the remaining folds as the training set. This process allows us to obtain k sets of evaluation metrics, which are then averaged to provide a more robust estimate of the model's performance.

*So, we further investigated model 2 by eliminating the unclassified data in a 10-fold cross-validation with StandardScaler normalization:*

| Index | Accuracy eliminare unclassified|
|-------|----------------------|
| 1     | 0.7736843305883158   |
| 2     | 0.7742546250883956   |
| 3     | 0.7743458722084084   |
| 4     | 0.7754636494285649   |
| 5     | 0.774236963365117    |
| 6     | 0.7742597746247548   |
| 7     | 0.7724804963730098   |
| 8     | 0.776860258223459    |
| 9     | 0.7734157580181578   |
| 10    | 0.7745739900084404   |
| Average| 0.7743575717926624|

*And with unclassified data using the same normalization technique but model 3:*

| Index | Accuracy unclassified|
|-------|----------------------|
|1      |0.64492207560310100|
|2      |0.64016480273069900|
|3      |0.63174276922631000|
|4      |0.63247296595841300|
|5      |0.63203139471985700|
|6      |0.65296461385378700|
|7       |0.63482979946338000|
|8       |0.63916353538767000|
|9       |0.64159487202740200|
|10      |0.65337037281582800|
|Average  |0.64032572017864400|

## Conclusion

One of the primary challenges we encountered with this dataset is the pronounced overexpression of certain interactions. This imbalance poses difficulties in effectively predicting them using traditional Neural Networks. Surprisingly, even when exclusively predicting the **HBOND** interaction, the NN achieves remarkably high accuracy. To address this issue, we believe it is crucial to leverage the data and strive for a more balanced dataset. The problem of manipulation the data where limited to the fact that the task give us the first dataset and we have to work with it.

This leads us to the challenging decision of determining the most suitable course of action for handling this data. Furthermore, the incorporation of novel features and interactions has not only increased the dataset's complexity but also enhanced its realism, capturing the intricacies of real-world scenarios, where multiple interactions can coexist.

Furthermore, we attempted to perform a basic training without adding any data and used our model, which resulted in very poor performance. In comparison, when we added additional data, the prediction of interactions significantly improved. This suggests that incorporating more data is an effective approach to enhance the performance of our interaction prediction model.

## Future Work

A potential avenue for future research could involve implementing a mechanism to "penalize" mispredictions based on the overrepresentation of certain features. By introducing this approach, the model would become more sensitive to the imbalanced distribution and be incentivized to improve its performance on the underrepresented features. This would allow us to achieve a more balanced prediction.

## Images

![](data/images/CM_NO_unclassified.png)

**Figure 4**: The image represents a confusion matrix, where the diagonal elements show the correct predictions. Above the matrix, there is a bar histogram representing the predicted classes, and on the right side, there is another histogram showing the distribution of correct labels. Notably, both histograms are very similar, indicating that the predictions closely match the actual labels. This similarity between the two histograms suggests that the model's predictions were accurate and aligned well with the true data.
Below also there is the legend of the prediction labels

**Labels for figure 4:**

| Index | Interactions                     |
|-------|----------------------------------|
| 0     | {'HBOND', 'PICATION', 'VDW'}     |
| 1     | {'HBOND', 'PICATION'}            |
| 2     | {'HBOND', 'SSBOND', 'VDW'}       |
| 3     | {'HBOND', 'VDW'}                 |
| 4     | {'HBOND'}                        |
| 5     | {'IONIC', 'HBOND', 'VDW'}        |
| 6     | {'IONIC', 'HBOND'}               |
| 7     | {'IONIC', 'VDW'}                 |
| 8     | {'IONIC'}                        |
| 9     | {'PICATION', 'VDW'}              |
| 10    | {'PICATION'}                     |
| 11    | {'PIPISTACK', 'HBOND', 'VDW'}    |
| 12    | {'PIPISTACK', 'HBOND'}           |
| 13    | {'PIPISTACK', 'VDW'}             |
| 14    | {'PIPISTACK'}                    |
| 15    | {'SSBOND', 'VDW'}                |
| 16    | {'VDW'}                         |

![](data/images/CM_unclassified.png)

**Figures 5**: The image represents a confusion matrix, where the diagonal elements show the correct predictions. Above the matrix, there is a bar histogram representing the predicted classes, and on the right side, there is another histogram showing the distribution of correct labels. Notably, both histograms are very similar, indicating that the predictions closely match the actual labels. In this case the metrix seems more sparse then the other with some row/colums under represented.

**Labels for figure 5:**

| Index | Interactions                   |
|-------|-------------------------------|
| 0     | {'HBOND', 'IONIC'}            |
| 1     | {'HBOND', 'PIPISTACK', 'VDW'} |
| 2     | {'HBOND', 'PIPISTACK'}        |
| 3     | {'HBOND', 'SSBOND', 'VDW'}    |
| 4     | {'HBOND', 'Unclassified', 'VDW'} |
| 5     | {'HBOND', 'Unclassified'}     |
| 6     | {'HBOND', 'VDW', 'IONIC'}     |
| 7     | {'HBOND', 'VDW'}              |
| 8     | {'HBOND'}                    |
| 9     | {'IONIC'}                    |
| 10    | {'PICATION', 'HBOND', 'VDW'}  |
| 11    | {'PICATION', 'HBOND'}         |
| 12    | {'PICATION', 'VDW'}           |
| 13    | {'PICATION'}                 |
| 14    | {'PIPISTACK', 'VDW'}          |
| 15    | {'PIPISTACK'}                |
| 16    | {'SSBOND', 'VDW'}            |
| 17    | {'Unclassified', 'VDW'}       |
| 18    | {'Unclassified'}             |
| 19    | {'VDW', 'IONIC'}             |
| 20    | {'VDW'}                      |






