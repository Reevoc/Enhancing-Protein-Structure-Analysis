# Enhancing Protein Structure Analysis 
## Introduction
The Residue Interaction Network Generator (RING) and Ribbon Diagrams have revolutionized the study of protein structures, enabling researchers to delve into the intricate details of residue interactions.
The aim of this powerfull tool in the field of structural biology is to provide a simple and intuitive way to visualize the interactions between residues in a protein structure.
In our case we are going to develop a software that will be able to predict the interactions between residues in a protein structure, given a dataset of protein structures and its corresponding interactions.
In our code, we have prioritized flexibility, allowing us to experiment with different configurations and observe the prediction results in the output. This adaptability empowers us to fine-tune our analysis and explore various scenarios effortlessly.

## Example of usage
In the README of your GitHub repository, you can include the following explanation to guide users on how to use the `main.py` script with various parameters:

## Running the Script

The `main.py` script allows you to train and evaluate different models with various normalization techniques and data options. To run the script, you can use the following command-line arguments:

```bash
python3 main.py -m [model] -n [normalization] -d [data_option]
```

**Parameters:**

- `[model]`: Choose one of the available models: "model_1", "model_2", or "model_3". Each model represents a specific machine learning model or architecture.
- `[normalization]`: Choose one of the available normalization techniques: "MinMaxScaler", "StandardScaler", or "no_normalization". This parameter allows you to preprocess the data before training the models.
- `[data_option]`: Choose one of the available data options: "eliminate_unclassified" or "unclassified". This option determines how the unclassified data is handled during training and evaluation.

**Example:**

```bash
python3 main.py -m model_1 -n MinMaxScaler -d eliminate_unclassified
python3 main.py -m model_1 -n MinMaxScaler -d unclassified
# ... (similar commands for other models and combinations)
```

Feel free to experiment with different combinations to find the best configuration for your specific use case.

By following these instructions, users can easily utilize the `main.py` script and explore various models, normalization techniques, and data options to identify the optimal configuration for their machine learning tasks.

**REMEMBER**: The dataset is not included in the repository, so you have to download create it with the **collect_data.py** script.

# Dataset
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
- **ss3**: 
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

## Looking at the data

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
## Models


## K-fold
K-fold cross-validation is a widely used technique in machine learning to assess the performance and generalization ability of a model. It involves partitioning the dataset into k equally sized folds, where k is a pre-defined value. The model is then trained and evaluated k times, each time using a different fold as the validation set and the remaining folds as the training set. This process allows us to obtain k sets of evaluation metrics, which are then averaged to provide a more robust estimate of the model's performance.

In summary, k-fold cross-validation helps to mitigate the potential bias introduced by a single train-test split and provides a more reliable evaluation of the model's performance across different data subsets. This technique is widely regarded for its ability to yield more accurate and stable performance metrics, making it a valuable tool in assessing the efficacy of machine learning models.

## Results








