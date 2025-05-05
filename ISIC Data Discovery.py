# Mikayla's Preprocessing 

# import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# function for printing maxes, mins, and averages for numerical columns
def print_stats(data,column): 
    print(column)
    print("\tMax --> ", data[column].max())
    print("\tMin --> ",data[column].min())
    print("\tMean --> ",data[column].mean())
    print("\tMode --> ",data[column].mode()[0])

# function for counting missing data 
def countMissingData(data): 
    print ("Missing data Counts:\n", data.isna().sum())

# function for counting total number of records in the data 
def countData(data): 
    print ("Data Count: ", data.index.size)

def preprocess(dataSet, diagnoses_or_no):
    # Dropping the following features for the reasons listed: 
        # unique to each record - isic_id, patient_id, lesion_id
        # doesn't provide necessary context - attribution, copyright_license, image_type, acquisition_day, concomitant_biopsy, diagnosis_confirm_type
        # missing in majority of the data - anatom_site_special, mel_class, mel_thick_mm, mel_ulcer, nevus_type
    processed_dataSet = dataSet.drop(['isic_id','patient_id','lesion_id',
                            'attribution','copyright_license','image_type','acquisition_day','concomitant_biopsy', 'diagnosis_confirm_type',
                            'anatom_site_special','mel_class','mel_thick_mm','mel_ulcer','nevus_type'], axis=1)
        # only 225 pieces of data aren't missing any values

    # Since the model only needs 600 instances to train on, we'll remove records that don't include the most relevant features 
        # fitzpatrick_skin_type - an essential part of our study since the aim to diagnose individuals with diverse skin tones
    processed_dataSet = processed_dataSet.dropna(subset=['fitzpatrick_skin_type'])
        # new data set size is 626 pieces of data
        # 82 records with missing data in the dermoscopic_type column
            # going to keep the data since it determines the type of image and may be beneficial to image processing 
            # replacing missing values with non-polarized
    processed_dataSet['dermoscopic_type'] = processed_dataSet['dermoscopic_type'].replace(np.NaN, 'contact non-polarized')

    if diagnoses_or_no == 1: 
        # dropping diagnosis columns 
        fully_processed_dataSet = preprocess_no_diagnoses(processed_dataSet)
        # put the new dataset in a csv to train the model
        fully_processed_dataSet.to_csv('CSVs/1_processed_no_diagnoses.csv', index=False)
    else: 
        # including diagnosis columns
        fully_processed_dataSet = processed_dataSet
        # put the new dataset in a csv to train the model
        fully_processed_dataSet.to_csv('CSVs/2_processed_with_diagnoses.csv', index=False)
        # analyze diagnosis columns
        diagnoses_analysis(fully_processed_dataSet)

    return (fully_processed_dataSet)

def preprocess_no_diagnoses(processed_dataSet):
    # Dropping Diagnoses columns
    fully_processed_dataSet = processed_dataSet.drop(['diagnosis_1','diagnosis_2','diagnosis_3','diagnosis_4','diagnosis_5'],axis=1)
    return(fully_processed_dataSet)

def correlations(dataSet): 
    # Find Correlations Between Features
        # Using Correlation Matrix
        # convert categorical columns into numerical values 
    categoricalColumns = dataSet.select_dtypes(include=['object', 'bool']).columns
    for column in categoricalColumns:
        encoder = LabelEncoder()
        dataSet[column] = encoder.fit_transform(dataSet[column])

        # correlation for all features
    matrix = dataSet.corr()
    sns.heatmap(matrix, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
        # put into csv file for readability
    matrix = matrix.round(2)
    matrix.to_csv('CSVs/3_correlation_matrix.csv')
        # finding all the positive-strong and negative-strong data (putting into csv)
    strongCorr = matrix[(matrix > 0.10) | (matrix < -0.10)]
    strongCorr.to_csv('CSVs/4_strong_correlation_matrix.csv', index=False)

def diagnoses_analysis(dataSet): 
    # Make copy to avoid copy warning
    diagnosis_list = dataSet[['diagnosis','diagnosis_1','diagnosis_2','diagnosis_3','diagnosis_4','diagnosis_5']].copy()

    # converting objects to strings
    diagnosis_list = diagnosis_list.applymap(str)

    # Normalize any diagnosis that contains "Nevus" to "nevus"
    for column in diagnosis_list.columns:
        diagnosis_list.loc[diagnosis_list[column].str.contains('actinic', case=False, na=False), column] = 'akiec'
        diagnosis_list.loc[diagnosis_list[column].str.contains('basal', case=False, na=False), column] = 'bcc'
        diagnosis_list.loc[diagnosis_list[column].str.contains('solar', case=False, na=False), column] = 'bkl'
        diagnosis_list.loc[diagnosis_list[column].str.contains('seborrheic', case=False, na=False), column] = 'bkl'
        diagnosis_list.loc[diagnosis_list[column].str.contains('lichen', case=False, na=False), column] = 'bkl'
        diagnosis_list.loc[diagnosis_list[column].str.contains('dermatofibroma', case=False, na=False), column] = 'df'
        diagnosis_list.loc[diagnosis_list[column].str.contains('melanoma', case=False, na=False), column] = 'mel'
        diagnosis_list.loc[diagnosis_list[column].str.contains('melanocytic', case=False, na=False), column] = 'nv'
        diagnosis_list.loc[diagnosis_list[column].str.contains('lentigo', case=False, na=False), column] = 'nv'
        diagnosis_list.loc[diagnosis_list[column].str.contains('nevus', case=False, na=False), column] = 'nv'
        diagnosis_list.loc[diagnosis_list[column].str.contains('angio', case=False, na=False), column] = 'vasc'
        diagnosis_list.loc[diagnosis_list[column].str.contains('pyogenic', case=False, na=False), column] = 'vasc'
        diagnosis_list.loc[diagnosis_list[column].str.contains('benign', case=False, na=False), column] = 'nv'
        diagnosis_list.loc[diagnosis_list[column].str.contains('Indeterminate', case=False, na=False), column] = '?'
        diagnosis_list.loc[diagnosis_list[column].str.contains('nan', case=False, na=False), column] = '?'

    # Convert to csv 
    diagnosis_list.to_csv('CSVs/5_diagnosis_analysis.csv', index=False)
    
    # Read diagnosis data
    diagnoses = pd.read_csv('CSVs/5_diagnosis_analysis.csv')

    # Stack all diagnoses into a single column
    diagnosis_all = diagnoses.stack().reset_index(drop=True)

    # Count occurrences
    diagnosis_counts = diagnosis_all.value_counts()
    print (diagnosis_counts)

    # Plot bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values, palette='viridis')
    plt.title('Diagnosis Frequencies')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main(): 
    # Import SCIN dataset 
    # Downloaded filtered data set that didn't include records that were missing the following information:
        # Age, Sex, Anatomic Site, Clinical Size, Family History
        # Only included metadata for images with dermoscopic image type
    dataSet = pd.read_csv('CSVs/metadata.csv')

    # Conduct preprocessing tasks - without diagnoses
    processed_no_diagnoses = preprocess(dataSet, 1)
    # Conduct preprocessing tasks and - with diagnoses
    processed_with_diagnoses = preprocess(dataSet, 2)

    # Find correlations and make heatmap - with diagnoses
    correlations(processed_with_diagnoses)
    # Find correlations and make heatmap - without diagnoses
    correlations(processed_no_diagnoses)

if __name__ == "__main__":
    main() 
