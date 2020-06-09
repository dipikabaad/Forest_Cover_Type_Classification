import googleapiclient.discovery
# pip install --upgrade google-api-python-client
import pandas as pd
import numpy as np
from sklearn import preprocessing

import json
import itertools

def clean_data(test_df):
    # Drop columns Soil Type 7 and Soil Type 15
    test_df.drop(axis=1, columns=['Soil_Type7','Soil_Type15'], inplace=True)
    
    ## Apply to test_df
    # Convert the Wilderness Area one hot encoded to single column
    columns = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
        'Wilderness_Area4']
    wilderness_types = []
    for index, row in test_df.iterrows():
        dummy = 'Wilderness_Area_NA'
        for col in columns:
            if row[col] == 1:
                dummy = col
                break
        wilderness_types.append(dummy)
    test_df['Wilderness_Areas'] = wilderness_types
    # Convert the Soil Type one hot encoded to single column
    columns = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
        'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type8',
        'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
        'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
        'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
        'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
        'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
        'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
        'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
    soil_types = []
    for index, row in test_df.iterrows():
        dummy = 'Soil_Type_NA'
        for col in columns:
            if row[col] == 1:
                dummy = col
                break
        soil_types.append(dummy)
    test_df['Soil_Types'] = soil_types
    print(pd.value_counts(test_df['Soil_Types']))

    print(pd.value_counts(test_df['Wilderness_Areas']))

    test_df.drop(axis=1, columns=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
        'Wilderness_Area4'], inplace=True)
    test_df.drop(axis=1, columns=['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
        'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type8',
        'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
        'Soil_Type13', 'Soil_Type14', 'Soil_Type16',
        'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
        'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
        'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
        'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
        'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'], inplace=True)

    test_df['Soil_Types'].replace(to_replace={'Soil_Type8': 'Soil_Type_NA', 'Soil_Type25': 'Soil_Type_NA', 'Soil_Type7': 'Soil_Type_NA', 'Soil_Type15': 'Soil_Type_NA'}, inplace=True)
    print(pd.value_counts(test_df['Soil_Types']))

    return test_df

def normalize_data(test_df):
    # Create a min max processor object
    min_max_scaler = preprocessing.MinMaxScaler()
    feature_columns = ['Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
        'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    for column in feature_columns:
        print("Transforming column {}:".format(column))
        test_df[column] = min_max_scaler.fit_transform(test_df[[column]].values.astype(float))

    return test_df

if __name__ == "__main__":
    test_df = pd.read_csv('test.csv')
    test_df = clean_data(test_df)
    test_df = normalize_data(test_df)
    ids = test_df['Id']
    test_df.drop(axis=1, columns=['Id'], inplace=True)

    MODEL_NAME = 'ForestPredictor'
    VERSION_NAME = 'v7'
    PROJECT_ID = 'example_project' # replace this variable with your project name
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}/versions/{}'.format(PROJECT_ID, MODEL_NAME, VERSION_NAME)
    start_index = 0
    end_index = 100
    final_predictions = []
    chunk_size = 100
    if len(test_df) % chunk_size == 0:
        steps = len(test_df) // chunk_size
    else:
        steps = (len(test_df) // chunk_size) + 1
    final_predictions = []
    for step in range(steps):
        if step == steps-1:
            end_index = len(test_df)
        temp_df = test_df[start_index: end_index]
        instances = temp_df.to_json(orient="records")
        # print(instances)
        response = service.projects().predict(
            name=name,
            body={'instances': json.loads(instances)}
        ).execute()

        
        if 'error' in response:
            raise RuntimeError(response['error'])
        else:
            print(response['predictions'][0])
            x = response['predictions']
            y = []
            for value in x:
                y.append(value['output_1'])
            final_predictions.append((np.argmax(y, axis=-1)))
        start_index += chunk_size
        end_index += chunk_size
    final_predictions = list(itertools.chain(*final_predictions))
    # print(final_predictions)
    final_df = pd.DataFrame()
    final_df = final_df.from_dict({'Id': ids, 'Cover_Type': final_predictions})
    final_df.to_csv('final_predictions.csv')
        