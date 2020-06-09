from sklearn import preprocessing
import pandas as pd

class Forest_Transformer(object):
  def __init__(self):
    self._drop_id = True

  def preprocess(self, data):
    # Convert the Wilderness Area one hot encoded to single column
    columns = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
        'Wilderness_Area4']
    wilderness_types = []
    for index, row in data.iterrows():
        dummy = 'Wilderness_Area_NA'
        for col in columns:
            if row[col] == 1:
                dummy = col
                break
        wilderness_types.append(dummy)
    data['Wilderness_Areas'] = wilderness_types
    # Convert the Soil Type one hot encoded to single column
    columns = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
        'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
        'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
        'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
        'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
        'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
        'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
        'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
        'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
    soil_types = []
    for index, row in data.iterrows():
        dummy = 'Soil_Type_NA'
        for col in columns:
            if row[col] == 1:
                dummy = col
                break
        soil_types.append(dummy)
    data['Soil_Types'] = soil_types

    data.drop(axis=1, columns=['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
        'Wilderness_Area4'], inplace=True)
    data.drop(axis=1, columns=['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
        'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
        'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
        'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
        'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
        'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
        'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
        'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
        'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'], inplace=True)

    data['Soil_Types'].replace(to_replace={'Soil_Type8': 'Soil_Type_NA', 'Soil_Type25': 'Soil_Type_NA', 'Soil_Type7': 'Soil_Type_NA', 'Soil_Type15': 'Soil_Type_NA'}, inplace=True)
    # print(pd.value_counts(data['Soil_Types']))

    if self._drop_id:
        data.drop(axis=1, columns=['Id'], inplace=True)

    feature_columns = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    
    # Create a min max processor object
    min_max_scaler = preprocessing.MinMaxScaler()
    for column in feature_columns:
        data[column] = min_max_scaler.fit_transform(data[[column]].values.astype(float))

    return data