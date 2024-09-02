import pandas as pd
import os
import ast
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def treat_localities(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin-1', delimiter=',', header=None, usecols=[0, 1],
                         names=['Coordinates', 'Locations'], engine='python')
        df = treat_dataframe(df)

        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return pd.DataFrame()


def treat_locations2(file_path):
    try:
        df = pd.read_excel(file_path, header=None, skiprows=1, names=['Locations', 'Weights'])
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return pd.DataFrame()


def treat_deliverers(file_path):
    try:
        df = pd.read_excel(file_path, header=0)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return pd.DataFrame()


def read_localities(file_path, type_file):
    function_file = {
        'locations': treat_localities,
        'locations2': treat_locations2,
        'delivery': treat_deliverers
    }
    func = function_file.get(type_file)
    if func and os.path.exists(file_path):
        return func(file_path)
    else:
        print("File not found or invalid file type")
        return pd.DataFrame()


def treat_dataframe(df):
    df[['Lat', 'Long']] = df['Coordinates'].str.extract(r'POINT \(([^ ]+) ([^ ]+)\)')
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce').round(5)
    df['Long'] = pd.to_numeric(df['Long'], errors='coerce').round(5)
    df.drop(columns=['Coordinates'], inplace=True)
    return df


def identify_correspondences(df1, df2, column1, column2, limit=0.5):

    vectorizer = TfidfVectorizer().fit(df2[column2])
    tfidf_matrix = vectorizer.transform(df2[column2])
    results = []

    for item1 in df1[column1]:
        tfidf_vector = vectorizer.transform([item1])
        cosine_similarities = cosine_similarity(tfidf_vector, tfidf_matrix).flatten()
        max_index = cosine_similarities.argmax()
        max_value = cosine_similarities[max_index]

        if max_value >= limit:
            best_match = df2[column2].iloc[max_index]
            pesos_value = df2['Weights'].iloc[max_index]
            long = df1.loc[df1[column1] == item1, 'Long'].values[0]
            lat = df1.loc[df1[column1] == item1, 'Lat'].values[0]
            results.append((item1, best_match, pesos_value, long, lat))
        else:
            results.append((item1, None, None, None, None))

    return pd.DataFrame(results, columns=[column1, column2, 'Weights', 'Lat', 'Long'])


def identify_duplicates_by_coordinates(df):
    return df[df.duplicated(subset=['Lat', 'Long'], keep=False)]


def notify_duplicates(df):
    duplicates = identify_duplicates_by_coordinates(df)
    if not duplicates.empty:
        print("Duplicates found in coordinates (Lat, Long):")
        print(duplicates)
        print("Please check the names in the file")


def find_closest_neighbor(df, lat, long, allocated_locations, max_dist=float('inf')):
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Long'] = pd.to_numeric(df['Long'], errors='coerce')
    df = df.dropna(subset=['Lat', 'Long'])

    if allocated_locations:
        df = df[~df['Locations'].isin(allocated_locations)]

    if not df.empty:
        nn = NearestNeighbors(n_neighbors=1, radius=max_dist, metric='euclidean')
        nn.fit(df[['Lat', 'Long']])
        distance, index = nn.kneighbors([[lat, long]], return_distance=True)

        if distance[0][0] <= max_dist:
            return df.iloc[index[0][0]], distance[0][0]

    return None, None


def distribute_locations(df_correspondences, df_delivery_men, max_dist_acceptable=10):
    delivery_men = df_delivery_men.to_dict('records')
    results = []
    allocated_locations = set()
    used_deliverers = set()

    while len(allocated_locations) < len(df_correspondences):
        deliverer_was_used_inthis_round = False

        for delivery_man in delivery_men:
            name = delivery_man['DELIVERY_MAN']
            if name in used_deliverers:
                continue

            maximum_weight = delivery_man['HEIGHT']
            max_geo = delivery_man['MAX_GEO']
            restrictions = ast.literal_eval(delivery_man['RESTRICT']) if pd.notna(delivery_man['RESTRICT']) else []

            max_geo = float('inf') if pd.isna(max_geo) or max_geo == float('inf') else int(max_geo)

            df_filtered = df_correspondences[~df_correspondences['Locations'].isin(restrictions)]
            df_filtered = df_filtered[~df_filtered['Locations'].isin(allocated_locations)]
            df_filtered = df_filtered[df_filtered['Weights'] <= maximum_weight]

            df_filtered = df_filtered.dropna(subset=['Lat', 'Long'])

            if df_filtered.empty:
                continue

            deliverer_was_used_inthis_round = True
            cluster = []
            total_weight = 0
            lat, long = df_filtered.iloc[0]['Lat'], df_filtered.iloc[0]['Long']

            while total_weight < maximum_weight and len(cluster) < max_geo:
                neighbor, distance = find_closest_neighbor(df_filtered, lat, long, allocated_locations,
                                                           max_dist_acceptable)
                if neighbor is None or (total_weight + neighbor['Weights'] > maximum_weight):
                    break

                cluster.append(neighbor['Locations'])
                total_weight += neighbor['Weights']
                lat, long = neighbor['Lat'], neighbor['Long']
                allocated_locations.add(neighbor['Locations'])
                df_filtered = df_filtered[~df_filtered['Locations'].isin(allocated_locations)]

            if cluster:
                results.append({
                    'Delivery_man': name,
                    'Cluster': f'Cluster_{name}',
                    'Locations': cluster,
                    'Total_Weight': total_weight
                })
                used_deliverers.add(name)

        if not deliverer_was_used_inthis_round:
            break

    df_not_selected = df_correspondences[~df_correspondences['Locations'].isin(allocated_locations)]
    if not df_not_selected.empty:
        unallocated_locations = df_not_selected['Locations'].tolist()
        locations_not_allocated_adjusted = adjust_order(unallocated_locations, df_correspondences)
        results.append({
            'Delivery_man': 'Not Allocated',
            'Cluster': 'Cluster_Not_Allocated',
            'Locations': locations_not_allocated_adjusted,
            'Total_Weight': df_not_selected['Weights'].sum()
        })

    return pd.DataFrame(results)


def create_additional_deliverers(df_not_selected, df_outliers):
    if not df_not_selected.empty:
        return pd.DataFrame({
            'DELIVERY_MAN': ['Delivery_New'] * len(df_not_selected),
            'HEIGHT': [float('inf')] * len(df_not_selected),
            'MAX_GEO': [float('inf')] * len(df_not_selected),
            'RESTRICT': [''] * len(df_not_selected)
        })
    return pd.DataFrame()

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_total_distance(locations, df2):
    total_distance = 0
    for i in range(len(locations) - 1):
        coord1 = (df2[df2['Locations'] == locations[i]]['Lat'].values[0],
                  df2[df2['Locations'] == locations[i]]['Long'].values[0])
        coord2 = (df2[df2['Locations'] == locations[i + 1]]['Lat'].values[0],
                  df2[df2['Locations'] == locations[i + 1]]['Long'].values[0])
        total_distance += haversine(coord1, coord2)
    return total_distance

def adjust_order(locations, df2):
    current_order = [locations[0]]
    remaining_localities = set(locations[1:])

    while remaining_localities:
        last_locality = current_order[-1]
        closest_neighbor = None
        smallest_distance = float('inf')

        for locality in remaining_localities:
            last_coordinate = (df2[df2['Locations'] == last_locality]['Lat'].values[0],
                            df2[df2['Locations'] == last_locality]['Long'].values[0])
            locality_coordinate = (df2[df2['Locations'] == locality]['Lat'].values[0],
                                df2[df2['Locations'] == locality]['Long'].values[0])
            dist = haversine(last_coordinate, locality_coordinate)

            if dist < smallest_distance:
                smallest_distance = dist
                closest_neighbor = locality

        current_order.append(closest_neighbor)
        remaining_localities.remove(closest_neighbor)

    def best_order_with_substitution(locations, df2):
        best_order = locations
        smallest_distance = calculate_total_distance(locations, df2)

        for _ in range(len(locations)):
            new_order = [locations[-1]] + locations[:-1]
            current_distance = calculate_total_distance(new_order, df2)

            if current_distance < smallest_distance:
                best_order = new_order
                smallest_distance = current_distance
            locations = new_order

        new_order = [best_order[-1]] + best_order[:-1]
        current_distance = calculate_total_distance(new_order, df2)

        if current_distance < smallest_distance:
            best_order = new_order

        return best_order

    final_order = best_order_with_substitution(current_order, df2)

    return final_order

