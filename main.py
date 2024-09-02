from src.function_utils import read_localities, identify_correspondences, distribute_locations
from src.map_utils import create_map

if __name__ == "__main__":

    """
    Please ensure that the CSV file is formatted in WKT (Well-Known Text) format.
    The coordinate format should be: POINT (longitude latitude).
    Example:
    POINT (-46.49326400000001 -23.530967), Location 1
    POINT (-46.49326500000001 -23.530968), Location 2 
    """
    locations_path = r"test_files\locations.csv"
    df = read_localities(locations_path, "locations")

    """
    We don't merge this file with the main one because Google Maps may use different
    formats or names, and clients may have registered names that vary in length.
    This discrepancy can cause issues when merging the files directly,
    so we process each file's data separately.
    """
    locations2_path = r"test_files\locations2.xlsx"
    df2 = read_localities(locations2_path, "locations2")

    """
    This file should include all available information about the delivery personnel,
    such as their name, weight capacity, maximum distance they can cover, etc.
    """
    deliverers_path = r"test_files\delivery.xlsx"
    df3 = read_localities(deliverers_path, "delivery")

    """Specify the path where the final output will be saved as an Excel file."""
    final = r"results\output.xlsx"

    df_matches = identify_correspondences(df, df2, 'Locations', 'Locations')
    df_matches.columns = ['Loc1', 'Locations', 'Weights', 'Lat', 'Long']
    df_matches = df_matches.drop('Loc1', axis=1)
    df_distributed_deliveries = distribute_locations(df_matches, df3)
    df_distributed_deliveries.to_excel(final, index=False)
    create_map(df_distributed_deliveries, df_matches)
    print("Mischief managed!")
