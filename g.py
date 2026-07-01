import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def load_data(file_path):
    """
    Load CSV with error handling.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found.")
    except pd.errors.EmptyDataError:
        print("No data in file.")
    except pd.errors.ParserError:
        print("Error parsing file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")      

def mask_data(df):
    """
    Mask data to include only asteroids with a diameter greater than 100 meters.
    """
    df['Close Approach Date'] = pd.to_datetime(df['Close Approach Date'], errors='coerce')
    filtered_df = df[df['Close Approach Date'].dt.year >= 2000]
    return filtered_df

def data_details(df):
    """
    Get details of the DataFrame.
    """
    columns_to_remove = ['Neo Reference ID', 'Orbiting Body', 'Equinox']
    df_clean = df.drop(columns=columns_to_remove, errors='ignore')

    # create a tuple with the required details
    if df_clean.empty:
        return (0, 0, [])

    # count rows ,count coloumns, count headers
    return (df_clean.shape[0], df_clean.shape[1], list(df_clean.columns))

def max_absolute_magnitude(df):
    """
    Find the maximum absolute magnitude in the DataFrame.
    """
    if 'Absolute Magnitude' not in df.columns:
        return None

    max_row = df.loc[df['Absolute Magnitude'].idxmax()]
    name = max_row['Name']
    max_magnitude = max_row['Absolute Magnitude']
    return (int(name), float(max_magnitude))  

def closest_to_earth(df):
    """
    Find the closest Near-Earth Object (NEO) to Earth based on the 'Miss Dist.(kilometers)' column.
    """
    if 'Miss Dist.(kilometers)' not in df.columns:
        return None

    max_row = df.loc[df['Miss Dist.(kilometers)'].idxmin()]
    return max_row['Name']

def common_orbit(df):
    """
    Find the most common orbit type among the Near-Earth Objects (NEOs).
    """

    dict = df['Orbit ID'].value_counts().to_dict()
    return dict

def min_max_diameter(df):
    """
    Find the maximum and minimum diameters of Near-Earth Objects (NEOs).
    """
    average_diameter = df['Est Dia in KM(max)'].mean()
    filtered_df = df[df['Est Dia in KM(max)'] > average_diameter]
    return filtered_df.shape[0]


def plt_hist_diameter(df):
    """
    Displays a histogram of asteroid counts by their average estimated diameter (in KM).
    The average diameter is calculated as the average of the min and max estimated diameters.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing asteroid data.
        
    Returns:
        None
    """
    # מחשבים את הקוטר הממוצע
    avg_diameter = (df['Est Dia in KM(min)'] + df['Est Dia in KM(max)']) / 2

    # מציגים היסטוגרמה
    plt.figure(figsize=(10, 6))
    plt.hist(avg_diameter, bins=100, color='skyblue', edgecolor='black')

    # הגדרות גרפיות
    plt.title('Distribution of Average Asteroid Diameter (KM)')
    plt.xlabel('Average Diameter (KM)')
    plt.ylabel('Number of Asteroids')
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def plt_hist_common_orbit(df):
    """
    Displays a histogram of asteroids by their Minimum Orbit Intersection (AU).
    The histogram is divided into 10 equal-width bins from the minimum to the maximum value.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing asteroid data.

    Returns:
        None
    """
    col = 'Minimum Orbit Intersection'
    
    if col not in df.columns:
        print(f"Column '{col}' not found in DataFrame.")
        return

    values = df[col].dropna().astype(float)

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=10, color='lightblue', edgecolor='black')
    plt.title('Asteroids by Minimum Orbit Intersection')
    plt.xlabel('Minimum Orbit Intersection (AU)')
    plt.ylabel('Number of Asteroids')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plt_pie_hazard(df):
    """
    Plot a pie chart showing the proportion of hazardous and non-hazardous asteroids.

    Parameters:
        df (pd.DataFrame): DataFrame containing a column named 'Hazardous' (boolean or string).

    Returns:
        None. Displays a pie chart.
    """
    counts = df['Hazardous'].value_counts()
    
    labels = counts.index.tolist()  # safer in case values are 'Y'/'N' or True/False
    colors = ['salmon', 'skyblue']
    explode = [0.05 if label == True or label == 'Y' else 0 for label in labels]

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
            explode=explode, startangle=90)

    plt.title('Hazardous vs Non-Hazardous Asteroids')
    plt.tight_layout()
    plt.show()


def plt_linear_motion_magnitude(df):
    """
    Plot a scatter plot of miss distance vs. velocity of asteroids, with a linear regression line.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Miss Dist.(kilometers)' and 'Miles per hour' columns.

    Returns:
        None. Displays a regression plot.

    Notes:
        Based on the visual output, the data does not show a strong linear correlation.
        The velocity of an asteroid does not appear to consistently increase or decrease
        with miss distance. Therefore, no meaningful linear relationship is observed.
    """
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Miss Dist.(kilometers)', 'Miles per hour'])

    plt.figure(figsize=(10, 6))
    sns.regplot(
        x='Miss Dist.(kilometers)',
        y='Miles per hour',
        data=df,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red'},
        ci=None
    )

    plt.title('Miss Distance vs Asteroid Velocity')
    plt.xlabel('Miss Distance (km)')
    plt.ylabel('Velocity (mph)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()