import streamlit as st
from PIL import Image
from st_click_detector import click_detector
import base64
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import base64
from pathlib import Path
from functions import display_image, img_to_html, img_to_bytes, dataframe_to_html, simplify_comment, background_colorize, sort_by_mean_difference, save_shapefile_to_zip, sort_by_mean
from functions import apply_page_config_and_styles, create_map
import firebase_admin
import time
import geopandas as gpd
import pandas as pd
import tempfile
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from streamlit_folium import st_folium
#from image_function import load_image_and_prediction, modify_image, display_image_API, load_img_only
from datetime import datetime



st.set_page_config(layout="wide")

apply_page_config_and_styles()

###################  INITIALIZATION ####################
# Initialiser Firestore et Firebase Storage
if 'firebase_initialized' not in st.session_state:
    if not firebase_admin._apps:
        # Utiliser la clé du compte de service Firebase
        firebase_secrets = st.secrets["firebase"]

        # Create credentials using Streamlit secrets
        cred = credentials.Certificate({
            "type": firebase_secrets["type"],
            "project_id": firebase_secrets["project_id"],
            "private_key_id": firebase_secrets["private_key_id"],
            "private_key": firebase_secrets["private_key"],
            "client_email": firebase_secrets["client_email"],
            "client_id": firebase_secrets["client_id"],
            "auth_uri": firebase_secrets["auth_uri"],
            "token_uri": firebase_secrets["token_uri"],
            "auth_provider_x509_cert_url": firebase_secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": firebase_secrets["client_x509_cert_url"]
        })
        #cred = credentials.Certificate("./koordinat-12a82-firebase-adminsdk-37ljz-3c2a5c77b3.json")
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'koordinat-12a82.appspot.com'  # Remplace par ton ID de projet Firebase
        })
    st.session_state.firebase_initialized = True

# Initialisation du client Firestore et du bucket de stockage
db = firestore.client()
bucket = storage.bucket('koordinat-12a82.appspot.com')  # Accéder au bucket de Firebase Storage


def load_feedback_data():
    feedback_ref = db.collection('feedback')
    feedback_docs = feedback_ref.stream()

    feedback_data = []
    for doc in feedback_docs:
        feedback_data.append(doc.to_dict())

    return pd.DataFrame(feedback_data)


@st.cache_data
def download_shapefile_from_firebase(filename_base):
    temp_dir = tempfile.mkdtemp()
    extensions = ['shp', 'shx', 'dbf', 'prj', 'cpg']

    for ext in extensions:
        blob_path = f'final_prediction/{filename_base}.{ext}'
        blob = bucket.blob(blob_path)
        local_file_path = os.path.join(temp_dir, f"{filename_base}.{ext}")
        blob.download_to_filename(local_file_path)
    
    return os.path.join(temp_dir, f"{filename_base}.shp")


def download_csv_from_firebase(filename_base):
    temp_dir = tempfile.mkdtemp()
    blob_path = f'municipalities/{muni}/final_prediction/{filename_base}.csv'
    blob = bucket.blob(blob_path)
    local_file_path = os.path.join(temp_dir, f"{filename_base}.csv")
    blob.download_to_filename(local_file_path)
    return local_file_path

@st.cache_data
def load_predicted_gdf():
    shapefile_path = download_shapefile_from_firebase('merged_shapefile')
    predicted_gdf = gpd.read_file(shapefile_path)
    return predicted_gdf


@st.cache_data


def download_shapefile_prediction(year):
    temp_dir = tempfile.mkdtemp()
    extensions = ['shp', 'shx', 'dbf']

    for ext in extensions:
        blob_path = f'municipalities/{muni}/predictions/{year}/Koordinat_{year}_stats.{ext}'
        blob = bucket.blob(blob_path)
        local_file_path = os.path.join(temp_dir, f"Koordinat_{year}_stats.{ext}")
        blob.download_to_filename(local_file_path)
    
    return os.path.join(temp_dir, f"Koordinat_{year}_stats.shp")

@st.cache_data
def load_mask(year):
    shapefile_path = download_shapefile_prediction(year)
    predicted_gdf = gpd.read_file(shapefile_path)
    return predicted_gdf


@st.cache_data
def download_shapefile_label():
    temp_dir = tempfile.mkdtemp()
    extensions = ['shp', 'shx', 'dbf']

    for ext in extensions:
        blob_path = f'municipalities/{muni}/labels/{muni}_label.{ext}'
        blob = bucket.blob(blob_path)
        local_file_path = os.path.join(temp_dir, f"{muni}_label.{ext}")
        blob.download_to_filename(local_file_path)
    
    return os.path.join(temp_dir, f"{muni}_label.{ext}")


@st.cache_data
def load_mask(year):
    shapefile_path = download_shapefile_prediction(year)
    predicted_gdf = gpd.read_file(shapefile_path)
    return predicted_gdf



@st.cache_data
def load_label():
    shapefile_path = download_shapefile_label()
    predicted_gdf = gpd.read_file(shapefile_path)
    return predicted_gdf

@st.cache_data
def load_csv_from_firebase():
    shapefile_path = download_csv_from_firebase('merged_shapefile')
    #csv to dataframe
    predicted_gdf = pd.read_csv(shapefile_path)

    # Convert each column from int to str
    begin_year = 2016
    end_year = 2023
    for year in range(begin_year, end_year + 1):
        predicted_gdf[str(year)] = predicted_gdf[str(year)].astype(str)
    return predicted_gdf




#Put this function in functions.py
def check_credentials(username, password):
    credentials = st.secrets["credentials"]
    if username in credentials and password == credentials[username]:
        return True
    return False

# Check if the user is logged in
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log ind"):
        if check_credentials(username, password):
            st.session_state.logged_in = True
            st.success("Logged in successfully")
            st.rerun()
        else:
            st.error("Invalid username or password")
else:
    #st.header("CONTENT HERE!")








    muni = "roskilde" #"høje-taastrup"#
    label_gdf = load_label()


    try:
        predicted_gdf = load_csv_from_firebase()
        
        #csv to dataframe
    except Exception as e:
        st.error(f"Erreur lors du chargement du shapefile : {e}")


    try:
        feedback_df = load_feedback_data()
        if not feedback_df.empty:
            
            # Merge the feedback data with the GeoDataFrame
            predicted_gdf = predicted_gdf[["Objekt_id", "2016","2017","2018","2019","2020","2021","2022","2023","AI Kommentar","geometry",'Systid_fra']].merge(feedback_df[['Objekt_id', 'agreeness', 'kommentar']], on='Objekt_id', how='left')
            
    except Exception as e:
        st.error(f"Error loading feedback data: {e}")


    def save_feedback_to_firestore(objekt_id, agreeness, comment):
        feedback_ref = db.collection('feedback').document(str(objekt_id))
        feedback_ref.set({
            'Objekt_id': objekt_id,
            'agreeness': agreeness,
            'kommentar': comment,
            'datetime': datetime.now()
        })



    @st.cache_data
    def download_image_from_firebase(filename):
        """Télécharge une image depuis Firebase Storage."""
        temp_dir = tempfile.mkdtemp()
        blob = bucket.blob(filename)
        local_file_path = os.path.join(temp_dir, os.path.basename(filename))
        blob.download_to_filename(local_file_path)
        return local_file_path

    # Nom du fichier dans Firebase Storage
    image_filename = 'images/logokoordinat.png'  # Chemin dans Firebase

    local_image_path = download_image_from_firebase(image_filename)

    def initialize_session_state():
        if "filtered_df" not in st.session_state:
            st.session_state.filtered_df = display_gdf.copy()

        if "selected_colors" not in st.session_state:
            st.session_state.selected_colors = {}
        if "number_of_rows" not in st.session_state:
            st.session_state.number_of_rows = len(display_gdf)
        if "search_comment" not in st.session_state:
            st.session_state.search_comment = ""
        if "last_date" not in st.session_state:
            st.session_state.last_date = [min_date, max_date]
            
        




    st.markdown(
        """
        <style>
        .fixed-bottom-right {
            position: fixed;
            bottom: 10px;
            right: 10px;
            z-index: 100;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='fixed-bottom-right'>{img_to_html(local_image_path)}</div>", 
        unsafe_allow_html=True
    )
    color_dict = {
        "1": 'darkred', "2": 'firebrick', "3": 'red', "4": 'grey', "5": 'lightgreen', "6": 'forestgreen', "7": 'darkgreen', "5/4" : 'red', "8" : "blue" }

    colors = [
        ("1", "darkred"),
        ("2", "firebrick"),
        ("3", "red"),
        ("4", "grey"),
        ("5", "lightgreen"),
        ("6", "forestgreen"),
        ("7", "darkgreen"),
        ("8", "blue"),
    ]

    color_map = {label: color for label, color in colors}

    def apply_color(val):
        return f'background-color: {color_map.get(val, "white")}; color: {color_map.get(val, "black")}'

    display_gdf = predicted_gdf
    objekt_id_to_display = None
    display_gdf["Systid_fra"] = pd.to_datetime(display_gdf["Systid_fra"])
    min_date = display_gdf['Systid_fra'].min().date() if not display_gdf.empty else datetime.now().date()
    max_date = display_gdf['Systid_fra'].max().date() if not display_gdf.empty else datetime.now().date()


    def reset_checkboxes():
        for year in ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]:
            for color in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                checkbox_key = f"checkbox_{color_dict[color]}_{year}"
                st.session_state[checkbox_key] = False  # Reset the checkbox
        st.session_state.search_objekt_id = ""
        st.session_state.number_of_rows = len(display_gdf)
        st.session_state.search_comment = ""
        st.session_state.last_date = [min_date, max_date]



    display_gdf = display_gdf[['Objekt_id', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', 'AI Kommentar', 'Systid_fra', 'agreeness', 'kommentar', 'geometry']]






    initialize_session_state()

    #### BUTTON FILTER #####


    if "showfilters" not in st.session_state:
        st.session_state["showfilters"] = False

    def toggle_filters():
        if st.session_state["showfilters"]:
            del st.session_state["showfilters"]
            st.session_state["showfilters"] = False
        else:
            del st.session_state["showfilters"]
            st.session_state["showfilters"] = True

    filter_container = st.container(border=True)






    # Apply CSS to position each widget
    st.markdown(
        """
        <style>
        /* Style for "Vis/skjul filtre" button - far left */
        .st-emotion-cache-ue6h4q e1y5xkzn3 {
            position: absolute;
            left: 0px;
            top: 10px;
        }

        /* Style for the slider - center */
        .st-ae st-af st-ag {
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            top: 10px;
            width: 50%;
        }

        .stMarkdownContainer {
            font-weight: bold;
        }


        .st-emotion-cache-oj1fi.ewgb6652 {
        display: flex;
        -webkit-box-align: center;
        align-items: center;
        margin-top: 10px;
        position: relative;
        left: 96%;
        }
        /
        </style>
        """,
        unsafe_allow_html=True
    )



    # Add "Vis/skjul filtre" button
    filter_container.button("Vis/skjul filtre", on_click=toggle_filters)

    # Add year slider
    filter_container.slider("", min_value=2016, max_value=2022, value=(2016, 2023), step=1, key="year_slider")



    if st.session_state["showfilters"]:
        st.markdown('<style>.st-emotion-cache-4uzi61 .st-emotion-cache-ocqkz7{display:none}</style>', unsafe_allow_html=True)
        
    #class="st-emotion-cache-ocqkz7 e1f1d6gn5"

    box_container = filter_container.container(border=False)








    selected_colors = {}

    spacing = 0.3
    selected_years = range(st.session_state.year_slider[0], st.session_state.year_slider[1] + 1)
    year_cols = box_container.columns([1, spacing] * len(selected_years))

    for year_idx, year in enumerate(selected_years):
        year_cols[year_idx * 2].subheader(str(year))
        cols = year_cols[year_idx * 2].columns(8)

        selected_colors[str(year)] = []

        for idx, color in enumerate(["1", "2", "3", "4", "5", "6", "7", "8"]):
            checkbox_key = f"checkbox_{color_dict[color]}_{year}"

            # Initialize the checkbox state in session_state if not present
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = False

            with cols[idx]:
                st.markdown(
                    f"<div style='height: 10px; width: 19px; background-color: {color_dict[color]}; border-radius: 5px; border-color: black; border-width: 1em;'></div>",
                    unsafe_allow_html=True
                )

                is_selected = st.checkbox(
                    "",
                    key=checkbox_key
                )

                if is_selected:
                    selected_colors[str(year)].append(color)

    seach_cols = box_container.columns(3)

    st.markdown(
        """
        <style>
        /* Set font size for text inputs */
        input {
            font-size: 1.25rem !important;
        }

        /* Set font size for date input */
        .stDateInput > div {
            font-size: 1.25rem !important;
        }

        /* Set font size for the labels */
        label {
            font-size: 1.25rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    if "search_objekt_id" not in st.session_state:
        st.session_state.search_objekt_id = ""

    unique_kommentars = sorted(display_gdf['AI Kommentar'].dropna().unique())
    with seach_cols[0]:
        id_selected = st.text_input("Filtrer efter objekt_id", key="search_objekt_id")
    with seach_cols[1]:
        selected_comment = st.text_input("Filtrer efter kommentar", key="search_comment")
    with seach_cols[2]:
        # 2010-12-06 Systid_fra filter datetime range picker
        selected_date_range = st.date_input(
            "Filtrer efter dato", 
            [min_date, max_date],
            key = "last_date"
        )
        # Ensure the dates are selected before reloading the code
        if not selected_date_range or len(selected_date_range) != 2:
            st.markdown("<div align='center'><h1>Select another date</h1></div>", unsafe_allow_html=True) #st.write("Select another date")
            st.stop()

    start_date = pd.Timestamp(selected_date_range[0])
    end_date = pd.Timestamp(selected_date_range[1])




    def display_le_gdf(filtered_df):

        st.write("Select an Objekt_id to display the image")

        image_filename = 'images/Arrow down-left.png'  # Chemin dans Firebase

        local_image_path_2 = download_image_from_firebase(image_filename)
        st.markdown(
            f"<div style='width: 50px;margin-top:-20px'>{img_to_html(local_image_path_2)}</div>", 
            unsafe_allow_html=True
        )
        col1, col2 = box_container.columns([1, 10])
        with col1:
            row_int_text = st.number_input("Antal rækker", min_value=0,key="number_of_rows")  #max_value=len(filtered_df)
        
        
        if row_int_text:
            filtered_df = filtered_df.head(row_int_text)

        event = st.dataframe(
            filtered_df[["Objekt_id"] + year_columns + ["AI Kommentar"] + ["Systid_fra"] + ['agreeness'] + ['kommentar']].style.map(
                apply_color, subset=year_columns
            ),
            column_config={
                "Objekt_id": st.column_config.Column(
                    width="small"
                ),
            },

            use_container_width=True,
            on_select="rerun",
            selection_mode='single-row',  # Add selection mode for each row
            hide_index=True,  # Drop index column
        )

        if len(filtered_df) == len(display_gdf):
            st.markdown(f"<div align='center'>Total Længde: {len(filtered_df)}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div align='center'>Filtreret: {len(filtered_df)} ud af {len(display_gdf)} valgte områder</div>", unsafe_allow_html=True)

        def output_path(folder_name):
            return os.path.join(tempfile.gettempdir(), folder_name)

        gdf_2023 = load_mask(2023)
        filtered_df = gpd.GeoDataFrame(filtered_df, geometry=gdf_2023.geometry)
        filtered_df = filtered_df[["Objekt_id"] + ["2023"] +["AI Kommentar"] +["geometry"]]
        output_path = output_path("filtered_shapefile")
        #st.write(output_path)
        zip_file_buffer = save_shapefile_to_zip(filtered_df, output_path)
        
        # Provide the ZIP file as a download button
        st.download_button(
            label="Download Shapefile",
            data=zip_file_buffer,
            file_name="filtered_shapefile.zip",
            mime
            ="application/zip"
        )




        return event



    # Add Reset button
    filter_container.button("Gendan filtre", on_click=reset_checkboxes, key="reset_button", help="Reset filter")

    #____________________OUT OF FILTER_____________
    sorting_buttons_style = '''
    <style>
    div {
        display: flex;
        width: 100%;

    }
    .button-container a {
    display: flex;
    flex: 1;
    margin: 0px 5px; /* Horizontal margin for spacing */
    background-color: #052418;



    }
    body {
        font-family: Arial, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0; /* Remove default margin */
        width: 100%;
        background-color: #052418;

    }
    .button-container {
        display: flex;
        width: 100%;
        background-color: #052418;/* Fill the full width of the container */
        
    }


    .button-container button {
        flex: 1; /* Each button takes equal space */
        padding: 10px;
        font-size: 16px;
        color: black; /* Button text color */
        border-radius: 0.5rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        cursor: pointer; /* Change cursor on hover */
        transition: background-color 0.3s; /* Smooth transition for hover effect */
        margin: 0px 50px; /* Horizontal margin for spacing */
        background-color: background-color: #052418;
    
    }

    .stDownloadButton {
        flex: 1; /* Each button takes equal space */
        padding: 10px;
        font-size: 16px;
        color: black; /* Button text color */
        border-radius: 0.5rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        cursor: pointer; /* Change cursor on hover */
        transition: background-color 0.3s; /* Smooth transition for hover effect */
        margin: 0px 50px; /* Horizontal margin for spacing */
        background-color: background-color: #052418;
            
        }

    /* Optional: To handle margins for the first and last buttons */
    .button-container button:first-child {
        margin-left: 0;
    }

    .button-container button:last-child {
        margin-right: 0;
    }
    a:hover {
        text-decoration: none;
    }
    button:hover {
        border-color: red;
        color: red;
    }



    button.clicked {
        background-color: red;
        color: white;
    }
    </style>
    '''

    filtered_df = st.session_state.filtered_df
    #year of the slider selection 
    year_columns = [str(year) for year in range(st.session_state.year_slider[0], st.session_state.year_slider[1] + 1)]
    filtered_df = filtered_df[['Objekt_id'] + year_columns + ['AI Kommentar']+ ['Systid_fra'] + ['agreeness'] + ['kommentar']]



    filtered_df['Systid_fra'] = pd.to_datetime(filtered_df['Systid_fra'])




    #st.write("ok")

    #komment chosen 


    if selected_comment != "Alle":
        filtered_df = filtered_df[filtered_df['AI Kommentar'].str.contains(selected_comment, case=False, na=False)]

    if id_selected:
        filtered_df = filtered_df[filtered_df['Objekt_id'].str.contains(id_selected, case=False, na=False)]

    if selected_date_range:
        #convert string to datetime
        
        filtered_df = filtered_df[filtered_df['Systid_fra'].between(pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1]))]
        # Filter the DataFrame
        #filtered_df = filtered_df[filtered_df['Systid_fra'].between(start_date, end_date)]

    filtered_df['Systid_fra'] = filtered_df['Systid_fra'].dt.strftime('%Y-%m-%d')





    # Example dataframe filtering logic based on selected colors for each year
    if selected_colors:
        for year, colors in selected_colors.items():
            if colors:
                # Assuming the dataframe has one column per year with the year as the column name
                filtered_df = filtered_df[filtered_df[year].isin(colors)]

    st.markdown(
        """
        <style>
        .small-input .stNumberInput input {
            width: 100px;  /* Adjust the width as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )



    if "clicked_sorting" not in st.session_state or st.session_state["clicked_sorting"] == "1":
        #SORTING FROM GREEN TO RED
        content_sorting = sorting_buttons_style + '''
        <div class="button-container">
                <a href="#" id="1"><button class="clicked">Sorter fra Grøn til Rød</button>
                <a href="#" id="2"><button>Sorter fra Rød til Grøn</button>
                <a href="#" id="3"><button>Mest Rød</button>
                <a href="#" id="4"><button>Mest Grøn</button>
        </div>
        '''
        clicked_sorting = click_detector(content_sorting, key="clicked_sorting")
        #PLACE TABLE HERE
        #st.write("Tabel sorted green to red")

        


        #st.image("Arrow down-left.png", use_column_width=True)
        
        filtered_df = sort_by_mean_difference(filtered_df, begin_year= int(year_columns[0]), end_year=int(year_columns[-1]), sort_order=True)
        #def plot_df():
        event = display_le_gdf(filtered_df)
        if event.selection and event.selection["rows"]:
            number_selected = int(event.selection["rows"][0])
            objekt_id_to_display = filtered_df.iloc[number_selected]['Objekt_id'] #session state bousillé
            st.write(objekt_id_to_display)
            #return objekt_id_to_display






        



    elif st.session_state["clicked_sorting"] == "2":
        #SORTING FROM RED TO GREEN
        content_sorting = sorting_buttons_style + '''
        <div class="button-container">
                <a href="#" id="1"><button>Sorter fra Grøn til Rød</button>
                <a href="#" id="2"><button class="clicked">Sorter fra Rød til Grøn</button>
                <a href="#" id="3"><button>Mest Rød</button>
                <a href="#" id="4"><button>Mest Grøn</button>
        </div>
        '''
        clicked_sorting = click_detector(content_sorting, key="clicked_sorting")
        #PLACE TABLE HERE
        #st.write("Tabel sorted red to green")

        filtered_df = sort_by_mean_difference(filtered_df, begin_year= int(year_columns[0]), end_year=int(year_columns[-1]), sort_order=False)
        #def plot_df():
        event = display_le_gdf(filtered_df)
        if event.selection and event.selection["rows"]:
            number_selected = int(event.selection["rows"][0])
            objekt_id_to_display = filtered_df.iloc[number_selected]['Objekt_id'] #session state bousillé
            #st.write(objekt_id_to_display)
            #return objekt_id_to_display



    elif st.session_state["clicked_sorting"] == "3":
        #SORTING most red
        content_sorting = sorting_buttons_style + '''
        <div class="button-container">
                <a href="#" id="1"><button>Sorter fra Grøn til Rød</button>
                <a href="#" id="2"><button>Sorter fra Rød til Grøn</button>
                <a href="#" id="3"><button class="clicked">Mest Rød</button>
                <a href="#" id="4"><button>Mest Grøn</button>
        </div>
        '''
        clicked_sorting = click_detector(content_sorting, key="clicked_sorting")
        #PLACE TABLE HERE
        #st.write("Tabel sorted most red")
        filtered_df = sort_by_mean(filtered_df, begin_year= int(year_columns[0]), end_year=int(year_columns[-1]), sort_order=True)
        
        event = display_le_gdf(filtered_df)
        if event.selection and event.selection["rows"]:
            number_selected = int(event.selection["rows"][0])
            objekt_id_to_display = filtered_df.iloc[number_selected]['Objekt_id'] #session state bousillé
            #return objekt_id_to_display

    

    elif st.session_state["clicked_sorting"] == "4":
        #SORTING most green
        content_sorting = sorting_buttons_style + '''
        <div class="button-container">
                <a href="#" id="1"><button>Sorter fra Grøn til Rød</button>
                <a href="#" id="2"><button>Sorter fra Rød til Grøn</button>
                <a href="#" id="3"><button>Mest Rød</button>
                <a href="#" id="4"><button class="clicked">Mest Grøn</button>
        </div>
        '''
        clicked_sorting = click_detector(content_sorting, key="clicked_sorting")
        #PLACE TABLE HERE
        #st.write("Tabel sorted most green")
        

        filtered_df = sort_by_mean(filtered_df, begin_year= int(year_columns[0]), end_year=int(year_columns[-1]), sort_order=False)
        event = display_le_gdf(filtered_df)
        if event.selection and event.selection["rows"]:
            number_selected = int(event.selection["rows"][0])
            objekt_id_to_display = filtered_df.iloc[number_selected]['Objekt_id'] #session state bousillé
            #st.write(objekt_id_to_display)
            #return objekt_id_to_display





        
        
    


    else:
        #SORTING FROM GREEN TO RED
        content_sorting = sorting_buttons_style + '''
        <div class="button-container">
                <a href="#" id="1"><button class="clicked">Sorter fra Grøn til Rød</button>
                <a href="#" id="2"><button>Sorter fra Rød til Grøn</button>
                <a href="#" id="3"><button>Mest Rød</button>
                <a href="#" id="4"><button>Mest Grøn</button>
        </div>
        '''
        clicked_sorting = click_detector(content_sorting, key="clicked_sorting")
        #PLACE TABLE HERE
        st.write("Tabel sorted green to red")

        filtered_df = sort_by_mean_difference(filtered_df, begin_year= int(year_columns[0]), end_year=int(year_columns[-1]), sort_order=True)
        event = display_le_gdf(filtered_df)

        if event.selection and event.selection["rows"]:
            number_selected = int(event.selection["rows"][0])
            objekt_id_to_display = filtered_df.iloc[number_selected]['Objekt_id'] #session state bousillé
            #st.write(objekt_id_to_display)
            #return objekt_id_to_display



    st.markdown("<div align='center'><h1>Vis Ortofoto for et område:</h1></div>", unsafe_allow_html=True)


    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    if "last_objekt_id" not in st.session_state:
        st.session_state["last_objekt_id"] = None

    if "last_year" not in st.session_state:
        st.session_state["last_year"] = year_columns[-1]

    if "zoom" not in st.session_state:
        st.session_state["zoom"] = 18

    if "zoom_level" not in st.session_state:
        st.session_state["zoom_level"] = 18

    #zoom_level = st.session_state["zoom_level"]

    if "st_data" not in st.session_state:
        st.session_state.st_data = None

    st_data = st.session_state.st_data

    if "show_label" not in st.session_state:
        st.session_state.show_label = False

    if "show_prediction" not in st.session_state:
        st.session_state.show_prediction = False

    # Selected year
    #selected_year = st.radio("", year_columns, horizontal=True) #Vælg År



    def update_selected_year(selected_year):
        st.session_state.selected_year = selected_year
    import streamlit as st

    # Initialize session state for the selected year
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = st.session_state.year_slider[0]

    def update_selected_year(selected_year):
        st.session_state.selected_year = selected_year

    try:
        # Center the container by creating surrounding empty columns
        col1, mid_col, _ = st.columns([1,3,3])
        
        with mid_col:  # Center the container
            selected_years = range(st.session_state.year_slider[0], st.session_state.year_slider[1] + 1)
            cols = st.columns(len(selected_years))
            
            for i, year in enumerate(selected_years):
                with cols[i]:
                    # Check if the value exists before accessing it
                    color_value_list = display_gdf.loc[display_gdf['Objekt_id'] == objekt_id_to_display, str(year)].values
                    if len(color_value_list) > 0:
                        color_value = color_value_list[0]
                        
                        # Display the color box
                        st.markdown(
                            f"<div style='height: 10px; width: 19px; background-color: {color_dict[color_value]}; "
                            f"border-radius: 5px; border-color: black; border-width: 1em;'></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div style='height: 10px; width: 19px; background-color: #052418; "
                            f"border-radius: 5px; border-color: black; border-width: 1em;'></div>",
                            unsafe_allow_html=True
                        )
                    
                    # Create a checkbox for selecting the year
                    is_selected = st.checkbox(
                        f"{year}",
                        value=(st.session_state.selected_year == year),
                        key=f"checkbox_{year}"
                    )
                    
                    # If a checkbox is selected, update the session state
                    if is_selected:
                        update_selected_year(year)
                    # Uncheck all other checkboxes
                    elif st.session_state.selected_year == year:
                        st.session_state.selected_year = None

    except KeyError:
        st.warning("Please select an Objekt_id from the DataFrame.")
    except IndexError:
        st.warning("No data available for the selected Objekt_id.")



    selected_year = st.session_state.selected_year
    # Check if objekt_id is in the filtered DataFrame and is not None
    if (objekt_id_to_display in filtered_df['Objekt_id'].astype(str).values) and (objekt_id_to_display is not None):
        try:
            # Create two columns for the map and comment section
            col1, col2 = st.columns([3, 1])

            with col1:
                # Get color value and mask for the selected year
                color_value = display_gdf.loc[display_gdf['Objekt_id'] == objekt_id_to_display, str(selected_year)].values[0]
                #int to str
                
                gdf_mask = load_mask(selected_year)
                komment = display_gdf.loc[display_gdf['Objekt_id'] == objekt_id_to_display, 'AI Kommentar'].values[0]
                #komment_consultant = display_gdf.loc[display_gdf['Objekt_id'] == objekt_id_to_display, 'kommentar'].values[0]
                #if komment_consultant is np.nan:
                #    komment_consultant = "no comment added"
                # add a vizualize to see the color value
                #st.write(f"Color value: {type(color_value)}")

                
                # add a rectangle to see the color value
                col3, col4 = st.columns([1, 7])
                #with col3:
                #   st.markdown(f"<div style='background-color: {color_map.get(color_value, 'white')}; width: 115%; height: 10px; border-radius: 2px; border: 1px solid black;'></div>", unsafe_allow_html=True)
                #with col4:
                st.markdown(
                    f"""
                <div style='padding: 10px; border: 1px solid #ddd;font-family: Arial; color: black; background-color: white'>
                        <h4>Kommentar:</h4>
                        <h5>{komment}</h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # If the selected year or objekt_id changes, update zoom level and session state
                if st.session_state.get("last_objekt_id") != objekt_id_to_display or st.session_state.get("last_year") != selected_year:
                    #
                    st.session_state["zoom"] = st.session_state["zoom_level"]
                    st.session_state["last_objekt_id"] = objekt_id_to_display
                    st.session_state["last_year"] = selected_year
                    #st.write(st.session_state["zoom"])


                
                #st.write(st.session_state["zoom"])

                # Create the map
                map = create_map(gdf_mask, label_gdf, objekt_id_to_display, selected_year, color_value="yellow", show_label=True, show_prediction=True,default_zoom=st.session_state["zoom"])#st.session_state["zoom"])

                # Display the map

                st_data = st_folium(map, width='100%', height=720) #  zoom=st.session_state["zoom"] reset hole app ?!
                
                #st.write(st_data["geo_json_layer"])#["show"])
                # Update show_label and show_prediction in session state if they have changed


                

                if st_data and "zoom" in st_data:
                    #st.write(f"Zoom level from map: {st_data['zoom']}")
                    # Update zoom_level in session state if zoom has changed
                    if st_data["zoom"] != st.session_state["zoom_level"]:
                        st.session_state["zoom_level"] = st_data["zoom"]
    
                

                #st.write(st_data["zoom"])
                #st.write(st.session_state["zoom_level"])

            with col2:
                #st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
                st.markdown("<div align='center'><h5>Venligst giv din feedback</h5></div>",  unsafe_allow_html=True)

                st.markdown("<div style='text-align: center;'><h5>Er du enig i forudsigelsen?</h5></div>", unsafe_allow_html=True)
                # Render the radio button below the heading
                agreeness = st.radio("", ("Enig", "Uenig"), horizontal=True)


                st.markdown("<div style=margin-top: 20px;'><h5>Konsulent Kommentar</h5></div>", unsafe_allow_html=True)
                consultant_comment = st.text_area("", "", height=100)

                if 'gdf_copy' not in st.session_state:
                    st.session_state.gdf_copy = predicted_gdf.copy()
                    st.session_state.gdf_copy['agreeness'] = None
                    st.session_state.gdf_copy['kommentar'] = None

                if st.button("OK"):
                    st.session_state.gdf_copy.loc[st.session_state.gdf_copy['Objekt_id'] == objekt_id_to_display, 'agreeness'] = (agreeness == "Enig")
                    st.session_state.gdf_copy.loc[st.session_state.gdf_copy['Objekt_id'] == objekt_id_to_display, 'kommentar'] = consultant_comment

                    save_feedback_to_firestore(objekt_id_to_display, agreeness == "Enig", consultant_comment)
                    st.success("Din feedback er blevet tilføjet succesfuldt. Opdater siden for at få den opdaterede dataramme.")
                    st.markdown("<div align='center'><h1>Feedback Data</h1></div>", unsafe_allow_html=True)
                    st.write(st.session_state.gdf_copy[st.session_state.gdf_copy['agreeness'].notnull()][['Objekt_id', 'agreeness', 'kommentar']])
                    st.markdown("<div align='center'><h3>Legende</h3></div>", unsafe_allow_html=True)
                    
                st.markdown("<h3 style='text-align: left;'>Legend</h3>", unsafe_allow_html=True)
                
                # Container for the legend items
                legend_container = """
                <div style='display: flex; flex-direction: column; align-items: flex-start;'>
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <div style='height: 15px; width: 15px; background-color: yellow; border-radius: 3px; border: 1px solid black;'></div>
                        <div style='margin-left: 5px; line-height: 1; font-weight: bold;'>AI generet afgrænsning</div>
                    </div>
                    <div style='display: flex; align-items: center;'>
                        <div style='height: 15px; width: 15px; background-color: pink; border-radius: 3px; border: 1px solid black;'></div>
                        <div style='margin-left: 5px; line-height: 1; font-weight: bold;'>Nuværende afgrænsning</div>
                    </div>
                </div>
                """
                
                st.markdown(legend_container, unsafe_allow_html=True)

                
                

        

                


                

        except FileNotFoundError as e:
            st.error(str(e))
        except KeyError:
            st.warning("Please select a year.")
    else:
        st.warning("Vælg venligst et Objekt_id fra DataFrame.")






