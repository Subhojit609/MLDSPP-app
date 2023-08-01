import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64 

# Function to calculate stability based on the provided stability map
def calculate_stability(sequence, stability_map):
    size = len(sequence)
    list_stability = []
    
    for pos in range(size-1):
        nucleotide_pair = sequence.upper()[pos:pos+2]
        stability = stability_map.get(nucleotide_pair, None)
        
        if stability is not None:
            list_stability.append(stability)
    
    return list_stability

# Function to create the DataFrame
def create_dataframe(sequence_content, stability_map):
    # Calculate the stability values for the sequence
    list_stability = calculate_stability(sequence_content, stability_map)

    # Calculate the number of rows and columns needed
    num_rows = (len(list_stability) - 1) // 99 + 1
    num_cols = min(len(list_stability), 99)

    # Create a 2D numpy array with NaN values to represent the DataFrame
    data_array = np.full((num_rows, num_cols), np.nan)

    # Fill the data_array with the stability values
    for i, val in enumerate(list_stability):
        row = i // 99
        col = i % 99
        data_array[row, col] = val

    # Create the DataFrame with 99 columns
    df_sequence = pd.DataFrame(data_array, columns=range(1, num_cols + 1))

    return df_sequence

# Function to load the model and its corresponding stability_map
def load_model(model_name):
    model_filename = model_options[model_name]["model_file"]
    stability_map = model_options[model_name]["stability_map"]
    
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    
    return model, stability_map

# Model options dictionary
model_options = {
    "SLD_model": {
        "model_file": "SLD_model.pickle",
        "stability_map": {
            'AA': -1, 'AT': -0.88, 'TA': -0.58, 'AG': -1.3, 'GA': -1.3, 'TT': -1,
            'AC': -1.45, 'CA': -1.45, 'TG': -1.44, 'GT': -1.44, 'TC': -1.28, 'CT': -1.28,
            'CC': -1.84, 'CG': -2.24, 'GC': -2.27, 'GG': -1.84,
        }
    },
    "MGD_model": {
        "model_file": "MGD_model.pickle",
        "stability_map": {
            'AA': -1.95, 'AT': -2.17, 'TA': -1.52, 'AG': -1.46, 'GA': -2.18, 'TT': -1.95,
            'AC': -2.81, 'CA': -1.44, 'TG': -1.44, 'GT': -2.81, 'TC': -2.18, 'CT': -1.46,
            'CC': -1.76, 'CG': -1.42, 'GC': -2.99, 'GG': -1.76,
        }
    },
    "JSD_model": {
        "model_file": "JSD_model.pickle",
        "stability_map": {
            'AA': -12.0, 'AT': -10.6, 'TA': -11.2, 'AG': -11.5, 'GA': -11.4, 'TT': -12.0,
            'AC': -11.8, 'CA': -12.3, 'TG': -12.3, 'GT': -11.8, 'TC': -11.4, 'CT': -11.5,
            'CC': -9.5, 'CG': -13.1, 'GC': -13.2, 'GG': -9.5,
        }
    },
    "DUD_model": {
        "model_file": "DUD_model.pickle",
        "stability_map": {
            'AA': -5.37, 'AT': -6.57, 'TA': -3.82, 'AG': -6.78, 'GA': -9.81, 'TT': -5.37,
            'AC': -10.51, 'CA': -6.57, 'TG': -6.57, 'GT': -10.51, 'TC': -9.81, 'CT': -6.78,
            'CC': -8.26, 'CG': -9.69, 'GC': -14.59, 'GG': -8.26,
        }
    },
    "APD_model": {
        "model_file": "APD_model.pickle",
        "stability_map": {
            'AA': -17.5, 'AT': -16.7, 'TA': -17.0, 'AG': -15.8, 'GA': -14.7, 'TT': -17.5,
            'AC': -18.1, 'CA': -19.5, 'TG': -19.5, 'GT': -18.1, 'TC': -14.7, 'CT': -15.8,
            'CC': -14.9, 'CG': -19.2, 'GC': -14.7, 'GG': -14.9,
        }
    },
    "APR_model": {
        "model_file": "APR_model.pickle",
        "stability_map": {
            'AA': -13.7, 'AT': -15.4, 'TA': -16.0, 'AG': -14.0, 'GA': -14.2, 'TT': -13.7,
            'AC': -13.8, 'CA': -14.4, 'TG': -14.4, 'GT': -13.8, 'TC': -14.2, 'CT': -14.0,
            'CC': -11.1, 'CG': -15.6, 'GC': -16.9, 'GG': -11.1,
        }
    },
    "CAD_model": {
        "model_file": "CAD_model.pickle",
        "stability_map": {
            'AA': 0.703, 'AT': 0.854, 'TA': 0.615, 'AG': 0.780, 'GA': 1.230, 'TT': 0.703,
            'AC': 1.323, 'CA': 0.790, 'TG': 0.790, 'GT': 1.323, 'TC': 1.230, 'CT': 0.780,
            'CC': 0.984, 'CG': 1.124, 'GC': 1.792, 'GG': 0.984,
        }
    },
}

# Example sequence
example_sequence = (
    "CATTTCGCCAAGCGTTCATGGCTTACAGAGCGTTTGAATGCGCGCTTGCAACTCAACAACCTCTTAAGCTATGATTTCATTCAAGAGAAAGCGAGCAAGA\n"
    "TAGGCATCCTTCTGGTAGTTTGTGGTGCGATTGTACTCAGTGGAGTGATTTCTGGATTGAGTGCACTCATTGTTTGTGGATTGGGTATTAGTACGATTTC\n"
    "AACAACAAGCTTTTCTAGTAAAACATCATCGGCTACAGATGGCACCAATTATGTTTTTAAAGATTCTGTAGTTATAGAAAATGTACCCAAAACAGGGGAA\n"
    "AAACGCGCAAAAAATGCAAAAATTCTAAATTTTCTCCAAATGACAAAAAAAAAAAAAACGATTTTATGCTACAATGCTTTTAATACATTCTTACTTAATG\n"
    "GACTTAATAATCCTTATAGTTATATTATTAGCTTTGTTTTTATGGCTTGACTTATCCCTAAAAATGCGCTATAGTTATGTCGCTTAATAACAATAAGCGC\n"
)

# Streamlit app
def main():
    st.title("XgBoost Promoter Prediction")

    st.sidebar.header("Choose Input Method")
    input_method = st.sidebar.radio(
        "Select how to provide sequences:",
        ["Upload a .txt File", "Paste Sequences"],
    )

    selected_model = st.sidebar.selectbox("Select a Model", list(model_options.keys()))

    if input_method == "Upload a .txt File":
        uploaded_file = st.file_uploader("Upload your sequence file(100 seq in each line[-80th to +19th position])", type=["txt"])

        if uploaded_file is not None:
            sequence_content = uploaded_file.read().decode("utf-8")
            model, stability_map = load_model(selected_model)  # Load the selected model and its stability_map
            df_stability = create_dataframe(sequence_content, stability_map)
            predictions = predict(df_stability, model)
            df_predictions = display_predictions(df_stability, predictions, sequence_content)  # Save the returned DataFrame
            download_button(df_predictions)  # Add download button

    elif input_method == "Paste Sequences":
        st.subheader("Use Example Sequence")
        if st.button("Use Example Sequence"):
            st.session_state.sequence_text_area = example_sequence

        sequences = st.text_area("Paste your sequences here(100 seq in each line[-80th to +19th position])", value=st.session_state.get("sequence_text_area", ""))

        if st.button("Predict"):
            model, stability_map = load_model(selected_model)  # Load the selected model and its stability_map
            df_stability = create_dataframe(sequences, stability_map)
            predictions = predict(df_stability, model)
            df_predictions = display_predictions(df_stability, predictions, sequences)  # Save the returned DataFrame
            download_button(df_predictions)  # Add download button


def predict(df_stability, model):
    predictions = model.predict(df_stability)
    return predictions

def display_predictions(df_stability, predictions, sequences):
    # Create a new column 'Prediction' in the df_stability DataFrame
    df_stability['Prediction'] = predictions

    # Create a new DataFrame 'df_predictions' with 'Sequence' column
    df_predictions = pd.DataFrame()
    df_predictions['Sequence'] = sequences.splitlines()
    df_predictions['Prediction'] = df_stability['Prediction'].map({1: "Promoter", 0: "Non-Promoter"})

    # Display the predictions to the user
    st.write("Predicted Results:")
    st.write(df_predictions)

    # Return the df_predictions DataFrame so that it can be used in the main function
    return df_predictions

def download_button(df_predictions):
    # Create a download link for the DataFrame as a CSV file
    csv = df_predictions.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert DataFrame to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
