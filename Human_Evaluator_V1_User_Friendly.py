import streamlit as st
import pandas as pd
import json
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Rating function
def rate_summary(input_text, summary, reference_summary):
    input_embedding = model.encode([input_text])
    summary_embedding = model.encode([summary])
    reference_embedding = model.encode([reference_summary])
    
    sim_to_input = cosine_similarity(input_embedding, summary_embedding)[0][0]
    sim_to_reference = cosine_similarity(reference_embedding, summary_embedding)[0][0]
    
    length_ratio = len(summary.split()) / len(reference_summary.split())
    length_penalty = max(0, min(1, 2 - abs(length_ratio - 1)))
    
    score = (sim_to_input * 0.4 + sim_to_reference * 0.5 + length_penalty * 0.1) * 10
    return round(score, 1)

# Streamlit app
st.title("Human Evaluator for Relation Hallucination in Abstractive Summarization")
st.write("Upload your JSON data to evaluate summaries manually and automatically.")
 # Optional: Add some styling and instructions
st.markdown("""
    ### Data Format for Upload
    The uploaded JSON file should have the following format:

    ```json
    {
        "0": {
            "Id": "10157432",
            "dataset": "xlsum",
            "InputText": "The full article text goes here...",
            "ReferenceSummary": "The original summary goes here...",
            "facebook/bart-large-cnn": "Model-generated summary here...",
            "google/pegasus-xsum": "Model-generated summary here...",
            "t5-large": "Model-generated summary here...",
            "gpt-3.5-turbo": "Model-generated summary here...",
            "RefSum": "Reference summary for comparison..."
        }
    }
    ```
    This structure allows the program to automatically evaluate model-generated summaries against the reference summary.
    """)

# File upload
uploaded_file = st.file_uploader("Upload JSON file", type="json")

if uploaded_file:
    # Load JSON data
    data = json.load(uploaded_file)
    results = []

    # Process each article
    for idx, article in data.items():
        input_text = article["InputText"]
        reference_summary = article["ReferenceSummary"]
        
        # Display the article details
        st.subheader(f"Article {idx}")
        st.write("**Input Text:**")
        st.text(input_text)
        st.write("**Reference Summary:**")
        st.text(reference_summary)

        # Allow user to rate the models manually
        ratings = {}
        for model_name in ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo"]:
            summary = article[model_name]
            st.write(f"**{model_name}** Summary:")
            st.text(summary)
            
            # Provide a unique key for each model to avoid DuplicateWidgetID error
            rating = st.slider(f"Rate {model_name} Summary (1-10)", 1, 10, 5, key=f"{model_name}_{idx}")
            ratings[model_name] = rating

        # Store results
        results.append({
            "Index": idx,
            "Input Text": input_text,
            "Reference Summary": reference_summary,
            **ratings
        })

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Display table for article-wise evaluation
    st.subheader("Evaluation Results by Article")
    st.dataframe(df)

    # Aggregate ratings for Input Text, Reference Summary, and all models
    aggregate_data = {
        "Input Text": df["Input Text"].apply(lambda x: rate_summary(x, x, x)),
        "Reference Summary": df["Reference Summary"].apply(lambda x: rate_summary(x, x, x)),
        **{
            model_name: df[model_name].mean() for model_name in ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo"]
        }
    }

    aggregate_df = pd.DataFrame(aggregate_data, index=["Average Rating"]).transpose()

    # Display aggregate ratings
    st.subheader("Aggregate Ratings")
    st.dataframe(aggregate_df)

    # Plot results for each model
    st.subheader("Model-Wise Ratings")
    for model_name in ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo"]:
        fig = px.bar(df, x="Index", y=model_name, title=f"Ratings for {model_name}", labels={"Index": "Article Index", model_name: "Rating"})
        st.plotly_chart(fig)

    # Plot frequency of ratings for each model
    st.subheader("Frequency of Ratings (1-10)")
    for model_name in ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-large", "gpt-3.5-turbo"]:
        rating_counts = df[model_name].value_counts().sort_index()
        fig = px.bar(x=rating_counts.index, y=rating_counts.values, title=f"Frequency of Ratings for {model_name}", labels={"x": "Rating", "y": "Frequency"})
        st.plotly_chart(fig)

    # Download results
    st.subheader("Download Annotated Data")
    annotated_data = df.to_json(orient="records", indent=4)
    st.download_button("Download JSON", data=annotated_data, file_name="annotated_data.json", mime="application/json")

   