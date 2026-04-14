"""
Predicting Movie Success - Streamlit app

Shawn Lokshin, Fedor Bentsa, Borys Kocherev
DS 4420 - Spring 2026

This app loads the manual MLP trained in the notebook and lets you play with
the inputs to see how the prediction changes.
"""
import streamlit as st
import numpy as np


# load the trained model and preprocessing info once (cached across reruns)
@st.cache_resource
def load_model():
    a = np.load("model_artifacts.npz", allow_pickle=True)
    return {
        "W1": a["W1"], "b1": a["b1"],
        "W2": a["W2"], "b2": a["b2"],
        "W3": a["W3"], "b3": a["b3"],
        "mean": a["feature_mean"],
        "std": a["feature_std"],
        "feature_cols": list(a["feature_cols"]),
        "all_genres": list(a["all_genres"]),
        "lang_vals": list(a["lang_vals"]),
    }


model = load_model()


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def forward(x, m):
    # same forward pass as the notebook, just without training
    z1 = x @ m["W1"] + m["b1"]
    a1 = relu(z1)
    z2 = a1 @ m["W2"] + m["b2"]
    a2 = relu(z2)
    z3 = a2 @ m["W3"] + m["b3"]
    return float(sigmoid(z3).ravel()[0])


def build_features(budget, runtime, popularity, vote_avg, vote_count,
                   release_year, release_month, genres, language, m):
    # build the feature vector in the same order the notebook used
    feature_cols = m["feature_cols"]
    all_genres = m["all_genres"]
    lang_vals = m["lang_vals"]
    lookup = {name: i for i, name in enumerate(feature_cols)}

    x = np.zeros(len(feature_cols))

    # numeric features with the same log transforms from training
    x[lookup["log_budget"]] = np.log1p(budget)
    x[lookup["runtime"]] = runtime
    x[lookup["log_popularity"]] = np.log1p(popularity)
    x[lookup["vote_average"]] = vote_avg
    x[lookup["log_vote_count"]] = np.log1p(vote_count)
    x[lookup["release_year"]] = release_year
    x[lookup["release_month"]] = release_month

    # genre dummies
    for g in all_genres:
        key = f"genre_{g}"
        if key in lookup:
            x[lookup[key]] = 1.0 if g in genres else 0.0

    # language dummies - the first value in lang_vals is the dropped baseline
    for lang in lang_vals[1:]:
        key = f"lang_{lang}"
        if key in lookup:
            x[lookup[key]] = 1.0 if language == lang else 0.0

    # standardize with the training mean and std
    return (x - m["mean"]) / m["std"]


# -------- page config --------
st.set_page_config(page_title="Predicting Movie Success", page_icon=":clapper:", layout="wide")

tab_about, tab_predict = st.tabs(["About", "Predict"])


# -------- About tab --------
with tab_about:
    st.title("Predicting Movie Success")
    st.markdown(
        "**Shawn Lokshin, Fedor Bentsa, Borys Kocherev**  \n"
        "DS 4420 - Machine Learning and Data Mining 2 - Spring 2026"
    )
    st.markdown("---")

    st.header("The problem")
    st.markdown(
        "The film industry is financially risky. Studios spend huge amounts of money "
        "on making and marketing a film, but many movies still lose money at the box "
        "office. We ask whether simple machine learning models trained on basic movie "
        "metadata can reliably predict whether a film will earn back more than its "
        "production budget."
    )

    st.header("The two models")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Manual MLP (Python)")
        st.markdown(
            "- Written from scratch in NumPy (no sklearn, no Keras)\n"
            "- Two hidden layers: 64 and 32 units, ReLU activations\n"
            "- Sigmoid output, binary cross-entropy loss\n"
            "- Mini-batch SGD (lr=0.01, batch=32), early stopping on validation loss\n"
            "- **Test accuracy: 81.0%, AUC: 0.862**\n"
            "- This is the model powering the Predict tab"
        )
    with c2:
        st.subheader("Bayesian Logistic Regression (R)")
        st.markdown(
            "- Fit in R with `rstanarm::stan_glm`\n"
            "- Weakly informative normal priors\n"
            "- 4 chains, 2000 iterations each\n"
            "- Gives full posterior distributions over coefficients\n"
            "- **Test accuracy: 75.4%**\n"
            "- Used for feature interpretation rather than prediction"
        )

    st.header("Key findings")
    st.markdown(
        "- The MLP beats the Bayesian model on every headline metric, which is what "
        "we would expect since a non-linear model can capture interactions that a "
        "linear model cannot.\n"
        "- **Popularity** is by far the strongest positive predictor of success.\n"
        "- **Budget** actually has a negative effect on success (defined as revenue > "
        "budget), because bigger budgets are harder to recoup.\n"
        "- **Drama** and **Thriller** films tend to underperform more targeted genres "
        "like Action and Science Fiction, after controlling for budget and popularity. "
        "This matches an observation from Simonoff and Sparrow (2000)."
    )

    st.header("Data")
    st.markdown(
        "We used the [TMDb 5000 Movies dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) "
        "from Kaggle, filtered down to about 3,762 films with valid budget, revenue, "
        "runtime and popularity data. The same filters were used for both models so "
        "the comparison is fair."
    )

    st.markdown("---")
    st.markdown(
        "Code on [GitHub](https://github.com/shawnlokshin/Predicting-Movie-Success)"
    )


# -------- Predict tab --------
with tab_predict:
    st.title("Try the model yourself")
    st.markdown(
        "Move the sliders on the left to describe a hypothetical movie, and the "
        "manual MLP will predict the probability that it earns more than its "
        "production budget. Everything updates live."
    )
    st.markdown("---")

    col_inputs, col_result = st.columns([5, 4])

    with col_inputs:
        st.subheader("Movie features")
        budget = st.number_input(
            "Budget (USD)", min_value=100_000, max_value=500_000_000,
            value=50_000_000, step=1_000_000,
        )
        runtime = st.slider("Runtime (minutes)", 60, 240, 120)
        popularity = st.slider("Popularity score", 0.0, 300.0, 20.0, step=0.5,
                               help="TMDb popularity score. A typical blockbuster is around 100-200.")
        release_year = st.slider("Release year", 1990, 2026, 2022)
        release_month = st.slider("Release month", 1, 12, 6)

        genre_options = model["all_genres"]
        genres = st.multiselect(
            "Genres", options=genre_options,
            default=["Action", "Adventure"],
        )

        lang_display = {
            "en": "English", "fr": "French", "es": "Spanish", "de": "German",
            "ja": "Japanese", "ko": "Korean", "zh": "Chinese (zh)",
            "cn": "Chinese (cn)", "hi": "Hindi", "ru": "Russian", "other": "Other",
        }
        lang_options = model["lang_vals"]
        default_idx = lang_options.index("en") if "en" in lang_options else 0
        language = st.selectbox(
            "Original language",
            options=lang_options,
            index=default_idx,
            format_func=lambda x: lang_display.get(x, x),
        )

        with st.expander("Post-release features (optional)"):
            st.caption(
                "These only make sense if the movie has already been released. Leave "
                "them at the defaults for a pure pre-release prediction."
            )
            vote_avg = st.slider("Vote average (1-10)", 1.0, 10.0, 6.5, step=0.1)
            vote_count = st.slider("Vote count", 0, 20_000, 1000, step=50)

    with col_result:
        st.subheader("Prediction")

        x = build_features(
            budget, runtime, popularity, vote_avg, vote_count,
            release_year, release_month, genres, language, model,
        )
        p = forward(x[np.newaxis, :], model)

        st.metric(label="Probability of success", value=f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))

        if p >= 0.5:
            st.success(f"Predicted: **SUCCESS** (revenue will exceed budget)")
        else:
            st.warning(f"Predicted: **FAILURE** (revenue will not exceed budget)")

        st.markdown("### What this means")
        st.markdown(
            f"The manual MLP estimates a **{p:.1%}** probability that a movie with "
            "these features would earn more than its production budget at the box "
            "office."
        )
        st.markdown(
            "Try moving the sliders around. A few things you might notice:\n"
            "- **Popularity** has a big effect. Pushing it up drives the probability "
            "toward 1.\n"
            "- **Budget** has the opposite effect: doubling it usually hurts the "
            "probability, because bigger budgets are harder to recoup.\n"
            "- **Runtime** has a small positive effect.\n"
            "- **Genre** can swing the probability a few points either way."
        )
