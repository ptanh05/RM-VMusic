"""
Label Shift Split Module.
Introduces controlled class prior shift between training and evaluation distributions.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def split_label_shift(df, dominant_genre="POP_BALLAD", genre_col="genre", random_state=42):
    """
    Constructs a controlled label shift partition by varying the prior of the dominant genre.
    """
    dom_df = df[df[genre_col] == dominant_genre]
    non_dom_df = df[df[genre_col] != dominant_genre]

    # Shift dominant genre distribution
    dom_tr, dom_temp = train_test_split(dom_df, test_size=0.22, random_state=random_state)
    dom_va, dom_te = train_test_split(dom_temp, test_size=0.55, random_state=random_state)

    # Shift non-dominant genres
    non_dom_tr, non_dom_temp = train_test_split(
        non_dom_df, test_size=0.38, random_state=random_state, stratify=non_dom_df[genre_col]
    )
    non_dom_va, non_dom_te = train_test_split(
        non_dom_temp, test_size=0.70, random_state=random_state, stratify=non_dom_temp[genre_col]
    )

    tr_df = pd.concat([dom_tr, non_dom_tr], ignore_index=True).sample(frac=1.0, random_state=random_state)
    va_df = pd.concat([dom_va, non_dom_va], ignore_index=True).sample(frac=1.0, random_state=random_state)
    te_df = pd.concat([dom_te, non_dom_te], ignore_index=True).sample(frac=1.0, random_state=random_state)

    return tr_df, va_df, te_df
