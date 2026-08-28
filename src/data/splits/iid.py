"""
IID (Independent and Identically Distributed) Split Module.
Stratified 70% Train / 15% Val / 15% Test.
"""
from sklearn.model_selection import train_test_split

def split_iid(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42, stratify_col="genre"):
    """
    Generates a standard stratified IID split (70/15/15).
    """
    test_size = val_ratio + test_ratio
    tr_df, temp_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[stratify_col]
    )
    val_rel_size = val_ratio / test_size
    va_df, te_df = train_test_split(
        temp_df, test_size=(1.0 - val_rel_size), random_state=random_state, stratify=temp_df[stratify_col]
    )
    return tr_df, va_df, te_df
