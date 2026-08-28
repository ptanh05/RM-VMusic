"""
Missing Modality Benchmark Set Builder.
Extracts clean evaluation sets for missing modality evaluation.
"""
def create_missing_modality_benchmark(iid_test_df):
    """
    Creates the standardized evaluation set for missing modality experiments.
    """
    return iid_test_df.copy()
