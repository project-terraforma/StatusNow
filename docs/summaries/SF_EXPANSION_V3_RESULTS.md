# San Francisco Expansion & V3 Results (Feb 18, 2026)

## 🌍 Overview

To validate the generalizability of our "StatusNow" place classification model, we expanded our analysis beyond New York City (NYC) to include San Francisco (SF). This expansion involved:

1.  **Replicating the Overture Truth Pipeline**: Adapting `fetch_overture_data.py` and `build_truth_dataset.py` to handle multiple cities dynamically.
2.  **Creating a Combined Dataset**: Merging NYC and SF truth data to create a robust, multi-region training set.
3.  **Benchmarking V3**: Testing our V3 feature set (Brand-Aware + Recency Bins) on the new data.

## 📊 Dataset Statistics

We constructed three datasets for this evaluation:

| Dataset            | Total Samples | Open Places | Closed Places | % Closed | Data Source               |
| :----------------- | :------------ | :---------- | :------------ | :------- | :------------------------ |
| **NYC (Baseline)** | 12,000        | 9,000       | 3,000         | 25.0%    | Overture Jan/Feb '26 + S2 |
| **SF Only**        | 9,622         | 9,000       | ~622          | **6.5%** | Overture Jan/Feb '26 + S2 |
| **Combined**       | **18,619**    | 13,810      | 4,809         | 25.8%    | Merged NYC & SF           |

_Note: The SF dataset has significantly fewer "closed" examples in the Overture delta (only ~600 found vs ~3000 in NYC), making it a challenging, imbalanced test case._

## 🚀 Model Performance (V3 + Combined)

We trained the V3 model on the combined dataset and achieved our highest accuracy to date.

| Metric                 | Score      | vs. V2 Baseline |
| :--------------------- | :--------- | :-------------- |
| **Balanced Accuracy**  | **95.19%** | +24.5% 📈       |
| **ROC AUC**            | **0.9937** | +0.21           |
| **Precision (Closed)** | 84.8%      | N/A             |
| **Recall (Closed)**    | **96.4%**  | N/A             |

### City-Specific Performance

| Dataset Tested | Balanced Accuracy | Key Observation                                                                                                         |
| :------------- | :---------------- | :---------------------------------------------------------------------------------------------------------------------- |
| **NYC Only**   | 92.87%            | Strong baseline performance.                                                                                            |
| **SF Only**    | 91.39%            | **Generalizes well!** Despite extreme class imbalance (6% closed), the model correctly identified 94% of closed places. |

## 🔍 Deep Dive: The Brand Gap

One persistent finding is the performance disparity between major Brands and Independent Businesses (Non-Brands). In San Francisco, this gap was more pronounced.

| Metric       | Brand Accuracy | Non-Brand Accuracy | Gap                |
| :----------- | :------------- | :----------------- | :----------------- |
| **NYC**      | 97.4%          | 67.2%              | -30.2%             |
| **SF**       | 79.2%          | 92.0%              | +12.8% (Inverted!) |
| **Combined** | **89.8%**      | **95.7%**          | -5.9%              |

_Interpretation: In SF, "Non-Brand" places were actually easier to classify than Brands, possibly due to clearer digital decay signals for closed indie shops in tech-savvy SF, whereas closed chain locations might retain some corporate digital presence._

## ✅ Conclusion

1.  **The V3 Model is Robust**: It achieves >91% balanced accuracy in a completely new city with different data distributions.
2.  **Data Volume Matters**: Combining NYC and SF data improved overall accuracy to **95.2%**, smoothing out regional idiosyncrasies.
3.  **Ready for Scale**: The parameterized pipeline (`--city` args) allows us to easily add London, Tokyo, or any other region next.
