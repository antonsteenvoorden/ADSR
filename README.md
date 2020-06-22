# Attribute-aware Diversification for Sequential Recommendations
This is the repository of the master thesis by Anton Steenvoorden performed at Ahold Delhaize.  
This repository contains the code for the Attribute-aware Diversifying Sequential Recommender (ADSR).  

## Abstract
Research has shown users prefer diverse recommendations over homogeneous ones. However, most recent work on Sequential Recommenders (SRs) does not consider diversity and only strives for maximum accuracy, resulting in accurate but redundant recommendations. In this work, we present a novel model called the Attribute-aware Diversifying Sequential Recommender (ADSR). The ADSR takes both accuracy and diversity into consideration while generating the list of recommendations. Specifically, the ADSR utilizes available attribute information when modeling a user’s sequential behavior. The ADSR simultaneously learns the user’s most likely item to interact with, and their preference for attributes, which is used to diversify the recommendations. The ADSR consists of three modules: the Attribute-aware Encoder (AE), the Attribute Predictor (AP) and the Attribute-aware Diversifying Decoder (ADD). First of all, the AE is responsible for encoding the input sequences and learning the preference over items. Second, the AP is responsible for learning the preference for the attributes. Third, the ADD incrementally generates a diversified list of recommendations based on the predicted attribute preference distribution, while taking the item-attributes already present in the recommended list into account.
   
Experiments on the publicly available datasets MovieLens-1M and TMall demonstrate that the ADSR can significantly increase diversity of recommendations while maintaining accuracy. More- over, an ablation study shows the positive effect each module has on performance by investigating the performance of stripped-down variants of the ADSR. Results from the comparison with other baselines show that the ADSR can provide highly diverse recommendations while outperforming a number of models. Furthermore, several studied cases show that the ADSR provides properly diverse recommendations, but that instances exist where the ADSR is limited in its diversifying capabilities. Finally, a follow-up experiment is performed in which the user identifier is used to learn an embed- ding used to make predictions. Results show that this helps the ADSR yield higher performance, indicating that further improvements over the model are possible in the future.


## Code 
The code is written using PyTorch and PyTorch-Ignite.  
The project starts in `code` in `main.py`.  
The `hyperparameters.py` are set to work on the MovieLens dataset.  

Start by obtaining and preprocessing the data (see readme in `data/movielens` and `data/tmall`).  

To use the ADSR:  
1. Train MTASR with the `--model_type=MTASR` flag
2. Copy stored weights in `trained_models/MTASR` to `trained_models/ADSR` and rename MTASR to ADSR in the filenames.
3. Run the ADSR with the `--model_type=ADSR` flag


### Overview
In the `main.py` you won't find a training loop. This has been implement with ignite.  
In each model-folder you will find the model file and a manager. Each manager has a brief description of the model (
same as in the paper).  
The manager inherits some functions from the `Model` class, and implements the update function as well as which metrics to evaluate.
Some model use the data_loader at the root of `models`, other re-implement it in their own folder.

All models report the `combined_loss` metric, which just the `relevance_loss` for BSR and ANAM.  
In `metrics.py` the predictions are kept track of, and during the evaluation step they are computed and saved to disk.  
