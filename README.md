# Prediction using pycaret
## 1. Regression
Regression → Linear Regression → Vanilla Linear Regression
### Advantages
- Works well if you have a few well defined variables and need a simple predictive model
- Fast training speed and prediction speeds
- Does well on small datasets
- Interpretable results, easy to explain
- Easy to update the model when new data comes in
- No parameter tuning required (the regularized linear models below need to tune the regularization parameter)
- Doesn't need feature scaling (the regularized linear models below need feature scaling)
- If dataset has redundant features, linear regression can be unstable
### Disadvantages
- Doesn't work well for non-linear data
- Low(er) prediction accuracy
- Can overfit (see regularized models below to counteract this)
- Doesn't separate signal from noise well – cull irrelevant features before use
- Doesn't learn feature interactions in the dataset

Regression → Linear Regression → Lasso, Ridge, Elastic-Net
### Advantages
- These models are linear regression with regularization
- Help counteract overfitting
- These models are much better at generalizing because they are simpler
- They work well when we only care about a few features
### Disadvantages
- Need feature scaling
- Need to tune the regularization parameter

Regression → Regression Trees → Decision Tree
### Advantages
- Fast training speed and prediction speeds
- Captures non-linear relationships in the dataset well
- Learns feature interactions in the dataset
- Great when your dataset has outliers
- Great for finding the most important features in the dataset
- Doesn't need feature scaling
- Decently interpretable results, easy to explain
### Disadvantages
- Low(er) prediction accuracy
- Requires some parameter tuning
- Doesn't do well on small datasets
- Doesn't separate signal from noise well
- Not easy to update the model when new data comes in
- Used very rarely in practice, use ensembled trees instead
- Can overfit

Regression → Regression Trees → Ensembles
### Advantages
- Collates predictions from multiple trees
- High prediction accuracy - does really well in practice
- Preferred algorithm in Kaggle competitions
- Great when your dataset has outliers
- Captures non-linear relationships in the dataset well
- Great for finding the most important features in the dataset
- Separates signal vs noise
- Doesn't need feature scaling
- Perform really well on high-dimensional data
### Disadvantages
- Not easy to interpret or explain
- Not easy to update the model when new data comes in
- Requires some parameter tuning 
- Harder to tune
- Doesn't do well on small datasets

Regression → Deep Learning
### Advantages
- High prediction accuracy - does really well in practice
- Captures very complex underlying patterns in the data
- Does really well with both big datasets and those with high-dimensional data
- Easy to update the model when new data comes in
- The network's hidden layers reduce the need for feature engineering remarkably
- Is state of the art for computer vision, machine translation, sentiment analysis and speech recognition tasks
### Disadvantages
- Very long training speed
- Need a huge amount of computing power
- Need lots of training data because it learns a vast number of parameters
- Outperformed by Boosting algorithms for nonimage, non-text, non-speech tasks
- Very flexible, come with lots of different architecture building blocks, thus require expertise to design the architecture

Regression → K Nearest Neighbors (Distance Based)
### Advantages
- Fast training speed
- Doesn't need much parameter tuning
- Interpretable results, easy to explain
- Works well for small datasets (<100k training set)
### Disadvantages
- Low(er) prediction accuracy
- Doesn't do well on small datasets
- Need to pick a suitable distance function
- Needs feature scaling to work well
- Prediction speed grows with size of dataset
- Doesn't separate signal from noise well 
– Don't work well with highdimensional data

