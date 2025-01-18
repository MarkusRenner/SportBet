# SportBet: Predicting NBA Outcomes
*AISA* study project

## Overview

**SportBet** is an *AISA* study project aimed at predicting NBA game outcomes through data analysis and machine learning. This project involves gathering NBA data, preprocessing it, and applying machine learning techniques to design predictive models. The primary goal is to develop a system that provides insights into betting strategies for NBA games.

## TASK

### Emphasize the potential real-world impact of their work:

- **Sports**: Revolutionize coaching, fan engagement, or athlete training with
ML-powered insights.
- **Gaming**: Enhance player experiences by personalizing game content or
improving multiplayer matchmaking.
- **Music**: Democratize music creation with AI, from personalized
recommendations to automated composition.

### The ML pipeline

- **Learning Opportunity**: The hackathon provides a structured opportunity to
explore end-to-end ML pipelines.
- **Hands-on experience with**:
    - **Data Collection**: How to gather meaningful datasets.
    - **Preprocessing**: Cleaning and transforming data.
    - **Model Training**: Building ML models.
    - **Evaluation & Deployment**: Validating and showcasing solutions in practical, real-world settings.

### Requirements

#### 1- Data Collection

- **Available sources**: Mention where the data came from (e.g., APIs, web
scraping, sensors, public datasets).
- **Collection**: Collect your own data.
- **Challenges**: Briefly discuss any difficulties (e.g., missing data, limited
availability).
- **Volume**: Indicate the size and type of data (e.g., 10GB of CSVs, 100K
images).
- **Tools**: Mention any tools used for data collection (e.g., Python scripts, Google
Sheets, Mobile cameras).

#### 2- Data Preprocessing

- **Steps Taken**: Outline how you prepared the data (e.g., cleaning,
normalization, feature engineering).
- **Techniques/Tools**: Mention techniques (e.g., handling missing values, one-hot
encoding) and tools (e.g., Pandas, NumPy).
- **Challenges Solved**: Highlight any issues you overcame, such as imbalanced
datasets or noisy data.

#### 3- Model Training

- **Model Choice**: State the model(s) used (e.g., Random Forest, CNN,
Transformers) and why they were selected.
- **Frameworks**: List frameworks/libraries (e.g., TensorFlow, PyTorch,
Scikit-learn).
- **Training Specs**: Mention key details (e.g., epochs, hyperparameters, compute
resources).
- **Innovations**: Point out unique optimizations, customizations, or novel
approaches.

#### 4- ML Tasks

- Solve **3** different ML tasks for your theme, ex:
    - **Regression**: to predict player performance metrics like speed, stamina, or scoring potential.
    - **Classification**: to identify the genre of a song or moderating in-game chat.
    - **Clustering**: to identify team formations
    - **Object** detection and action recognition: to identify key actions like goals, passes, or fouls in
match footage.
    - **Gaze estimation**: to analyze where players are looking during critical moments
    - **Sentiment analysis**: on social media or fan forums

#### 5- Model Evaluation

- **Metrics**: Show metrics relevant to your problem (e.g., accuracy, precision,
recall, RMSE).
- **Comparison**: If applicable, compare models or baselines.
- **Visuals**: Include plots or charts (e.g., confusion matrix, ROC curve).
- **Insights**: Highlight key takeaways about the model’s performance and
limitations.

#### Bonus

- **Deployment**:
    - **Process**: Briefly explain how the model is deployed (e.g., Flask API, cloud platform).
    - **Usage**: Highlight how users interact with it (e.g., web app, mobile app).
    - **Scalability**: Mention steps taken to ensure the system can handle real-world usage

- **Demo**:
    - Brings the ML pipeline to life!
    - Create a live or interactive demo to demonstrate the practical viability of the pipeline beyond just theory or code.

### Presentation

- Day 4
- Showcase your project with slides
- 20 minutes
- *AISA* poster session


## Project Workflow

### 1. Data Collection via Web Scraping

- **Objective**: Gather up-to-date NBA game data from reliable sources.
- **Tools Used**: Web scraping techniques are employed using libraries such as Requests and Pandas to extract data from [basketball-reference.com](https://www.basketball-reference.com).
- **Outcome**: A comprehensive dataset that includes game statistics, team records, and <span style="color: red">player performances</span>.

### 2. Data Preprocessing

- **Objective**: Prepare the raw data for analysis.
- **Tasks**:
  - Handling missing values.
  - Converting categorical data to numerical formats.
  - <span style="color: red">Normalizing and scaling features for consistency.</span>
- **Outcome**: A clean, structured dataset ready for model training.

### 3. Machine Learning Models

- **Objective**: Develop and compare machine learning models for predicting game outcomes.
- **Methods**:
  - **<span style="color: red">Logistic Regression</span>**: Used for its simplicity and performance in binary classification tasks.
  - **<span style="color: red">Random Forest</span>**: Deployed for handling large datasets and achieving higher accuracy.
  - **<span style="color: red">Support Vector Machine (SVM)</span>**: Implemented for robust decision boundary finding.
- **Outcome**: Evaluation of model performance based on accuracy, precision, recall, and <span style="color: red">F1-score</span>.

## Installation and Setup

Follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/MarkusRenner/SportBet.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   or:
    ```bash
   conda env create -f environment.yml
   ```

3. Open Jupyter Notebook or JupyterLab from the terminal:
     ```bash
     jupyter notebook
     ```
     or
     ```bash
     jupyter lab
     ```

4. Navigate to data scraping script and run:
   ```bash
   web_scraping.ipynb
   ```

4. Execute preprocessing and training scripts:
    <br>
    <span style="color: red">preprocessing.ipynb</span>  
    <span style="color: red">regression_training.ipynb</span>  

## Data 
- Find the scraped dataset of NBA stats in the `data` directory.

## Results and Insights

- Detailed model comparison reports are available in the `results` directory.
- The final report discusses model performance and potential improvements.

## Future Work

- **Expand Data Sources**: Integrate more diverse data sources for a broader dataset.
- **Advanced Modeling**: Explore deep learning techniques for enhanced prediction accuracy.
- **Real-Time Predictions**: Implement a system for live predictions with ongoing data updates.

## License

This project is licensed under the MIT License.

## How to safe conda environment

1. Requirements.txt
    ```bash
        conda list -e > requirements.txt
    ```
   
2. Environment.yaml
    ```bash
        conda env export > environment.yaml
    ```

