import logging
import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Tuple

class DataAnalyzer:
    """Class for handling data analysis and feature engineering tasks."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize DataAnalyzer with output directory setup and logging configuration.
        
        Parameters:
        -----------
        output_dir : str
            Directory to store all outputs (logs, analysis results, models)
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self):
        """Create necessary directories for outputs."""
        dirs = ['logs', 'analysis', 'models', 'submissions']
        for dir_name in dirs:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Configure logging to both file and console."""
        log_file = self.output_dir / 'logs' / f'analysis_.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_dataframe(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Enhanced analyze_dataframe with logging and file output."""
        self.logger.info(f"Starting analysis for dataset: {name}")
        
        analysis = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'duplicate_rows': df.duplicated().sum()
            }
        }
                
        # Save analysis results
        analysis_file = self.output_dir / 'analysis' / f'{name}_analysis_.json'
        pd.Series(analysis).to_json(analysis_file)
        
        self.logger.info(f"Analysis completed for {name}. Results saved to {analysis_file}")
        return analysis

class ModelTrainer:
    """Class for handling model training and optimization."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
    def optimize_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_test: pd.DataFrame, y_test: pd.Series, region: str) -> Tuple[Any, Any]:
        """Enhanced optimize_model with logging and model saving."""
        self.logger.info(f"Starting model optimization for region: {region}")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Train Random Forest
        rf_model = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', verbose=1)
        rf_grid.fit(X_train, y_train)
        rf_preds = rf_grid.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_preds)
        
        # Train LightGBM
        lgb_model = LGBMClassifier(verbose=-1)
        lgb_grid = GridSearchCV(lgb_model, param_grid, cv=5, scoring='accuracy', verbose=1)
        lgb_grid.fit(X_train, y_train)
        lgb_preds = lgb_grid.predict(X_test)
        lgb_accuracy = accuracy_score(y_test, lgb_preds)
        
        # Log results
        self.logger.info(f"{region} - Random Forest Accuracy: {rf_accuracy:.4f}")
        self.logger.info(f"{region} - LightGBM Accuracy: {lgb_accuracy:.4f}")
        
        # Save models
        model_dir = self.output_dir / 'models' / region
        model_dir.mkdir(parents=True, exist_ok=True)
        
        pd.to_pickle(rf_grid.best_estimator_, model_dir / f'rf_model.pkl')
        pd.to_pickle(lgb_grid.best_estimator_, model_dir / f'lgb_model.pkl')
        
        return rf_grid.best_estimator_, lgb_grid.best_estimator_


def create_new_features(df):
    """
    Generate remote sensing spectral indices and features from satellite imagery bands.
    
    This function calculates various vegetation indices (VI), water indices, built-up indices,
    and other spectral features commonly used in remote sensing analysis. It preserves the 
    original data by working on a copy of the input DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing satellite imagery bands. Required columns:
        - blue_p50: Blue band (50th percentile)
        - green_p50: Green band (50th percentile)
        - red_p50: Red band (50th percentile)
        - nir_p50: Near-infrared band (50th percentile)
        - swir1_p50: Shortwave infrared 1 band (50th percentile)
        - swir2_p50: Shortwave infrared 2 band (50th percentile)
        - re1_p50: Red edge 1 band (50th percentile)
        - re2_p50: Red edge 2 band (50th percentile)
        - re3_p50: Red edge 3 band (50th percentile)
        - VV_p50: VV polarization band (50th percentile)
        - VH_p50: VH polarization band (50th percentile)
    
    Returns
    -------
    pandas.DataFrame
        Original DataFrame with additional columns:
        - ndvi: Normalized Difference Vegetation Index [-1, 1]
        - evi: Enhanced Vegetation Index (soil-adjusted)
        - ndwi: Normalized Difference Water Index
        - ndbi: Normalized Difference Built-up Index
        - vv_vh_ratio: Radar backscatter ratio
        - nir_red_ratio: Simple ratio between NIR and Red bands
        - swir_nir_ratio: Ratio between SWIR1 and NIR bands
        - re_ratios: Combined red edge band ratios
        - spectral_var: Variance across visible and NIR bands
        - msavi: Modified Soil Adjusted Vegetation Index
        - rep_ratio1: Red edge position ratio 1
        - rep_ratio2: Red edge position ratio 2
        - reflectance_range: Range of reflectance values across bands
    
    Notes
    -----
    - All input bands should be in consistent units (typically reflectance values)
    - Some indices may produce NaN values if denominators are zero
    - The function assumes positive reflectance values for ratio calculations
    """
    data = df.copy()
    
    # Normalized Difference Vegetation Index (NDVI)
    data['ndvi'] = (data['nir_p50'] - data['red_p50']) / (data['nir_p50'] + data['red_p50'])
    
    # Enhanced Vegetation Index (EVI)
    data['evi'] = 2.5 * ((data['nir_p50'] - data['red_p50']) / 
                         (data['nir_p50'] + 6 * data['red_p50'] - 7.5 * data['blue_p50'] + 1))
    
    # Normalized Difference Water Index (NDWI)
    data['ndwi'] = (data['green_p50'] - data['nir_p50']) / (data['green_p50'] + data['nir_p50'])
    
    # Normalized Difference Built-up Index (NDBI)
    data['ndbi'] = (data['swir1_p50'] - data['nir_p50']) / (data['swir1_p50'] + data['nir_p50'])
    
    # Radar ratio (useful for surface roughness)
    data['vv_vh_ratio'] = data['VV_p50'] / data['VH_p50']
    
    # Band Ratios
    data['nir_red_ratio'] = data['nir_p50'] / data['red_p50']
    data['swir_nir_ratio'] = data['swir1_p50'] / data['nir_p50']
    data['re_ratios'] = (data['re1_p50'] / data['re2_p50']) * (data['re2_p50'] / data['re3_p50'])
    
    # Simple variance between bands
    data['spectral_var'] = data[['blue_p50', 'green_p50', 'red_p50', 'nir_p50']].var(axis=1)
    
    # Modified Soil Adjusted Vegetation Index (MSAVI)
    data['msavi'] = (2 * data['nir_p50'] + 1 - np.sqrt((2 * data['nir_p50'] + 1)**2 - 
                    8 * (data['nir_p50'] - data['red_p50']))) / 2
    
    # Red Edge Position (REP) related features
    data['rep_ratio1'] = data['re1_p50'] / data['re2_p50']
    data['rep_ratio2'] = data['re2_p50'] / data['re3_p50']
    
    # Range of reflectance values
    refl_cols = ['blue_p50', 'green_p50', 'red_p50', 'nir_p50', 'swir1_p50', 'swir2_p50']
    data['reflectance_range'] = data[refl_cols].max(axis=1) - data[refl_cols].min(axis=1)
    
    return data

def main():
    """Main execution function."""
    # Initialize classes
    analyzer = DataAnalyzer()
    trainer = ModelTrainer()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting agricultural plastic cover mapping analysis")
    
    # Load datasets
    data_paths = {
        'kenya_train': "files\Kenya_training.csv",
        'kenya_test': "files\Kenya_testing.csv",
        'spain_train': "files\Spain_training.csv",
        'spain_test': "files\Spain_validation.csv",
        'vnm_train': "files\VNM_training.csv",
        'vnm_test': "files\VNM_testing.csv"
    }
    
    # Load and analyze data
    datasets = {}
    for name, path in data_paths.items():
        logger.info(f"Loading dataset: {name}")
        datasets[name] = pd.read_csv(path)
        analyzer.analyze_dataframe(datasets[name], name)
    
    # Feature engineering
    logger.info("Starting feature engineering")
    for name in datasets:
        datasets[name] = create_new_features(datasets[name])
    
    # Prepare training data
    regions = ['kenya', 'spain', 'vnm']
    models = {}

    for region in regions:
        logger.info(f"Processing region: {region}")
        
        # Split data
        X = datasets[f'{region}_train'].drop(columns=['TARGET'])
        y = datasets[f'{region}_train']['TARGET']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, test_size=0.20
        )
        
        # Train models
        rf_model, lgb_model = trainer.optimize_model(
            X_train, y_train, X_test, y_test, region
        )
        models[region] = (rf_model, lgb_model)
    
    # Generate predictions and create submission
    logger.info("Generating predictions and creating submission file")
    submissions = []
    
    for region in regions:
        test_data = datasets[f'{region}_test'].fillna(0)
        rf_model, lgb_model = models[region]
        
        # Generate predictions
        preds = (rf_model.predict(test_data) + lgb_model.predict(test_data)) / 2
        
        # Prepare submission
        submission = pd.DataFrame({
            'ID': f"{region.capitalize()}_" + test_data['ID'].astype(str),
            'TARGET': preds.astype(int)
        })
        submissions.append(submission)
    
    # Combine and save submission
    final_submission = pd.concat(submissions, ignore_index=True)
    submission_path = analyzer.output_dir / 'submissions' / f'submission_{analyzer.timestamp}.csv'
    final_submission.to_csv(submission_path, index=False)
    
    logger.info(f"Process completed. Submission saved to {submission_path}")

if __name__ == "__main__":
    main()