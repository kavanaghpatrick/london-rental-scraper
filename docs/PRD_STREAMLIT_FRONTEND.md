# PRD: London Rent Price Predictor - Streamlit Frontend

## Overview
Build a simple Streamlit web app that allows users to input property details and receive a predicted monthly rent price, along with comparable properties from the database.

## Goals
1. Enable users to get rent predictions without running Python scripts
2. Show comparable properties to validate predictions
3. Buildable in 1-2 days by solo developer

## Tech Stack
- **Frontend**: Streamlit (pure Python)
- **Model**: XGBoost (existing `rental_price_models_v15.py`)
- **Database**: SQLite (`output/rentals.db`)

## Features

### F1: User Input Form
- Address/postcode text input (extract postcode district via regex)
- Bedrooms (number input, 0-10)
- Bathrooms (number input, 0-5)
- Size in sqft (number input, 100-10000)
- Property type (dropdown: Flat, House, Apartment, Maisonette, Studio)
- Furnished status (dropdown: Furnished, Unfurnished, Part-furnished)

### F2: Rent Prediction Display
- Predicted monthly rent (£X,XXX pcm)
- Price per sqft (£X.XX/sqft)
- Confidence indicator (model accuracy context)

### F3: Comparable Properties Table
- Query similar listings from SQLite
- Match criteria: same postcode district, ±1 bedroom, active listings
- Display: address, price_pcm, bedrooms, sqft, source, URL
- Sortable/filterable table

### F4: Basic Validation
- Postcode format validation
- Required field checks
- Error messages for invalid inputs

## File Structure
```
frontend/
├── app.py              # Streamlit main entry point
├── predictor.py        # Model loading and prediction logic
├── comparables.py      # SQLite query helpers
└── requirements.txt    # streamlit, pandas, xgboost
```

## Non-Goals (Simplicity)
- No user authentication
- No data persistence (predictions not saved)
- No complex styling/theming
- No API layer (Streamlit handles everything)
- No deployment automation (local dev first)

## Acceptance Criteria
1. User can enter property details and click "Predict"
2. Predicted rent displays within 2 seconds
3. At least 5 comparable properties shown (if available)
4. App runs with `streamlit run frontend/app.py`
5. Total code < 300 lines across all files

## Implementation Issues
- Issue #1: Project setup & model integration
- Issue #2: User input form
- Issue #3: Prediction display
- Issue #4: Comparable properties
