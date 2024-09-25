from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from io import BytesIO
import matplotlib.pyplot as plt
import os
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store the trained model and expected columns globally
model = None
expected_columns = None


@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/train", response_class=HTMLResponse)
async def train_model(
        request: Request,
        train_file: UploadFile = File(...),
        target_column: str = Form(...)
):
    global model, expected_columns

    try:
        # Load the training data
        train_data = pd.read_csv(BytesIO(await train_file.read()))

        # Check if target_column exists
        if target_column not in train_data.columns:
            error_message = f"Target column '{target_column}' not found in the dataset."
            return templates.TemplateResponse("index.html", {"request": request, "error": error_message})

        # Separate features and target
        X = train_data.drop(columns=[target_column])
        y = train_data[target_column]

        # Store expected columns for validation during prediction
        expected_columns = X.columns.tolist()

        # Select categorical and numerical columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

        # Drop categorical columns with most unique labels
        threshold = 0.8 * len(X)  # threshold for dropping unique columns
        to_drop = [col for col in categorical_columns if X[col].nunique() > threshold]
        X = X.drop(columns=to_drop)
        categorical_columns = [col for col in categorical_columns if col not in to_drop]

        # Define preprocessing steps for categorical and numerical data
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
            ('num', StandardScaler(), numerical_columns)
        ])

        # Build a pipeline with preprocessing and RandomForestClassifier
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Split the data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model_pipeline.fit(X_train, y_train)

        # Store the trained model
        model = model_pipeline

        # Evaluate on validation set
        val_score = model_pipeline.score(X_val, y_val)

        success_message = f"Model trained successfully! Validation Accuracy: {val_score:.2f}"

        # Redirect to prediction page with success message
        response = RedirectResponse(url="/predict", status_code=303)
        response.headers["message"] = success_message
        return response

    except Exception as e:
        error_message = f"An error occurred during training: {str(e)}"
        return templates.TemplateResponse("index.html", {"request": request, "error": error_message})


@app.get("/predict", response_class=HTMLResponse)
async def get_predict_page(request: Request):
    message = request.headers.get("message")
    return templates.TemplateResponse("predict.html", {"request": request, "message": message})


@app.post("/predict", response_class=HTMLResponse)
async def make_prediction(
        request: Request,
        test_file: UploadFile = File(...)
):
    global model, expected_columns

    if model is None:
        error_message = "No trained model found. Please train the model first."
        return templates.TemplateResponse("predict.html", {"request": request, "error": error_message})

    try:
        # Load the test data
        test_data = pd.read_csv(BytesIO(await test_file.read()))

        # Validate columns
        test_columns = test_data.columns.tolist()
        missing_columns = set(expected_columns) - set(test_columns)
        extra_columns = set(test_columns) - set(expected_columns)

        if missing_columns:
            error_message = f"Test data is missing the following required columns: {', '.join(missing_columns)}"
            return templates.TemplateResponse("predict.html", {"request": request, "error": error_message})

        if extra_columns:
            error_message = f"Test data has unexpected extra columns: {', '.join(extra_columns)}"
            return templates.TemplateResponse("predict.html", {"request": request, "error": error_message})

        # Reorder columns to match training data
        test_data = test_data[expected_columns]

        # Predict using the trained model
        predictions = model.predict(test_data)

        # Prepare predictions with index
        prediction_results = list(zip(test_data.index, predictions))

        # Plot pie chart for predicted labels
        label_counts = pd.Series(predictions).value_counts()
        plt.figure(figsize=(6, 6))
        label_counts.plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        pie_chart_path = os.path.join("static", "pie_chart.png")
        plt.savefig(pie_chart_path)
        plt.close()

        return templates.TemplateResponse("result.html", {
            "request": request,
            "predictions": prediction_results,
            "pie_chart": pie_chart_path
        })

    except Exception as e:
        error_message = f"An error occurred during prediction: {str(e)}"
        return templates.TemplateResponse("predict.html", {"request": request, "error": error_message})


# To run the FastAPI server, use: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
