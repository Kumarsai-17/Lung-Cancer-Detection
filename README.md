# LungCare AI - Lung Cancer Detection System

A comprehensive web application for lung cancer risk assessment using advanced machine learning models including XGBoost and Convolutional Neural Networks (CNN).

## Features

- **Dual Analysis Methods**:
  - Feature-based analysis using XGBoost model (23 medical/lifestyle factors)
  - Image-based analysis using CNN model for medical imaging

- **Modern Web Interface**:
  - Responsive design with dark blue and white theme
  - Interactive checkboxes with smooth animations
  - Rich CSS with transitions and modern UI components
  - Mobile-friendly responsive layout

- **Advanced Functionality**:
  - Real-time form validation
  - Image upload with drag-and-drop support
  - Detailed result analysis with probability scores
  - Print and share functionality
  - Auto-save form data

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd lung-cancer-detection
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are in place**:
   Make sure the following files exist in the `models/` directory:
   - `xgb_model.pkl` - XGBoost model for feature analysis
   - `cnn_model.h5` - CNN model for image analysis
   - `class_indices.pkl` - Class mapping for CNN predictions

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Access the application**:
   Open your browser and navigate to `http://localhost:5000`

3. **Navigate through the application**:
   - **Home Page**: Overview and introduction
   - **Analysis Page**: Choose between feature-based or image-based analysis
   - **Feature Analysis**: Fill out the medical questionnaire
   - **Image Upload**: Upload medical images for CNN analysis
   - **Results**: View detailed analysis results
   - **About**: Learn more about the technology and disclaimers

## Project Structure

```
lung-cancer-detection/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── models/               # ML model files
│   ├── xgb_model.pkl
│   ├── cnn_model.h5
│   └── class_indices.pkl
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── features.html
│   ├── upload.html
│   ├── result.html
│   └── about.html
└── static/              # Static assets
    ├── css/
    │   └── style.css    # Main stylesheet
    ├── js/
    │   └── script.js    # JavaScript functionality
    └── uploads/         # Uploaded images (created automatically)
```

## Features Analysis

The feature-based analysis uses 23 different factors:

### Demographics
- Age
- Gender

### Environmental Factors
- Air Pollution Exposure
- Dust Allergy
- Occupational Hazards

### Lifestyle Factors
- Alcohol Use
- Smoking
- Passive Smoker
- Balanced Diet
- Obesity

### Medical History
- Genetic Risk
- Chronic Lung Disease

### Current Symptoms
- Chest Pain
- Coughing Blood
- Fatigue
- Weight Loss
- Shortness of Breath
- Wheezing
- Swallowing Difficulty
- Clubbing of Finger Nails
- Frequent Cold
- Dry Cough
- Snoring

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: 
  - XGBoost for feature-based analysis
  - TensorFlow/Keras CNN for image analysis
- **Frontend**: 
  - HTML5, CSS3, JavaScript
  - Font Awesome icons
  - Google Fonts (Inter)
- **Image Processing**: OpenCV, Pillow
- **Data Processing**: NumPy, Pandas

## Model Information

### XGBoost Model
- **Purpose**: Feature-based lung cancer risk assessment
- **Input**: 23 numerical features (0 or 1 for most boolean features)
- **Output**: Binary classification with probability scores
- **Accuracy**: ~95% (based on training data)

### CNN Model
- **Purpose**: Image-based lung cancer detection
- **Input**: Medical images (224x224 pixels, RGB)
- **Architecture**: Convolutional Neural Network
- **Output**: Classification with confidence scores

## Security & Privacy

- Images are processed locally and not stored permanently
- Form data is temporarily saved in browser localStorage for user convenience
- No personal data is transmitted to external servers
- Secure file upload with size and type validation

## Disclaimer

**Important Medical Disclaimer**: This application is for educational and screening purposes only. It should not replace professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical advice and proper diagnosis.

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes. Please ensure you have proper licensing for any medical data or models used in production.

## Support

For technical support or questions about the application, please refer to the About page in the application or contact the development team.