import sys
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QTimer, QRect, QPropertyAnimation, QEasingCurve

# -----------------------------
# Load Models
# -----------------------------
tfid = pickle.load(open("best_models/tfidf.pkl", "rb"))
svc  = pickle.load(open("best_models/svc_best.pkl", "rb"))
nb   = pickle.load(open("best_models/nb_model.pkl", "rb"))


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------
# Loading Spinner Widget
# -----------------------------
class LoadingSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(45, 45)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self.setVisible(False)
        
    def _rotate(self):
        self._angle = (self._angle + 10) % 360
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        
        gradient = QConicalGradient(22.5, 22.5, self._angle)
        gradient.setColorAt(0, QColor("#6366f1"))
        gradient.setColorAt(0.5, QColor("#8b5cf6"))
        gradient.setColorAt(1, QColor("#6366f1"))
        
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(5, 5, 35, 35)

# -----------------------------
# Main Window
# -----------------------------
class SpamClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_model = "SVC"
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Spam Message Classifier")
        self.setGeometry(300, 150, 750, 600)
        self.setStyleSheet("background-color: #f8fafc;")
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(25, 25, 25, 25)
        
        # Header section
        header_frame = QFrame()
        header_frame.setFixedHeight(100)
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed);
                border-radius: 12px;
            }
        """)
        
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 15, 20, 15)
        header_layout.setSpacing(5)
        
        title = QLabel("Spam Message Classifier")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("AI-powered message classification system")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: rgba(255, 255, 255, 0.9);")
        header_layout.addWidget(subtitle)
        
        main_layout.addWidget(header_frame)
        
        # Control panel
        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
        """)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(15)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.setSpacing(10)
        
        model_label = QLabel("Select Model:")
        model_label.setFixedWidth(100)
        model_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #334155;")
        model_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItem("SVC Classifier")
        self.model_combo.addItem("Naive Bayes")
        self.model_combo.setFixedHeight(40)
        self.model_combo.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                border: 2px solid #cbd5e1;
                border-radius: 8px;
                background-color: white;
                font-size: 14px;
                color: #334155;
            }
            QComboBox:hover {
                border-color: #94a3b8;
            }
            QComboBox:focus {
                border-color: #6366f1;
            }
        """)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        
        control_layout.addLayout(model_layout)
        
        # Input section
        input_label = QLabel("Enter Message:")
        input_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #334155;")
        control_layout.addWidget(input_label)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type or paste your message here...")
        self.text_input.setFixedHeight(120)
        self.text_input.setStyleSheet("""
            QTextEdit {
                font-size: 14px;
                padding: 12px;
                border: 2px solid #cbd5e1;
                border-radius: 8px;
                background-color: white;
                color: #334155;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTextEdit:focus {
                border-color: #6366f1;
            }
        """)
        control_layout.addWidget(self.text_input)
        
        # Character counter
        char_layout = QHBoxLayout()
        char_layout.addStretch()
        
        self.char_counter = QLabel("0 characters")
        self.char_counter.setFixedHeight(24)
        self.char_counter.setStyleSheet("""
            QLabel {
                font-size: 12px; 
                color: #64748b; 
                background-color: #f1f5f9; 
                padding: 4px 12px; 
                border-radius: 12px;
            }
        """)
        self.text_input.textChanged.connect(self.update_char_counter)
        char_layout.addWidget(self.char_counter)
        
        control_layout.addLayout(char_layout)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        # Clear button
        clear_btn = QPushButton("Clear Text")
        clear_btn.setFixedHeight(40)
        clear_btn.setFixedWidth(120)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #64748b;
                font-size: 14px;
                font-weight: 500;
                border-radius: 8px;
                border: 2px solid #e2e8f0;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                color: #475569;
            }
        """)
        clear_btn.clicked.connect(self.clear_input)
        button_layout.addWidget(clear_btn)
        
        button_layout.addStretch()
        
        # Sample button
        sample_btn = QPushButton("Load Example")
        sample_btn.setFixedHeight(40)
        sample_btn.setFixedWidth(140)
        sample_btn.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #6366f1;
                font-size: 14px;
                font-weight: 500;
                border-radius: 8px;
                border: 2px solid #e2e8f0;
            }
            QPushButton:hover {
                background-color: #e0e7ff;
                color: #4f46e5;
            }
        """)
        sample_btn.clicked.connect(self.load_example_message)
        button_layout.addWidget(sample_btn)
        
        # Classify button
        self.classify_btn = QPushButton("Classify Message")
        self.classify_btn.setFixedHeight(40)
        self.classify_btn.setFixedWidth(150)
        self.classify_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed);
                color: white;
                font-size: 14px;
                font-weight: 600;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4338ca, stop:1 #6d28d9);
            }
            QPushButton:disabled {
                background: #cbd5e1;
                color: #94a3b8;
            }
        """)
        self.classify_btn.clicked.connect(self.classify_message)
        button_layout.addWidget(self.classify_btn)
        
        control_layout.addLayout(button_layout)
        main_layout.addWidget(control_frame)
        
        # Results section
        results_frame = QFrame()
        results_frame.setFixedHeight(200)
        results_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
        """)
        
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(20, 20, 20, 20)
        results_layout.setSpacing(15)
        
        # Results title
        results_title = QLabel("Classification Results")
        results_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #334155; padding-bottom: 8px; border-bottom: 2px solid #f1f5f9;")
        results_layout.addWidget(results_title)
        
        # Results content
        content_layout = QVBoxLayout()
        content_layout.setSpacing(12)
        
        # Loading spinner container
        spinner_container = QFrame()
        spinner_container.setFixedHeight(60)
        spinner_container.setStyleSheet("background-color: transparent;")
        
        spinner_layout = QVBoxLayout(spinner_container)
        spinner_layout.setAlignment(Qt.AlignCenter)
        
        self.loading_spinner = LoadingSpinner()
        spinner_layout.addWidget(self.loading_spinner)
        
        content_layout.addWidget(spinner_container)
        
        # Prediction result
        self.prediction_label = QLabel("Results will appear here")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setFixedHeight(30)
        self.prediction_label.setStyleSheet("font-size: 16px; color: #94a3b8;")
        content_layout.addWidget(self.prediction_label)
        
        # Confidence bar
        confidence_container = QFrame()
        confidence_container.setFixedHeight(24)
        confidence_container.setStyleSheet("background-color: #f1f5f9; border-radius: 12px;")
        
        self.confidence_bar = QFrame(confidence_container)
        self.confidence_bar.setGeometry(0, 0, 0, 24)
        self.confidence_bar.setStyleSheet("background-color: #10b981; border-radius: 12px;")
        
        content_layout.addWidget(confidence_container)
        
        # Confidence info
        info_layout = QHBoxLayout()
        
        self.confidence_label = QLabel("Confidence: 0%")
        self.confidence_label.setStyleSheet("font-size: 13px; color: #64748b; font-weight: 500;")
        info_layout.addWidget(self.confidence_label)
        
        info_layout.addStretch()
        
        self.model_info_label = QLabel("")
        self.model_info_label.setStyleSheet("font-size: 12px; color: #94a3b8;")
        info_layout.addWidget(self.model_info_label)
        
        content_layout.addLayout(info_layout)
        results_layout.addLayout(content_layout)
        
        main_layout.addWidget(results_frame)
        
        # Status bar
        self.status_label = QLabel("Ready to classify messages")
        self.status_label.setFixedHeight(30)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px; 
                color: #64748b; 
                background-color: white; 
                padding: 6px; 
                border-radius: 6px; 
                border: 1px solid #e2e8f0;
            }
        """)
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
        
    def update_char_counter(self):
        text = self.text_input.toPlainText()
        self.char_counter.setText(f"{len(text)} characters")
        
    def clear_input(self):
        self.text_input.clear()
        self.status_label.setText("Text cleared")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px; 
                color: #6366f1; 
                font-weight: bold; 
                background-color: white; 
                padding: 6px; 
                border-radius: 6px; 
                border: 1px solid #e2e8f0;
            }
        """)
        
    def load_example_message(self):
        example = "Congratulations! You've won $1,000,000! Click here to claim your prize: bit.ly/winprize123\nLimited time offer!"
        self.text_input.setPlainText(example)
        self.status_label.setText("Example message loaded")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px; 
                color: #8b5cf6; 
                font-weight: bold; 
                background-color: white; 
                padding: 6px; 
                border-radius: 6px; 
                border: 1px solid #e2e8f0;
            }
        """)
        
    def on_model_changed(self, text):
        if "Naive Bayes" in text:
            self.current_model = "Naive Bayes"
            self.status_label.setText("Using Naive Bayes model")
        else:
            self.current_model = "SVC"
            self.status_label.setText("Using SVC model")
            
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px; 
                color: #6366f1; 
                font-weight: bold; 
                background-color: white; 
                padding: 6px; 
                border-radius: 6px; 
                border: 1px solid #e2e8f0;
            }
        """)
        
    def classify_message(self):
        message = self.text_input.toPlainText().strip()
        
        if not message:
            self.status_label.setText("Please enter a message to classify")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 12px; 
                    color: #ef4444; 
                    font-weight: bold; 
                    background-color: white; 
                    padding: 6px; 
                    border-radius: 6px; 
                    border: 1px solid #e2e8f0;
                }
            """)
            return
            
        if len(message) < 5:
            self.status_label.setText("Message is too short (minimum 5 characters)")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 12px; 
                    color: #ef4444; 
                    font-weight: bold; 
                    background-color: white; 
                    padding: 6px; 
                    border-radius: 6px; 
                    border: 1px solid #e2e8f0;
                }
            """)
            return
            
        # Show loading state
        self.classify_btn.setEnabled(False)
        self.classify_btn.setText("Processing...")
        self.loading_spinner.show()
        self.loading_spinner._timer.start(50)
        self.prediction_label.setText("Analyzing message...")
        self.status_label.setText("Processing message...")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px; 
                color: #f59e0b; 
                font-weight: bold; 
                background-color: white; 
                padding: 6px; 
                border-radius: 6px; 
                border: 1px solid #e2e8f0;
            }
        """)
        
        # Process after short delay
        QTimer.singleShot(500, lambda: self.process_classification(message))
        
    def process_classification(self, message):
        try:
            transformed = transform_text(message)
            vector = tfid.transform([transformed]).toarray()
            
            if self.current_model == "SVC":
                model = svc
                model_name = "SVC Classifier"
            else:
                model = nb
                model_name = "Naive Bayes"
                
            prediction = model.predict(vector)[0]
            probabilities = model.predict_proba(vector)[0]
            
            if prediction == 1:
                result_text = "SPAM DETECTED"
                result_color = "#ef4444"
                confidence = probabilities[1] * 100
                bar_color = "#ef4444"
            else:
                result_text = "NORMAL MESSAGE"
                result_color = "#10b981"
                confidence = probabilities[0] * 100
                bar_color = "#10b981"
                
            self.prediction_label.setText(f"<b>{result_text}</b>")
            self.prediction_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {result_color};")
            
            self.model_info_label.setText(f"{model_name} • {len(transformed.split())} words processed")
            
            bar_width = int(confidence * 2.5)
            animation = QPropertyAnimation(self.confidence_bar, b"geometry")
            animation.setDuration(700)
            animation.setEasingCurve(QEasingCurve.OutQuart)
            animation.setStartValue(QRect(0, 0, 0, 24))
            animation.setEndValue(QRect(0, 0, bar_width, 24))
            animation.start()
            
            self.confidence_label.setText(f"Confidence: {confidence:.1f}%")
            
            if confidence > 90:
                conf_color = "#10b981"
            elif confidence > 70:
                conf_color = "#f59e0b"
            else:
                conf_color = "#ef4444"
                
            self.confidence_label.setStyleSheet(f"font-size: 14px; color: {conf_color}; font-weight: bold;")
            self.confidence_bar.setStyleSheet(f"background-color: {bar_color}; border-radius: 12px;")
            
            self.status_label.setText(f"Classification complete • {result_text}")
            self.status_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 12px; 
                    color: {result_color}; 
                    font-weight: bold; 
                    background-color: white; 
                    padding: 6px; 
                    border-radius: 6px; 
                    border: 1px solid #e2e8f0;
                }}
            """)
            
        except Exception as e:
            self.prediction_label.setText("Classification Error")
            self.prediction_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ef4444;")
            
            self.confidence_label.setText("Error occurred")
            self.confidence_label.setStyleSheet("font-size: 14px; color: #ef4444; font-weight: bold;")
            
            self.status_label.setText(f"Error: {str(e)[:50]}...")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 12px; 
                    color: #ef4444; 
                    font-weight: bold; 
                    background-color: white; 
                    padding: 6px; 
                    border-radius: 6px; 
                    border: 1px solid #e2e8f0;
                }
            """)
            
        finally:
            self.classify_btn.setEnabled(True)
            self.classify_btn.setText("Classify Message")
            self.loading_spinner.hide()
            self.loading_spinner._timer.stop()

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = SpamClassifierApp()
    window.show()
    
    sys.exit(app.exec_())