import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import pickle
import matplotlib
matplotlib.use('Agg')
from flask_cors import CORS
import matplotlib.pyplot as plt
# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
DATA_FILE = 'retrain_data.pkl'  # File lưu dữ liệu sai

# Load mô hình AI
model = tf.keras.models.load_model('handwritten_digits.keras',
                                   custom_objects={'softmax_v2': tf.keras.activations.softmax})

# Load dữ liệu MNIST để kết hợp khi huấn luyện lại
mnGetter = tf.keras.datasets.mnist
(X_mnist_train, y_mnist_train), (_, _) = mnGetter.load_data()
X_mnist_train = tf.keras.utils.normalize(X_mnist_train, axis=1)

# Hàm xử lý và dự đoán ảnh
def predict_digit(img_path):
    try:
        # Đọc hình ảnh dưới dạng thang độ xám
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Sử dụng adaptive thresholding để tách chữ số khỏi nền
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)

        # Làm dày nét chữ
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.dilate(binary_image, kernel, iterations=1)

        # Tìm contour để căn giữa chữ số
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)

            # Cắt chữ số và căn giữa
            digit = binary_image[y:y+h, x:x+w]
            max_dim = max(w, h)
            top = (max_dim - h) // 2
            bottom = max_dim - h - top
            left = (max_dim - w) // 2
            right = max_dim - w - left
            digit = cv2.copyMakeBorder(digit, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

            # Resize về 20x20 pixel (để chừa biên 4 pixel mỗi bên, giống MNIST)
            digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

            # Tạo hình ảnh 28x28 và đặt chữ số vào trung tâm
            final_image = np.zeros((28, 28), dtype=np.uint8)
            final_image[4:24, 4:24] = digit

            # Chuẩn hóa về [0, 1]
            final_image = final_image / 255.0

            # Thêm batch dimension để dự đoán
            final_image = np.array([final_image])

            # Dự đoán
            prediction = model.predict(final_image)
            predicted_digit = np.argmax(prediction)
            confidence = round(float(max(prediction[0]) * 100), 2)
            return f"{predicted_digit} ({confidence}%)", True
        else:
            return "No digit found in the image", False
    except Exception as e:
        return str(e), False

# Hàm lưu dữ liệu sai
def save_retrain_data(img, true_digit):
    X_retrain, y_retrain = [], []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            X_retrain, y_retrain = pickle.load(f)
    
    X_retrain.append(img[0])  # Lưu ảnh
    y_retrain.append(int(true_digit))  # Lưu nhãn đúng
    
    with open(DATA_FILE, 'wb') as f:
        pickle.dump((X_retrain, y_retrain), f)
    
    return len(X_retrain)  # Trả về số lượng dữ liệu đã lưu

# Hàm huấn luyện lại mô hình
def retrain_model():
    if not os.path.exists(DATA_FILE):
        return False, "No retraining data available.", None

    with open(DATA_FILE, 'rb') as f:
        X_retrain, y_retrain = pickle.load(f)

    X_retrain = np.array(X_retrain)
    y_retrain = np.array(y_retrain)

    # Kết hợp với dữ liệu MNIST (lấy ngẫu nhiên 1000 mẫu)
    indices = np.random.choice(X_mnist_train.shape[0], 1000, replace=False)
    X_combined = np.concatenate((X_retrain, X_mnist_train[indices]), axis=0)
    y_combined = np.concatenate((y_retrain, y_mnist_train[indices]), axis=0)

    # Huấn luyện lại mô hình và lưu lịch sử
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_combined, y_combined, epochs=10, verbose=0)
    model.save('handwritten_digits.keras')

    # Xóa dữ liệu tạm sau khi huấn luyện
    os.remove(DATA_FILE)
    return True, "Model retrained successfully!", history

# Hàm vẽ biểu đồ và lưu dưới dạng file
def plot_training_history(history):
    if not os.path.exists('static'):
        os.makedirs('static')

    # Vẽ biểu đồ độ chính xác
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('static/accuracy_plot.png')
    plt.close()

    # Vẽ biểu đồ hàm mất mát
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('static/loss_plot.png')
    plt.close()

    return 'accuracy_plot.png', 'loss_plot.png'

# Route để phục vụ file từ uploads/
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route trang chủ
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        files = request.files.getlist('file')
        results = []

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                predicted_result, success = predict_digit(file_path)
                if success:
                    results.append({
                        'filename': filename,
                        'prediction': predicted_result,
                        'image_path': url_for('uploaded_file', filename=filename)
                    })
                else:
                    results.append({
                        'filename': filename,
                        'prediction': f"Error: {predicted_result}",
                        'image_path': url_for('uploaded_file', filename=filename)
                    })

        return render_template('result.html', results=results)
    
    return render_template('index.html')

# Route xử lý báo sai và huấn luyện lại
@app.route('/retrain', methods=['POST'])
def retrain():
    filename = request.form.get('filename')
    true_digit = request.form.get('true_digit')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return "File not found", 404

    # Xử lý ảnh để lưu vào dữ liệu huấn luyện lại
    img = cv2.imread(file_path)[:, :, 0]
    img = cv2.resize(img, (28, 28))
    img = np.invert(np.array([img]))
    img = tf.keras.utils.normalize(img, axis=1)
    
    # Lưu dữ liệu sai
    num_samples = save_retrain_data(img, true_digit)
    
    # Thử huấn luyện lại nếu đủ dữ liệu
    success, message, history = retrain_model()
    
    # Vẽ biểu đồ nếu huấn luyện thành công
    accuracy_plot, loss_plot = None, None
    if success and history is not None:
        accuracy_plot, loss_plot = plot_training_history(history)
    
    # Dự đoán lại sau khi huấn luyện
    new_prediction, _ = predict_digit(file_path)
    
    return render_template('result.html', results=[{
        'filename': filename,
        'prediction': new_prediction,
        'image_path': url_for('uploaded_file', filename=filename)
    }], message=f"Added to retraining data ({num_samples} samples). {message}",
    accuracy_plot=accuracy_plot, loss_plot=loss_plot)

if __name__ == '__main__':
    app.run(debug=True)