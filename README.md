Este proyecto implementa una red neuronal desde cero utilizando NumPy, sin utilizar frameworks como TensorFlow o Keras. Se entrena para clasificar datos generados aleatoriamente mediante distribuciones gaussianas.

📦 Instalación

Asegúrate de tener las dependencias instaladas con pip install numpy matplotlib scikit-learn.

🚀 Uso

Ejecuta el script principal con python main.py.

🛠️ Tecnologías

Este proyecto usa Python, NumPy, Matplotlib y Scikit-learn.

📄 Descripción del Código

El código genera un conjunto de datos de dos clases con make_gaussian_quantiles(), define funciones de activación como ReLU y sigmoide, y una función de pérdida basada en MSE. Luego inicializa los pesos de la red, la entrena con backpropagation y gradiente descendente, y finalmente evalúa los resultados mostrando una gráfica de clasificación.