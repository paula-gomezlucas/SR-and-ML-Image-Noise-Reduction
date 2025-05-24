# SR-and-ML-Image-Noise-Reduction
Stochastic Resonance and Machine Learning Based Noise Reduction Framework

1. Instalación de dependencias:

pip install -r requirements.txt

2. No es necesario ejecutar SExtractor:
Los datos de referencia (catálogo .cat) ya están incluidos y el pipeline usa directamente esos resultados.
Si quieres ejecutar el software igualmente, se necesita usar Linux o WSL en su defecto, pero 
la referencia ya está procesada (data\benchmark\hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.cat 
es el fichero resultante)

3. Para visualizar los resultados del modelo de detección:
Ejecutar este script:

python src/insight.py

Este genera mapas de calor y superposiciones (output_visualizations/) mostrando dónde el modelo detecta objetos.

4. No requiere GPU ni conexión a internet.
El modelo ya está entrenado y guardado en checkpoints/supervised_model.pt.