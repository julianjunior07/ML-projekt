# ML-projekt
repo projektu z Uczenia Maszyn

**w plikach należy zmienić ścieżki do plików z danymi na własne**

pliki algorytmów znajdują się w folderze \models

wyniki działania algorytmu dostępne są w formie graficznej PNG i tekstowej w folderach _figures_9, _figures_26 oraz imgs

aby uruchomić:
- model predykcji szeregu czasowego, uruchomić plik model.py
- model autoencoderów do wykrywania anomalii, uruchomić plik encoder.py

Predykcja wartości bitrate robione zarówno dla MAPE i MSE jako loss function w modelu:
parametry:
int9:
- ratio = 0.8 #podział danych na treningowe/testowe (80% całego pliku to dane treningowe)
- liczba próbek: train - 11000, test - 3850
- liczba epoch: 30
- input_size = 1
- num_layers = 2
- hidden_size = 64
- batch_size = 16

us26:
- ratio = 0.8 #podział danych na treningowe/testowe (80% całego pliku to dane treningowe)
- liczba próbek: train - 4000, test - 3850
- liczba epoch: 30
- input_size = 1
- num_layers = 2
- hidden_size = 64
- batch_size = 16
