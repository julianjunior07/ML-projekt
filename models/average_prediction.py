import re
import statistics

def extract_float_values(file_path):
    float_values = []
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Using regular expression to find float values
        float_pattern = r'\b\d+\.\d+\b'
        matches = re.findall(float_pattern, content)
        
        # Converting matched strings to float and appending to the list
        float_values.extend(map(float, matches))
    
    return float_values

# 16 MAPE
file_path = 'D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_9\error_values_16_MAPE_fn.txt'  # Replace with the path to your text file
float_values_1 = extract_float_values(file_path)

# usunięcie wartosci powyzej 1000% (błędy)
filtered_values_1 = [value for value in float_values_1 if value <= 1000]
average_1 = statistics.mean(filtered_values_1)

file_averages_prediction = open("D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_averages.txt", "a")
file_averages_prediction.write("16_MAPE_fn average: "+ str(average_1) + "\n")


# 16 MSE
file_path = 'D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_9\error_values_16_MSE_fn.txt'  # Replace with the path to your text file
float_values_2 = extract_float_values(file_path)

# usunięcie wartosci powyzej 1000% (błędy)
filtered_values_2 = [value for value in float_values_2 if value <= 1000]
average_2 = statistics.mean(filtered_values_2)

file_averages_prediction = open("D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_averages.txt", "a")
file_averages_prediction.write("16_MSE_fn average: "+ str(average_2) + "\n")


# 49 MAPE
file_path = 'D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_26\error_values_49_MAPE_fn.txt'  # Replace with the path to your text file
float_values_3 = extract_float_values(file_path)

# usunięcie wartosci powyzej 1000% (błędy)
filtered_values_3 = [value for value in float_values_3 if value <= 1000]
average_3 = statistics.mean(filtered_values_3)

file_averages_prediction = open("D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_averages.txt", "a")
file_averages_prediction.write("49_MAPE_fn average: "+ str(average_3) + "\n")


# 49 MSE
file_path = 'D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_figures_26\error_values_49_MSE_fn.txt'  # Replace with the path to your text file
float_values_4 = extract_float_values(file_path)

# usunięcie wartosci powyzej 1000% (błędy)
filtered_values_4 = [value for value in float_values_4 if value <= 1000]
average_4 = statistics.mean(filtered_values_4)

file_averages_prediction = open("D:\Polibudka\Magister\Sezon 2\Proj Sieci Komp i ML\ML\_kod_github\ML-projekt\_averages.txt", "a")
file_averages_prediction.write("49_MSE_fn average: "+ str(average_4) + "\n")
