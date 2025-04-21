import random


# Array from which we will select random items
# array = ["Gr 1", "Gr 2", "Gr 3", "Gr 4","Gr 5", "Gr 6", "Gr 7", "Gr 10", "Gr 11", "Gr 12", "Gr 13"]
array = ["Gr 1", "Gr 2", "Gr 3", "Gr 5", "Gr 6", "Gr 10", "Gr 11", "Gr 12", "Gr 13"]
# Randomly select 3 items from the array
random_items = random.sample(array, 3)

# Print the selected items
print("Randomly selected groups:", random_items)