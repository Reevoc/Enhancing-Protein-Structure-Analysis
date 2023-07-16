def append_last_element(dict1, dict2):
    for key in dict1:
        if key in dict2:
            dict2[key].append(dict1[key][-1])


# Example dictionaries
dict1 = {"key1": [1, 2, 3, 4], "key2": [5, 6, 7]}
dict2 = {"key1": [8, 9], "key3": [10, 11]}

# Append the last element from dict1 to dict2 with matching keys
append_last_element(dict1, dict2)

# Print the updated dict2
print(dict2)
