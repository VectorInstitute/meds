import json
import re

def categorize_lab_values(lab_values):
    numeric_pattern = re.compile(r"^\d+(\.\d+)?$")
    only_numeric = []
    only_categorical = []
    both = []

    for lab_code, values in lab_values.items():
        has_numeric = any(numeric_pattern.match(v) for v in values if v)
        has_categorical = any(not numeric_pattern.match(v) for v in values)
        
        if has_numeric and has_categorical:
            both.append(lab_code)
        elif has_numeric:
            only_numeric.append(lab_code)
        else:
            only_categorical.append(lab_code)

    return only_numeric, only_categorical, both

def filter_and_get_min_max(lab_values, lab_codes):
    numeric_pattern = re.compile(r"^\d+(\.\d+)?$")
    result = {}

    for lab_code in lab_codes:
        numeric_values = [float(v) for v in lab_values[lab_code] if v and numeric_pattern.match(v)]
        if numeric_values:
            result[lab_code] = {
                "min": min(numeric_values),
                "max": max(numeric_values),
            }


    return result

def main():
    with open("/mnt/nfs/home/alinoorim/odyssey_gemini/lab_values.json", "r") as f:
        labs = json.load(f)
    
    only_numeric, only_categorical, both = categorize_lab_values(labs)
    
    all_labs = filter_and_get_min_max(labs, only_numeric + both)
    
    with open("./filtered_lab_values.json", "w") as f:
        json.dump(all_labs, f, indent=4)
        
if __name__ == "__main__":
    main()