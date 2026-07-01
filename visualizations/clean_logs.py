def clean_logs(input_file, output_file):
    """
    Reads a text file with messy logs and extracts only
    the training metrics information.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            # Check if the line contains our keyword
            if "epoch:" in line:
                # Slice the string to keep only the useful part
                start_idx = line.find("epoch:")
                clean_line = line[start_idx:].strip()
                
                # Write the cleaned line to the new file
                f_out.write(clean_line + "\n")

# --- Execution ---
# 1. Save your original text in a file named 'logs.txt' in the same folder.
# 2. The script will create a 'clean_logs.txt' file with the results.

clean_logs('visualizations/pretraining-console.txt', 'visualizations/clean_logs.txt')
print("Cleanup complete! Check the clean_logs.txt file.")