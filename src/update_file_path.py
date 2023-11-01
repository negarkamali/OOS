import os

# Directory structure
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')

# Function to update file paths in a Python file
def update_file_paths(file_path):
    with open(file_path, 'r') as f:
        content = f.readlines()

    new_content = []
    import_os_added = False
    for line in content:
        if 'import os' in line:
            import_os_added = True

        if '/Users/Negar/Library/CloudStorage/GoogleDrive-nkamal5@uic.edu/My Drive/2nd PhD/Research/With Jessica/Conformal Prediction/Conformlab/OOS_Intent_Classification/' in line:
            # Replace the absolute path with a relative path
            updated_line = line.replace('/Users/Negar/Library/CloudStorage/GoogleDrive-nkamal5@uic.edu/My Drive/2nd PhD/Research/With Jessica/Conformal Prediction/Conformlab/OOS_Intent_Classification/', 'os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ')
            new_content.append(updated_line)
        else:
            new_content.append(line)

    if not import_os_added:
        new_content.insert(0, 'import os\n')

    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.writelines(new_content)

# Loop through all Python files in the src directory
for file_name in os.listdir(src_dir):
    if file_name.endswith('.py'):
        file_path = os.path.join(src_dir, file_name)
        update_file_paths(file_path)
        print(f'Updated paths in {file_name}')
