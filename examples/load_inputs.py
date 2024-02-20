from scipy.io import loadmat

# Replace 'your_file.mat' with the path to your MATLAB file
mat_file = loadmat('inputs/Modulation_connection_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform.mat')
mat_file1 =  loadmat('inputs/GSNR_connection_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform.mat')
mat_file3 =  loadmat('inputs/All_connections_Profile_JPN12_k7SP_CHBFullyLoaded_SCL_Uniform.mat')
# Access variables from the loaded mat file
variable_name = mat_file['variable_name']

# Do something with the variable
print(variable_name)