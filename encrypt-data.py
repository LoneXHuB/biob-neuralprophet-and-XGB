import csv
from cryptography.fernet import Fernet
import os

# Generate a key (use this only once, save the key for decryption)
def generate_key():
    key = Fernet.generate_key()
    with open('key.key', 'wb') as key_file:
        key_file.write(key)
    print("Key saved to 'key.key'")

# Load the key
def load_key():
    with open('BB.key', 'rb') as key_file:
        return key_file.read()

# Encrypt CSV data to a binary file
def encrypt_csv(input_csv, output_bin):
    key = load_key()
    cipher = Fernet(key)
    with open(input_csv, 'r') as csv_file:
        data = csv_file.read().encode()  # Read and encode the data
    encrypted_data = cipher.encrypt(data)
    with open(output_bin, 'wb') as bin_file:
        bin_file.write(encrypted_data)
    print(f"Data encrypted and saved to '{output_bin}'")

# Decrypt binary file back to CSV format
def decrypt_to_csv(input_bin, output_csv):
    key = load_key()
    cipher = Fernet(key)
    with open(input_bin, 'rb') as bin_file:
        encrypted_data = bin_file.read()
    decrypted_data = cipher.decrypt(encrypted_data).decode()  # Decode back to text
    with open(output_csv, 'w') as csv_file:
        csv_file.write(decrypted_data)
    print(f"Data decrypted and saved to '{output_csv}'")

# Example Usage
if __name__ == "__main__":

    print("Current working directory:", os.getcwd())
    os.chdir(os.path.dirname(__file__))

    #if not os.path.exists("BB.key"):
    #    generate_key()

    # Encrypt a CSV file
    #encrypt_csv('data.csv', 'data_encrypted.bin')

    # Decrypt the binary file back to CSV
    decrypt_to_csv('data\enc_4\enc_Mouleuse_20240819.bin', 'data\dec_4\Mouleuse_20240819.csv')
    decrypt_to_csv('data\enc_4\enc_Mouleuse_20240829.bin', 'data\dec_4\Mouleuse_20240829.csv')
    decrypt_to_csv('data\enc_4\enc_Mouleuse_20240917.bin', 'data\dec_4\Mouleuse_20240917.csv')
   
