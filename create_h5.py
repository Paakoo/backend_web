import os
import h5py
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

def create_face_embeddings(base_folder, output_file):
    embeddings = {}
    
    for username in os.listdir(base_folder):
        user_path = os.path.join(base_folder, username)
        if not os.path.isdir(user_path):
            continue
            
        print(f"\nProcessing user: {username}")
        user_embeddings = []
        
        for img_name in tqdm(os.listdir(user_path)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(user_path, img_name)
                try:
                    embedding_obj = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet512",
                        detector_backend="retinaface",
                        enforce_detection=True,
                        align=True
                    )
                    
                    # Ensure we get a single embedding array
                    if isinstance(embedding_obj, list):
                        embedding_obj = embedding_obj[0]
                    
                    # Extract and reshape embedding to ensure consistency
                    embedding_vector = np.array(embedding_obj['embedding'], dtype=np.float32)
                    embedding_vector = embedding_vector.reshape(1, -1)  # Reshape to 2D array
                    
                    if embedding_vector.shape[1] == 512:
                        user_embeddings.append(embedding_vector)
                        print(f"{img_name}: shape {embedding_vector.shape}")
                    else:
                        print(f"Skipping {img_name} - wrong shape: {embedding_vector.shape}")
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        if user_embeddings:
            try:
                # Concatenate all embeddings for this user
                user_data = np.concatenate(user_embeddings, axis=0)
                print(f"\nFinal shape for {username}: {user_data.shape}")
                embeddings[username] = user_data
            except ValueError as e:
                print(f"Error concatenating embeddings for {username}: {e}")
                continue
    
    # Save to H5 file
    with h5py.File(output_file, 'w') as hf:
        for username, user_data in embeddings.items():
            hf.create_dataset(f"user_{username}", data=user_data)
    
    print(f"\nEmbeddings saved to {output_file}")

if __name__ == "__main__":
    BASE_FOLDER = "databaru"
    OUTPUT_FILE = "ds_model_face.h5"
    create_face_embeddings(BASE_FOLDER, OUTPUT_FILE)