from preprocess import Forest_Transformer
import pickle
if __name__ == "__main__":
    scaler = Forest_Transformer()

    with open ('preprocessor.pkl', 'wb') as f:
        pickle.dump(scaler, f)