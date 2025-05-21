from models.random_forest import train_RFC
from models.xgboost import train_XGB
from models.neural_network import train_NN

def main():
    model_no = input("""Train classification ML models for molecular toxicity.
Select model to train:
    1) Random Forest Classifier
    2) XGBoost
    3) Neural Network\n""")
    
    models = [
        ("Random Forest Classifier", train_RFC),
        ("XGBoost", train_XGB),
        ("Neural Network", train_NN),
    ]
    
    try:
        model = models[int(model_no)-1]
        model[1]()
    except:
        raise Exception("Unsupported model. Please select another one.") 

if __name__ == "__main__":
    main()
