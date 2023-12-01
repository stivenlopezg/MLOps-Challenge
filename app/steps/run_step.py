import sys
from app.config.config import logger
from app.steps.training import training
from app.steps.inference import inference


def main():
    print("Argumentos de la línea de comandos:", sys.argv)
    if len(sys.argv) < 4:
        logger.info("No se proporciono el tipo de ejecución [train, prediction], tracking_uri y experiment_name")

    execution_type = sys.argv[1].lower()
    tracking_uri = sys.argv[2]
    experiment_name = sys.argv[3]

    if execution_type == "train":
        training.execute_train(tracking_uri=tracking_uri, experiment_name=experiment_name)
    elif execution_type == "prediction":
        inference.execute_inference(tracking_uri=tracking_uri, experiment_name=experiment_name)
    else:
        print("Ejecución invalida. Usa 'train' o 'prediction'")
        sys.exit(1)


if __name__ == "__main__":
    main()
