import wandb


class Logger():

    def __init__(self, run) -> None:
        self.run = run

    def logger(self, dict):
        self.run.log(dict)
