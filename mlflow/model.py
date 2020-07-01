class simple_model():
    def __init__(self, args) -> None:
        self.args = args
    
    def inference(self, input):
        print(f'Inference on simple model: {input}')
        return 42 

def load_simple_model(args):
    print(f'Loading simple model with args: {args}')
    new_model = simple_model(args)
    return new_model
