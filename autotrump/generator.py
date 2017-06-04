from util.utilities import *


class Generator(object):
    def __init__(self, decoder):
        self.decoder = decoder

    def generate(self, prime_str='I ', predict_len=120, temperature=0.5):
        hidden = self.decoder.init_hidden()
        prime_input = char_tensor(prime_str)
        predicted = prime_str
        for p in range(len(prime_str) - 1):
            _, hidden = self.decoder(prime_input[p], hidden)
        inp = prime_input[-1]
        for p in range(predict_len):
            output, hidden = self.decoder(inp, hidden)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_i]
            predicted += predicted_char
            inp = char_tensor(predicted_char)
        return predicted
