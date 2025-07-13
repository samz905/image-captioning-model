import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        
        self.A = nn.Linear(attention_dim, 1)
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)
        w_ah = self.W(hidden_state)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))
        
        attention_weights = self.A(combined_states)
        attention_weights = attention_weights.squeeze(2)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        context_vector = features * attention_weights.unsqueeze(2)
        context_vector = context_vector.sum(dim=1)
        
        return attention_weights, context_vector

# Attention Decoder
class DecoderRNN(nn.Module):
    def __init__(
        self,
        embed_size,
        vocab_size,
        attention_dim,
        encoder_dim,
        decoder_dim,
        drop_prob=0.3,
    ):
        super().__init__()

        # save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        nn.init.xavier_uniform_(self.init_h.weight)
        nn.init.xavier_uniform_(self.init_c.weight)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions, teacher_forcing_ratio=1.0):
        import random

        # vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        # get the seq length to iterate
        seq_length = len(captions[0]) - 1  # Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(device)

        input_word = embeds[:, 0]

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((input_word, context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))
            preds[:, s] = output
            alphas[:, s] = alpha

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing and s < seq_length - 1:
                input_word = embeds[:, s + 1]
            else:
                predicted_word_idx = output.argmax(dim=1)
                input_word = self.embedding(predicted_word_idx)

        return preds, alphas

    def generate_caption(self, features, max_len=20, vocab=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        # starting input
        word = torch.tensor(vocab.stoi["<SOS>"]).view(1, -1).to(device)
        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alpha, context = self.attention(features, h)

            # store the apla score
            alphas.append(alpha.cpu().detach().numpy())

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)

            # select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            # save the generated word
            captions.append(predicted_word_idx.item())

            # end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            # send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        # covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions], alphas

    def generate_caption_beam(self, features, beam_size=3, max_len=20, vocab=None):
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)
        
        sequences = [[vocab.stoi["<SOS>"]]]
        scores = [0.0]
        
        for i in range(max_len):
            all_candidates = []
            
            for j, seq in enumerate(sequences):
                if seq[-1] == vocab.stoi["<EOS>"]:
                    all_candidates.append((scores[j], seq))
                    continue
                    
                word = torch.tensor([seq[-1]]).view(1, -1).to(device)
                embeds = self.embedding(word)
                
                alpha, context = self.attention(features, h)
                lstm_input = torch.cat((embeds[:, 0], context), dim=1)
                h, c = self.lstm_cell(lstm_input, (h, c))
                output = self.fcn(self.drop(h))
                output = torch.log_softmax(output, dim=1)
                
                top_scores, top_words = output.topk(beam_size)
                
                for k in range(beam_size):
                    word_idx = top_words[0][k].item()
                    score = scores[j] + top_scores[0][k].item()
                    candidate = seq + [word_idx]
                    all_candidates.append((score, candidate))
            
            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            sequences = [seq for score, seq in ordered[:beam_size]]
            scores = [score for score, seq in ordered[:beam_size]]
            
            if all(seq[-1] == vocab.stoi["<EOS>"] for seq in sequences):
                break
        
        best_sequence = sequences[0]
        return [vocab.itos[idx] for idx in best_sequence], []

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

class Model(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        
    def forward(self, images, captions, teacher_forcing_ratio=1.0):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, teacher_forcing_ratio)
        return outputs
